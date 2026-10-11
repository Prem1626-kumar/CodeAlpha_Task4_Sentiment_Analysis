"""
app.py
Streamlit dashboard for the CodeAlpha Task 4 Sentiment Analysis project.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# ── Path constants ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CSV  = PROJECT_ROOT / "data" / "raw" / "reviews.csv"
FIGURES_DIR  = PROJECT_ROOT / "outputs" / "figures"
REPORT_PATH  = PROJECT_ROOT / "outputs" / "reports" / "sentiment_analysis_report.md"

SENTIMENT_COLORS = {
    "positive": "#2ecc71",
    "negative": "#e74c3c",
    "neutral":  "#3498db",
}

# ── Page config ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    page_icon="💬",
    layout="wide",
)


# ── Helpers ─────────────────────────────────────────────────────────────────────

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8")
    return df


def run_vader(text: str) -> dict:
    """Run VADER on a single string and return scores + label."""
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        sia = SentimentIntensityAnalyzer()
        scores = sia.polarity_scores(text)
        compound = scores["compound"]
        if compound >= 0.05:
            label = "positive"
        elif compound <= -0.05:
            label = "negative"
        else:
            label = "neutral"
        return {"label": label, "compound": compound, **scores}
    except ImportError:
        return {"label": "error", "compound": 0.0,
                "error": "vaderSentiment not installed"}


def sentiment_badge(label: str) -> str:
    """Return an HTML badge string for a sentiment label."""
    colors = {
        "positive": ("✅", "#d4edda", "#155724"),
        "negative": ("❌", "#f8d7da", "#721c24"),
        "neutral":  ("➖", "#d1ecf1", "#0c5460"),
        "error":    ("⚠", "#fff3cd", "#856404"),
    }
    icon, bg, fg = colors.get(label, ("?", "#eee", "#333"))
    return (
        f'<span style="background:{bg};color:{fg};padding:6px 14px;'
        f'border-radius:6px;font-weight:bold;font-size:1.1em;">'
        f"{icon} {label.capitalize()}</span>"
    )


# ── Sidebar ──────────────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Settings")
st.sidebar.markdown("---")

use_upload = st.sidebar.checkbox("Upload a custom CSV", value=False)
if use_upload:
    uploaded = st.sidebar.file_uploader("Upload reviews CSV", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        st.sidebar.success(f"Loaded {len(df)} rows from upload.")
    else:
        st.sidebar.info("No file uploaded — using default dataset.")
        df = load_data(str(DEFAULT_CSV)) if DEFAULT_CSV.exists() else pd.DataFrame()
else:
    if DEFAULT_CSV.exists():
        df = load_data(str(DEFAULT_CSV))
    else:
        df = pd.DataFrame()

st.sidebar.markdown("---")
st.sidebar.info(
    "⚠️ **Disclaimer**: The default dataset is **synthetic** and does not "
    "represent real customer reviews."
)

# ── Main layout ──────────────────────────────────────────────────────────────────
st.title("💬 Sentiment Analysis Dashboard")
st.caption("CodeAlpha Data Analytics Internship — Task 4")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔍 Analyze Text", "🖼 Visualizations", "📄 Report"])

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ════════════════════════════════════════════════════════════════════════════════
with tab1:
    if df.empty:
        st.warning(
            "No dataset loaded. Run `python data/generate_dataset.py` to create the default dataset."
        )
    else:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Reviews", f"{len(df):,}")
        col2.metric("Categories", df["product_category"].nunique() if "product_category" in df.columns else "N/A")
        col3.metric("Rating Range", f"1 – {int(df['rating'].max())}" if "rating" in df.columns else "N/A")
        col4.metric("Date Range",
            f"{str(df['date'].min())[:7]} → {str(df['date'].max())[:7]}"
            if "date" in df.columns else "N/A"
        )

        st.markdown("---")

        col_bar, col_pie = st.columns(2)

        with col_bar:
            st.subheader("Sentiment Counts")
            if "sentiment" in df.columns:
                counts = df["sentiment"].value_counts().reindex(
                    ["positive", "negative", "neutral"]
                ).fillna(0).astype(int)
                fig_bar, ax = plt.subplots(figsize=(6, 4))
                ax.bar(
                    counts.index,
                    counts.values,
                    color=[SENTIMENT_COLORS[s] for s in counts.index],
                    edgecolor="white",
                )
                ax.set_xlabel("Sentiment")
                ax.set_ylabel("Count")
                ax.set_title("Sentiment Distribution")
                ax.spines[["top", "right"]].set_visible(False)
                fig_bar.tight_layout()
                st.pyplot(fig_bar)
                plt.close(fig_bar)
            else:
                st.warning("Column 'sentiment' not found in dataset.")

        with col_pie:
            st.subheader("Sentiment Percentages")
            if "sentiment" in df.columns:
                counts_pie = df["sentiment"].value_counts()
                fig_pie, ax2 = plt.subplots(figsize=(6, 4))
                ax2.pie(
                    counts_pie.values,
                    labels=counts_pie.index,
                    colors=[SENTIMENT_COLORS.get(s, "#999") for s in counts_pie.index],
                    autopct="%1.1f%%",
                    startangle=140,
                )
                ax2.set_title("Sentiment %")
                fig_pie.tight_layout()
                st.pyplot(fig_pie)
                plt.close(fig_pie)

        st.markdown("---")
        st.subheader("Dataset Sample")
        st.dataframe(df.head(20), use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — ANALYZE TEXT
# ════════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("🔍 Classify a Review")
    st.markdown(
        "Type or paste any product review text below. VADER will classify it instantly "
        "without any model training."
    )

    user_text = st.text_area(
        "Review text",
        placeholder="e.g. This product is fantastic! Works perfectly and arrived on time.",
        height=120,
    )

    if st.button("Analyse Sentiment", type="primary"):
        if not user_text.strip():
            st.warning("Please enter some text first.")
        else:
            with st.spinner("Classifying ..."):
                result = run_vader(user_text.strip())

            if "error" in result:
                st.error(f"Classification error: {result['error']}")
            else:
                st.markdown("**Sentiment:**")
                st.markdown(sentiment_badge(result["label"]), unsafe_allow_html=True)
                st.markdown("---")
                col_c, col_p, col_n, col_neu = st.columns(4)
                col_c.metric("Compound Score", f"{result['compound']:.4f}")
                col_p.metric("Positive", f"{result.get('pos', 0):.3f}")
                col_n.metric("Negative", f"{result.get('neg', 0):.3f}")
                col_neu.metric("Neutral",   f"{result.get('neu', 0):.3f}")

                st.info(
                    "VADER compound score: ≥ 0.05 → positive | ≤ −0.05 → negative | else → neutral"
                )

    st.markdown("---")
    st.subheader("Try these examples")
    examples = [
        ("Positive", "Absolutely love this product! Works perfectly and delivery was super fast."),
        ("Negative", "Terrible quality. Broke after one day and customer service was no help at all."),
        ("Neutral",  "It does the job. Nothing special but nothing wrong either."),
        ("Tricky",   "Not bad for the price, though I expected a bit more."),
    ]
    for label, text in examples:
        with st.expander(f"{label}: "{text[:60]}...""):
            r = run_vader(text)
            st.markdown(sentiment_badge(r["label"]), unsafe_allow_html=True)
            st.write(f"Compound: **{r['compound']:.4f}**")


# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — VISUALIZATIONS
# ════════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("🖼 Generated Visualizations")

    if not FIGURES_DIR.exists():
        st.warning(
            f"`{FIGURES_DIR}` does not exist. "
            "Run `python run_analysis.py` first to generate charts."
        )
    else:
        charts = sorted(FIGURES_DIR.glob("*.png"))
        if not charts:
            st.warning("No charts found. Run `python run_analysis.py` to generate them.")
        else:
            st.success(f"Found {len(charts)} chart(s).")
            # Display in a 2-column grid
            for i in range(0, len(charts), 2):
                cols = st.columns(2)
                for j, col in enumerate(cols):
                    if i + j < len(charts):
                        chart_path = charts[i + j]
                        col.image(str(chart_path), caption=chart_path.stem, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 4 — REPORT
# ════════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("📄 Analysis Report")

    if not REPORT_PATH.exists():
        st.warning(
            f"Report not found at `{REPORT_PATH}`. "
            "Run `python run_analysis.py` to generate it."
        )
    else:
        report_text = REPORT_PATH.read_text(encoding="utf-8")
        st.markdown(report_text, unsafe_allow_html=False)

        # Download button
        st.download_button(
            label="⬇ Download Report (.md)",
            data=report_text.encode("utf-8"),
            file_name="sentiment_analysis_report.md",
            mime="text/markdown",
        )
