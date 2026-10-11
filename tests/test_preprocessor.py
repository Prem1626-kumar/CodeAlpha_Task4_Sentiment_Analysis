"""
tests/test_preprocessor.py
Tests for src/preprocessor.py
"""

import pytest
from src.preprocessor import clean_text, preprocess_series


class TestCleanText:

    def test_empty_string_returns_empty(self):
        assert clean_text("") == ""

    def test_whitespace_only_returns_empty(self):
        assert clean_text("   ") == ""
        assert clean_text("\t\n") == ""

    def test_lowercase(self):
        result = clean_text("GREAT Product", remove_stopwords=False)
        assert result == result.lower()

    def test_url_removal(self):
        result = clean_text("Visit https://example.com for more info", remove_stopwords=False)
        assert "http" not in result
        assert "example.com" not in result

    def test_html_tag_removal(self):
        result = clean_text("<b>Bold</b> and <i>italic</i> text", remove_stopwords=False)
        assert "<" not in result
        assert ">" not in result
        assert "bold" in result
        assert "italic" in result

    def test_negation_preserved_with_apostrophe(self):
        # "don't" should survive after cleaning (apostrophe kept)
        result = clean_text("I don't like this", remove_stopwords=False)
        assert "don't" in result or "dont" in result or "don" in result

    def test_negation_word_not_preserved(self):
        # "not" should survive stop-word removal
        result = clean_text("This is not good", remove_stopwords=True)
        assert "not" in result

    def test_no_preserved(self):
        result = clean_text("No way this works", remove_stopwords=True)
        assert "no" in result

    def test_never_preserved(self):
        result = clean_text("I never want this again", remove_stopwords=True)
        assert "never" in result

    def test_punctuation_stripped(self):
        result = clean_text("Hello, world! How are you?", remove_stopwords=False)
        assert "," not in result
        assert "!" not in result
        assert "?" not in result

    def test_stopwords_removed(self):
        # "the", "is", "a" should be removed
        result = clean_text("The product is a great item", remove_stopwords=True)
        assert "the" not in result.split()
        assert "is" not in result.split()

    def test_non_string_input_handled(self):
        # Should not raise; should return cleaned string
        result = clean_text(12345, remove_stopwords=False)
        assert isinstance(result, str)

    def test_long_text(self):
        long_text = "Great product. " * 100
        result = clean_text(long_text, remove_stopwords=True)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_only_urls_returns_empty_or_short(self):
        result = clean_text("https://www.example.com https://foo.bar", remove_stopwords=True)
        # After URL removal and stop-word removal, should be very short or empty
        assert len(result.split()) < 3

    def test_mixed_case_normalised(self):
        r1 = clean_text("GOOD", remove_stopwords=False)
        r2 = clean_text("good", remove_stopwords=False)
        assert r1 == r2


class TestPreprocessSeries:
    def test_series_same_length(self):
        import pandas as pd
        s = pd.Series(["Great product", "Terrible item", "Okay"])
        result = preprocess_series(s, remove_stopwords=True)
        assert len(result) == 3

    def test_nan_becomes_empty_string(self):
        import pandas as pd
        s = pd.Series(["Good", None, float("nan")])
        result = preprocess_series(s, remove_stopwords=False)
        assert result.iloc[1] == ""
        assert result.iloc[2] == ""

    def test_all_results_are_strings(self):
        import pandas as pd
        s = pd.Series(["Text", 123, None, "More text"])
        result = preprocess_series(s)
        assert all(isinstance(v, str) for v in result)
