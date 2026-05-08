import pytest
from news_classifier.utils.validation import ValidationError, clamp_limit, require_non_blank


def test_require_non_blank_returns_stripped_value():
    assert require_non_blank("  AI  ", "keyword") == "AI"


def test_require_non_blank_raises():
    with pytest.raises(ValidationError):
        require_non_blank(" ", "keyword")


def test_clamp_limit():
    assert clamp_limit(-1) == 1
    assert clamp_limit(200) == 100
    assert clamp_limit(20) == 20
