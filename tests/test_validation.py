import pytest
from news_classifier.utils.validation import (
    ValidationError,
    require_non_blank,
    validate_limit,
)


def test_require_non_blank_returns_stripped_value():
    assert require_non_blank("  AI  ", "keyword") == "AI"


def test_require_non_blank_raises():
    with pytest.raises(ValidationError):
        require_non_blank(" ", "keyword")


def test_validate_limit_accepts_value_in_range():
    assert validate_limit(20) == 20


@pytest.mark.parametrize("limit", [0, 101])
def test_validate_limit_rejects_value_outside_range(limit):
    with pytest.raises(ValidationError):
        validate_limit(limit)
