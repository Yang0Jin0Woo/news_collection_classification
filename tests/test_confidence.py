from news_classifier.classifiers.confidence import confidence_level, is_ambiguous


def test_confidence_high():
    assert confidence_level(0.85, 0.20) == "높음"


def test_confidence_medium():
    assert confidence_level(0.55, 0.06) == "보통"


def test_confidence_low():
    assert confidence_level(0.40, 0.01) == "낮음"


def test_is_ambiguous_by_score():
    assert is_ambiguous(0.30, 0.20) is True


def test_is_ambiguous_by_margin():
    assert is_ambiguous(0.90, 0.01) is True
