from news_classifier.utils.text import clean_text, normalize_description, strip_html, normalize_key


def test_clean_text_collapses_space():
    assert clean_text("  A\n\tB   C  ") == "A B C"


def test_strip_html_removes_tags():
    assert strip_html("<p>Hello <b>World</b></p>") == "Hello World"


def test_normalize_description_removes_noise():
    assert normalize_description("[단독] AI 반도체 신제품 출시 (종합)") == "AI 반도체 신제품 출시"


def test_normalize_description_removes_too_short_noise():
    assert normalize_description("[단독] 속보") == ""


def test_normalize_key_removes_space_and_lowercase():
    assert normalize_key("  AI 반도체 News ") == "ai반도체news"
