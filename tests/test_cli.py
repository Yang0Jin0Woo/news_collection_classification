import pytest

from news_classifier.cli import build_parser, resolve_keyword
from news_classifier.utils.validation import ValidationError


def test_collect_keyword_is_optional():
    parser = build_parser()

    args = parser.parse_args(["collect"])

    assert args.keyword is None


def test_resolve_keyword_uses_cli_value():
    assert resolve_keyword("  AI 반도체  ") == "AI 반도체"


def test_resolve_keyword_prompts_when_missing(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda prompt: "AI 반도체")

    assert resolve_keyword(None) == "AI 반도체"


def test_resolve_keyword_rejects_blank_prompt(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda prompt: " ")

    with pytest.raises(ValidationError):
        resolve_keyword(None)
