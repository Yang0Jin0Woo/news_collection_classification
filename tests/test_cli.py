import pytest

from news_classifier.cli import build_parser, resolve_keyword, run_cli
from news_classifier.models import (
    PipelineError,
    PipelineResult,
    PipelineStatistics,
    PipelineStatus,
)
from news_classifier.storage.csv_store import CsvNewsStore
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


class StubPipeline:
    def __init__(self, result: PipelineResult):
        self.result = result

    def run(self, keyword: str, limit: int, enrich_content: bool):
        return self.result


def test_cli_returns_nonzero_and_does_not_save_failed_result(
    monkeypatch, tmp_path, capsys
):
    result = PipelineResult(
        status=PipelineStatus.MODEL_ERROR,
        errors=[
            PipelineError(
                stage="CLASSIFICATION",
                code="MODEL_INFERENCE_FAILURE",
                message="모델 로드 실패",
            )
        ],
        statistics=PipelineStatistics(requested_limit=10),
    )
    monkeypatch.setattr(
        "news_classifier.service.build_pipeline",
        lambda settings: StubPipeline(result),
    )
    path = tmp_path / "failed.csv"

    exit_code = run_cli(
        ["collect", "--keyword", "AI", "--csv", str(path)]
    )

    assert exit_code == 1
    assert not path.exists()
    captured = capsys.readouterr()
    assert "Pipeline status: MODEL_ERROR" in captured.out
    assert "실패한 실행 결과는 저장하지 않았습니다." in captured.err


def test_cli_writes_header_csv_for_no_results(monkeypatch, tmp_path, capsys):
    result = PipelineResult(
        status=PipelineStatus.NO_RESULTS,
        statistics=PipelineStatistics(requested_limit=10),
    )
    monkeypatch.setattr(
        "news_classifier.service.build_pipeline",
        lambda settings: StubPipeline(result),
    )
    path = tmp_path / "empty.csv"

    exit_code = run_cli(
        ["collect", "--keyword", "AI", "--csv", str(path)]
    )

    assert exit_code == 0
    assert CsvNewsStore(str(path)).load().empty
    output = capsys.readouterr().out
    assert "Pipeline status: NO_RESULTS" in output
    assert "CSV saved with headers only" in output
