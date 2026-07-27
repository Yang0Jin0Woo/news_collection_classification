import pytest
import sqlite3
from types import SimpleNamespace

from news_classifier.classifiers.postprocessor import ClassificationPostProcessor
from news_classifier.classifiers.rule_engine import RuleEngine
from news_classifier.cli import (
    build_parser,
    resolve_keyword,
    run_cli,
    run_dashboard,
)
from news_classifier.models import (
    ModelPrediction,
    NewsItem,
    PipelineError,
    PipelineResult,
    PipelineStatistics,
    PipelineStatus,
)
from news_classifier.rules.default_rules import RULES
from news_classifier.storage.csv_store import CsvNewsStore
from news_classifier.utils.validation import ValidationError


def test_collect_keyword_is_optional():
    parser = build_parser()

    args = parser.parse_args(["collect"])

    assert args.keyword is None


@pytest.mark.parametrize("limit", ["0", "101", "not-a-number"])
def test_collect_rejects_invalid_limit(limit):
    parser = build_parser()

    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["collect", "--limit", limit])

    assert exc_info.value.code == 2


@pytest.mark.parametrize("limit", ["1", "100"])
def test_collect_accepts_limit_boundaries(limit):
    args = build_parser().parse_args(["collect", "--limit", limit])

    assert args.limit == int(limit)


def test_sqlite_option_accepts_default_or_explicit_path():
    parser = build_parser()

    default_path_args = parser.parse_args(["collect", "--sqlite"])
    explicit_path_args = parser.parse_args(
        ["collect", "--sqlite", "custom.db"]
    )

    assert default_path_args.sqlite == ""
    assert explicit_path_args.sqlite == "custom.db"


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


def successful_result() -> PipelineResult:
    item = NewsItem(
        keyword="AI",
        title="AI 반도체 기술 개발",
        link="https://example.com/news",
    )
    classified = ClassificationPostProcessor(RuleEngine(RULES)).process(
        item,
        ModelPrediction("기술개발", 0.90, 0.50),
    )
    return PipelineResult(
        status=PipelineStatus.SUCCESS,
        results=[classified],
        statistics=PipelineStatistics(
            requested_limit=10,
            collected_count=1,
            deduplicated_count=1,
            classified_count=1,
        ),
    )


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


def test_cli_uses_configured_default_sqlite_path(
    monkeypatch, tmp_path, capsys
):
    result = successful_result()
    monkeypatch.setattr(
        "news_classifier.service.build_pipeline",
        lambda settings: StubPipeline(result),
    )
    db_path = tmp_path / "configured.db"
    csv_path = tmp_path / "result.csv"
    monkeypatch.setenv("NEWS_OUTPUT_DB", str(db_path))

    exit_code = run_cli(
        [
            "collect",
            "--keyword",
            "AI",
            "--csv",
            str(csv_path),
            "--sqlite",
        ]
    )

    assert exit_code == 0
    assert db_path.exists()
    with sqlite3.connect(db_path) as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM classified_news"
        ).fetchone()[0] == 1
    assert f"SQLite saved: {db_path}" in capsys.readouterr().out


def test_cli_explicit_sqlite_path_overrides_configured_default(
    monkeypatch, tmp_path
):
    result = successful_result()
    monkeypatch.setattr(
        "news_classifier.service.build_pipeline",
        lambda settings: StubPipeline(result),
    )
    configured_path = tmp_path / "configured.db"
    explicit_path = tmp_path / "explicit.db"
    csv_path = tmp_path / "result.csv"
    monkeypatch.setenv("NEWS_OUTPUT_DB", str(configured_path))

    exit_code = run_cli(
        [
            "collect",
            "--keyword",
            "AI",
            "--csv",
            str(csv_path),
            "--sqlite",
            str(explicit_path),
        ]
    )

    assert exit_code == 0
    assert explicit_path.exists()
    assert not configured_path.exists()


def test_cli_skips_sqlite_without_option(monkeypatch, tmp_path):
    result = successful_result()
    monkeypatch.setattr(
        "news_classifier.service.build_pipeline",
        lambda settings: StubPipeline(result),
    )
    configured_path = tmp_path / "configured.db"
    csv_path = tmp_path / "result.csv"
    monkeypatch.setenv("NEWS_OUTPUT_DB", str(configured_path))

    exit_code = run_cli(
        [
            "collect",
            "--keyword",
            "AI",
            "--csv",
            str(csv_path),
        ]
    )

    assert exit_code == 0
    assert not configured_path.exists()


def test_cli_dashboard_uses_streamlit_launcher(monkeypatch):
    monkeypatch.setattr(
        "news_classifier.cli.run_dashboard",
        lambda: 7,
    )

    assert run_cli(["dashboard"]) == 7


def test_dashboard_launcher_runs_streamlit_module(monkeypatch):
    captured = {}

    def fake_run(command, check):
        captured["command"] = command
        captured["check"] = check
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("news_classifier.cli.subprocess.run", fake_run)

    assert run_dashboard() == 0
    assert captured["command"][1:4] == [
        "-m",
        "streamlit",
        "run",
    ]
    assert captured["command"][4].endswith("dashboard_streamlit.py")
    assert captured["check"] is False
