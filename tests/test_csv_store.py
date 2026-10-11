import json
from pathlib import Path

import pandas as pd
import pytest

import news_classifier.storage.csv_store as csv_store_module

from news_classifier.models import CLASSIFIED_NEWS_COLUMNS, ModelPrediction, NewsItem
from news_classifier.dedup.title_deduplicator import TitleSourceDeduplicator
from news_classifier.classifiers.rule_engine import RuleEngine
from news_classifier.classifiers.postprocessor import ClassificationPostProcessor
from news_classifier.rules.default_rules import DEFAULT_RULE_SET
from news_classifier.storage.csv_store import CsvNewsStore


def test_empty_csv_contains_fixed_headers(tmp_path):
    path = tmp_path / "empty.csv"
    store = CsvNewsStore(str(path))

    store.save([])
    loaded = store.load()

    assert path.exists()
    assert loaded.empty
    assert list(loaded.columns) == CLASSIFIED_NEWS_COLUMNS


def test_csv_preserves_grouped_sources_and_links(tmp_path):
    item = TitleSourceDeduplicator().deduplicate([
        NewsItem(keyword="AI", title="같은 뉴스 제목", source="A", link="https://example.com/1"),
        NewsItem(keyword="AI", title="같은 뉴스 제목", source="A", link="https://example.com/2"),
    ])[0]
    classified = ClassificationPostProcessor(RuleEngine(DEFAULT_RULE_SET)).process(
        item, ModelPrediction("기술개발", 0.8, 0.2)
    )
    store = CsvNewsStore(str(tmp_path / "group.csv"))
    store.save([classified])
    loaded = store.load()
    assert len(loaded) == 1
    assert loaded.iloc[0]["group_article_count"] == 2
    references = json.loads(loaded.iloc[0]["related_articles"])
    assert references == [{"title": "같은 뉴스 제목", "source": "A", "published_at": "", "link": "https://example.com/2"}]


def test_csv_replaces_only_after_complete_write_and_close(tmp_path, monkeypatch):
    path = tmp_path / "result.csv"
    path.write_bytes(b"previous results")
    items = [
        NewsItem(keyword="공통", title="이전 뉴스", link="https://example.com/older", published_at="2026-10-10T12:00:00+00:00"),
        NewsItem(keyword="공통", title="최근 뉴스", link="https://example.com/newer", published_at="2026-10-11T12:00:00+00:00"),
    ]
    processor = ClassificationPostProcessor(RuleEngine(DEFAULT_RULE_SET))
    rows = [processor.process(item, ModelPrediction("기술개발", 0.8, 0.2)) for item in items]
    expected = pd.DataFrame([row.to_row() for row in rows], columns=CLASSIFIED_NEWS_COLUMNS)
    expected = expected.sort_values(by="published_at", ascending=False, na_position="last")
    expected_bytes = b"\xef\xbb\xbf" + expected.to_csv(index=False).encode("utf-8")
    original_to_csv = pd.DataFrame.to_csv
    original_replace = Path.replace
    opened = []

    def record_write(frame, file, **kwargs):
        opened.append(file)
        assert Path(file.name).parent == path.parent
        assert Path(file.name) != path
        assert path.read_bytes() == b"previous results"
        return original_to_csv(frame, file, **kwargs)

    def check_replace(temporary, target):
        assert target == path
        assert opened[0].closed
        assert temporary.read_bytes() == expected_bytes
        assert path.read_bytes() == b"previous results"
        return original_replace(temporary, target)

    monkeypatch.setattr(pd.DataFrame, "to_csv", record_write)
    monkeypatch.setattr(Path, "replace", check_replace)
    store = CsvNewsStore(str(path))
    store.save(rows)

    assert path.read_bytes() == expected_bytes
    assert store.load()["title"].tolist() == ["최근 뉴스", "이전 뉴스"]
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("existing_target", [False, True])
@pytest.mark.parametrize("failure_stage", ["create", "write", "sync", "replace"])
def test_csv_failure_preserves_target_and_removes_owned_temporary(
    tmp_path, monkeypatch, existing_target, failure_stage,
):
    path = tmp_path / "result.csv"
    original_bytes = b"\xef\xbb\xbfprevious,results\r\ncomplete,data\r\n"
    if existing_target:
        path.write_bytes(original_bytes)
    unrelated = tmp_path / ".result.csv.unrelated.tmp"
    unrelated.write_bytes(b"another writer's temporary file")

    def fail_create(**kwargs):
        raise PermissionError("temporary creation blocked")

    def fail_write(frame, file, **kwargs):
        file.write("partial CSV output")
        raise OSError("partial CSV write failed")

    def fail_sync(descriptor):
        raise OSError("CSV sync failed")

    def fail_replace(temporary, target):
        assert temporary.parent == path.parent
        assert target == path
        raise PermissionError("CSV replacement blocked")

    if failure_stage == "create":
        monkeypatch.setattr(csv_store_module, "NamedTemporaryFile", fail_create)
    elif failure_stage == "write":
        monkeypatch.setattr(pd.DataFrame, "to_csv", fail_write)
    elif failure_stage == "sync":
        monkeypatch.setattr(csv_store_module.os, "fsync", fail_sync)
    else:
        monkeypatch.setattr(Path, "replace", fail_replace)

    with pytest.raises(OSError):
        CsvNewsStore(str(path)).save([])

    if existing_target:
        assert path.read_bytes() == original_bytes
    else:
        assert not path.exists()
    assert unrelated.read_bytes() == b"another writer's temporary file"
    expected_files = {unrelated, path} if existing_target else {unrelated}
    assert set(tmp_path.iterdir()) == expected_files


def test_csv_cleanup_failure_does_not_hide_save_error(tmp_path, monkeypatch, caplog):
    path = tmp_path / "result.csv"
    path.write_bytes(b"previous results")
    original_unlink = Path.unlink

    def fail_write(frame, file, **kwargs):
        file.write("partial CSV output")
        raise OSError("original write failure")

    def fail_cleanup(temporary, *args, **kwargs):
        if temporary.parent == tmp_path and temporary.name.startswith(".result.csv."):
            raise PermissionError("cleanup blocked")
        return original_unlink(temporary, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "to_csv", fail_write)
    monkeypatch.setattr(Path, "unlink", fail_cleanup)
    with pytest.raises(OSError, match="original write failure"):
        CsvNewsStore(str(path)).save([])

    assert path.read_bytes() == b"previous results"
    assert "CSV 임시 파일 정리 실패" in caplog.text
    assert len(list(tmp_path.glob(".result.csv.*.tmp"))) == 1


def test_csv_does_not_create_missing_parent_directory(tmp_path):
    path = tmp_path / "missing" / "result.csv"
    with pytest.raises(FileNotFoundError):
        CsvNewsStore(str(path)).save([])
    assert not path.parent.exists()


def test_csv_success_preserves_unrelated_temporary_file(tmp_path):
    path = tmp_path / "result.csv"
    unrelated = path.with_suffix(".csv.tmp")
    unrelated.write_bytes(b"unrelated temporary file")

    CsvNewsStore(str(path)).save([])

    assert CsvNewsStore(str(path)).load().empty
    assert unrelated.read_bytes() == b"unrelated temporary file"
    assert set(tmp_path.iterdir()) == {path, unrelated}
