import json
from pathlib import Path

import pytest

from news_classifier.utils.json_reports import write_new_json


@pytest.mark.parametrize("trailing_newline", [False, True])
@pytest.mark.parametrize("string_path", [False, True])
def test_new_json_report_preserves_format_and_returns_printable_text(tmp_path, trailing_newline, string_path):
    path = tmp_path / "nested" / "report.json"
    payload = {"제목": "기사", "values": [1, 2], "nested": {"success": True}}
    expected = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
    serialized = write_new_json(str(path) if string_path else path, payload,
                                trailing_newline=trailing_newline)
    assert serialized == expected
    assert path.read_text(encoding="utf-8") == expected + ("\n" if trailing_newline else "")
    assert not path.read_bytes().startswith(b"\xef\xbb\xbf")
    assert json.loads(serialized) == payload


def test_existing_json_report_is_not_overwritten(tmp_path):
    path = tmp_path / "report.json"
    path.write_bytes(b"original report")
    with pytest.raises(FileExistsError):
        write_new_json(path, {"new": "report"})
    assert path.read_bytes() == b"original report"


def test_exclusive_report_creation_keeps_file_created_by_another_writer(tmp_path, monkeypatch):
    path = tmp_path / "report.json"
    original_open = Path.open

    def race(target, mode="r", *args, **kwargs):
        if target == path and mode == "x":
            with original_open(target, "w", encoding="utf-8") as file:
                file.write("another writer's report")
        return original_open(target, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", race)
    with pytest.raises(FileExistsError):
        write_new_json(path, {"new": "report"})
    assert path.read_text(encoding="utf-8") == "another writer's report"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_strict_json_serialization_does_not_create_a_partial_report(tmp_path, value):
    path = tmp_path / "new-directory" / "report.json"
    with pytest.raises(ValueError):
        write_new_json(path, {"value": value})
    assert not path.exists()
    assert not path.parent.exists()


def test_legacy_allow_nan_option_remains_available(tmp_path):
    path = tmp_path / "report.json"
    payload = {"value": float("nan")}
    expected = json.dumps(payload, ensure_ascii=False, indent=2)
    assert write_new_json(path, payload, allow_nan=True) == expected
    assert path.read_text(encoding="utf-8") == expected
