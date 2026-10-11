import json
from pathlib import Path

import pytest

from news_classifier.utils.json_profiles import replace_json_profile


@pytest.mark.parametrize("exclusive_temporary", [False, True])
@pytest.mark.parametrize("string_path", [False, True])
def test_profile_replacement_keeps_strict_utf8_format(tmp_path, exclusive_temporary, string_path):
    path = tmp_path / "profile.json"
    path.write_text("previous profile", encoding="utf-8")
    payload = {"제목": "분류", "thresholds": {"score": 0.5, "margin": 0.05}}
    replace_json_profile(str(path) if string_path else path, payload,
                         exclusive_temporary=exclusive_temporary)
    assert path.read_text(encoding="utf-8") == json.dumps(
        payload, ensure_ascii=False, indent=2, allow_nan=False,
    )
    assert not path.read_bytes().startswith(b"\xef\xbb\xbf")
    assert not path.with_suffix(".json.tmp").exists()


def test_confidence_profile_reuses_existing_temporary_file(tmp_path):
    path = tmp_path / "confidence.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text("previous unfinished confidence output", encoding="utf-8")
    replace_json_profile(path, {"new": "profile"})
    assert json.loads(path.read_text(encoding="utf-8")) == {"new": "profile"}
    assert not temporary.exists()


def test_decision_profile_keeps_existing_temporary_file_and_profile(tmp_path):
    path = tmp_path / "decision.json"
    path.write_bytes(b"previous decision profile")
    temporary = path.with_suffix(".json.tmp")
    temporary.write_bytes(b"unrelated unfinished output")
    with pytest.raises(FileExistsError):
        replace_json_profile(path, {"new": "profile"}, exclusive_temporary=True)
    assert temporary.read_bytes() == b"unrelated unfinished output"
    assert path.read_bytes() == b"previous decision profile"


@pytest.mark.parametrize("exclusive_temporary", [False, True])
def test_replacement_failure_keeps_original_profile_and_completed_temporary_file(
    tmp_path, monkeypatch, exclusive_temporary,
):
    path = tmp_path / "profile.json"
    path.write_bytes(b"previous profile")
    payload = {"new": "profile"}

    def fail_replace(source, target):
        raise PermissionError("replacement blocked")

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(PermissionError, match="replacement blocked"):
        replace_json_profile(path, payload, exclusive_temporary=exclusive_temporary)
    assert path.read_bytes() == b"previous profile"
    assert json.loads(path.with_suffix(".json.tmp").read_text(encoding="utf-8")) == payload


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("exclusive_temporary", [False, True])
def test_nonfinite_json_never_replaces_profile_and_preserves_temporary_policy(
    tmp_path, value, exclusive_temporary,
):
    path = tmp_path / "profile.json"
    path.write_bytes(b"previous profile")
    temporary = path.with_suffix(".json.tmp")
    if not exclusive_temporary:
        temporary.write_bytes(b"previous unfinished confidence output")
    with pytest.raises(ValueError):
        replace_json_profile(path, {"value": value}, exclusive_temporary=exclusive_temporary)
    assert path.read_bytes() == b"previous profile"
    if not exclusive_temporary:
        assert temporary.read_bytes() == b"previous unfinished confidence output"
    else:
        # The decision writer's existing streaming failure leaves its own temp,
        # never a replaced profile or overwritten unrelated temporary file.
        assert temporary.exists()


def test_profile_writer_keeps_directory_creation_caller_owned(tmp_path):
    path = tmp_path / "missing" / "profile.json"
    with pytest.raises(FileNotFoundError):
        replace_json_profile(path, {"new": "profile"})
    assert not path.parent.exists()
