from pathlib import Path
import tomllib


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_pyproject_defines_pinned_dependencies_and_cli_entrypoint():
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as file:
        pyproject = tomllib.load(file)

    project = pyproject["project"]
    assert project["scripts"]["news-classifier"] == "news_classifier.cli:main"
    assert project["dependencies"]
    assert all("==" in dependency for dependency in project["dependencies"])


def test_lock_file_uses_exact_versions():
    lines = (PROJECT_ROOT / "requirements.lock").read_text(
        encoding="utf-8"
    ).splitlines()
    requirements = [
        line for line in lines if line and not line.startswith(("#", "-e "))
    ]

    assert requirements
    assert all("==" in requirement for requirement in requirements)
