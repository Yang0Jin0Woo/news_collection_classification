from news_classifier.config import AppSettings, load_environment


def test_env_file_is_loaded_without_overriding_existing_variables(
    monkeypatch, tmp_path
):
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                "NEWS_OUTPUT_CSV=from-dotenv.csv",
                "NEWS_MODEL_REVISION=dotenv-revision",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("NEWS_OUTPUT_CSV", raising=False)
    monkeypatch.setenv("NEWS_MODEL_REVISION", "operating-system-revision")

    assert load_environment(env_path) is True
    settings = AppSettings()

    assert settings.output_csv == "from-dotenv.csv"
    assert settings.classification_model_revision == "operating-system-revision"


def test_default_model_revision_is_fixed(monkeypatch):
    monkeypatch.delenv("NEWS_MODEL_REVISION", raising=False)

    assert AppSettings().classification_model_revision == (
        "b5113eb38ab63efdd7f280f8c144ea8b13f978ce"
    )
