import logging
import os

from news_classifier.logging_config import QUIET_LOGGERS, configure_logging


def test_configure_logging_quiets_external_loggers(monkeypatch):
    monkeypatch.delenv("HF_HUB_DISABLE_PROGRESS_BARS", raising=False)
    monkeypatch.delenv("TRANSFORMERS_VERBOSITY", raising=False)

    configure_logging(logging.INFO)

    assert all(logging.getLogger(name).level == logging.ERROR for name in QUIET_LOGGERS)
    assert os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] == "1"
    assert os.environ["TRANSFORMERS_VERBOSITY"] == "error"
