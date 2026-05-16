import logging
import os
import sys
import warnings


QUIET_LOGGERS = (
    "httpx",
    "httpcore",
    "huggingface_hub",
    "huggingface_hub.utils._http",
    "transformers",
    "urllib3",
)


def configure_logging(level: int = logging.INFO) -> None:
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    warnings.filterwarnings("ignore", message=".*unauthenticated requests to the HF Hub.*")
    warnings.filterwarnings("ignore", message=".*HF_TOKEN.*")

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

    for logger_name in QUIET_LOGGERS:
        logging.getLogger(logger_name).setLevel(logging.ERROR)
