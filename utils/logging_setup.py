"""
Console logging: keep our messages visible, hide noisy HTTP / Hub chatter unless WARNING+.
"""

from __future__ import annotations

import logging
import os

# Hugging Face Hub tqdm bars during download/load
_NOISY_LOGGERS = (
    "httpx",
    "httpcore",
    "urllib3",
    "huggingface_hub",
    "transformers",
    "filelock",
    "PIL",
    "sentencepiece",
)


def prepare_hf_console() -> None:
    """Env tweaks before importing HF/transformers (quieter downloads and model load)."""
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    # Fewer transformers console messages during from_pretrained / load.
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")


def configure_logging(level: int = logging.INFO) -> None:
    """
    Root logger at `level`. Third-party loggers use WARNING (hides httpx INFO GET lines)
    when the app is DEBUG/INFO; if you pass ERROR, those loggers use ERROR too.
    """
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
            datefmt="%H:%M:%S",
        )
    root.setLevel(level)
    noisy_level = logging.ERROR if level >= logging.ERROR else logging.WARNING
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(noisy_level)
