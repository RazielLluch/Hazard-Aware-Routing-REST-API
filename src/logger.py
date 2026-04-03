import logging
import sys
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Ensure logs directory exists
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

LOG_FILE = LOG_DIR / "app.log"


def get_logger(name: str = "app") -> logging.Logger:
    logger = logging.getLogger(name)

    # Prevent duplicate handlers (important in FastAPI reload)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    # ===== Formatter =====
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ===== Console Handler =====
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # ===== File Handler (Rotating) =====
    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=5 * 1024 * 1024,  # 5 MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    # ===== Add Handlers =====
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


# Default app-wide logger
logger = get_logger()