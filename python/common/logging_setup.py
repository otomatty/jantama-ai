"""共通ロガー設定。

stdin/stdout は通信用なので、ログは必ず stderr に出す。
"""

from __future__ import annotations

import logging
import sys


def setup_stderr_logging(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] %(message)s")
        )
        logger.addHandler(handler)
    return logger
