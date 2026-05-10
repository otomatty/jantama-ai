"""共通ロガー設定。

stdin/stdout は通信用なので、ログは必ず stderr に出す。

Rust 親プロセス (`src-tauri/src/python_proc.rs`) が stderr を 1 行ずつ
パースして tracing log / Tauri Event `python-log` へ流すため、

    {level}\t{logger}\t{message}

の TAB 区切り 3 列フォーマットに揃える。例外スタックトレース等で改行を含む
メッセージは `\\n` にエスケープし、1 レコード = 1 行を維持する。
"""

from __future__ import annotations

import logging
import sys


class _StructuredTabFormatter(logging.Formatter):
    """TSV 構造化ログを 1 行に詰めるフォーマッタ。

    `logging.Formatter.format` が生成するメッセージにはトレースバック等で
    改行が混ざるが、Rust 親プロセスは行単位でパースするので、出力前に
    `\\n` / `\\r` を可視文字列へエスケープする (1 レコード = 1 行を保証)。
    """

    def format(self, record: logging.LogRecord) -> str:
        text = super().format(record)
        return text.replace("\r", "\\r").replace("\n", "\\n")


def setup_stderr_logging(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(_StructuredTabFormatter("%(levelname)s\t%(name)s\t%(message)s"))
        logger.addHandler(handler)
    return logger
