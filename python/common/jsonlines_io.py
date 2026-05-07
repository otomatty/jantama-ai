"""stdin/stdout JSON-lines I/O ヘルパ。

Tauri (Rust) から渡ってくる 1 行 1 JSON のリクエストを読み取り、
レスポンスも 1 行 1 JSON で返す。
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterator
from typing import Any


def read_request(stream=sys.stdin) -> Iterator[dict[str, Any]]:
    """stdin から 1 行ずつ JSON を読む。EOF で終了。"""
    for line in stream:
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError as e:
            write_response({"type": "error", "message": f"invalid json: {e}"})


def write_response(payload: dict[str, Any], stream=sys.stdout) -> None:
    """1 行 JSON を書き出して flush する。"""
    stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
    stream.flush()
