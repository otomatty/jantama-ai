"""Mortal 推論プロセスのエントリーポイント。

stdin から `{"type": "infer", "id": <int>, "tenhou_json": {...}}` を受け、
stdout に `{"type": "result", "id": <int>, "recommended": {...}, "candidates": [...]}` を返す。

MVP スケルトン: PyTorch ロード未実装。
- `--stub`: 常にスタブ応答を返す
- `--model <path>`: モデルパスを指定 (現状ログのみ)
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from typing import Any

from common import read_request, setup_stderr_logging, write_response

logger = setup_stderr_logging("mortal")


def stub_inference(_tenhou_json: dict[str, Any]) -> dict[str, Any]:
    """PRD §5.2 のサンプル数値をそのまま返すスタブ。"""
    candidates = [
        {"tile": "6m", "action_type": "discard", "expected_value": 0.32},
        {"tile": "9p", "action_type": "discard", "expected_value": 0.18},
        {"tile": "1z", "action_type": "discard", "expected_value": -0.05},
    ]
    return {
        "recommended": candidates[0],
        "candidates": candidates,
        "timestamp": datetime.now(UTC).isoformat(),
    }


def handle_infer(req: dict[str, Any]) -> dict[str, Any]:
    req_id = req.get("id", -1)
    tenhou = req.get("tenhou_json", {})
    # TODO(Phase D): Mortal モデルへ tenhou_json を渡して推論
    result = stub_inference(tenhou)
    return {"type": "result", "id": req_id, **result}


def main() -> int:
    parser = argparse.ArgumentParser(description="雀魂AIアシスタント Mortal 推論プロセス")
    parser.add_argument("--model", type=str, default=None, help="Mortal モデルファイル (.pth)")
    parser.add_argument("--stub", action="store_true", help="モデルをロードせずスタブ応答")
    args = parser.parse_args()

    logger.info("mortal process started (stub=%s, model=%s)", args.stub, args.model)

    if not args.stub and args.model:
        # TODO(Phase D): torch.load(args.model) して MortalAgent をインスタンス化
        logger.warning("model loading not implemented yet — falling back to stub mode")

    try:
        for req in read_request():
            req_type = req.get("type")
            if req_type == "infer":
                write_response(handle_infer(req))
            elif req_type == "ping":
                write_response({"type": "pong", "id": req.get("id")})
            else:
                write_response(
                    {"type": "error", "id": req.get("id"), "message": f"unknown type: {req_type}"}
                )
    except KeyboardInterrupt:
        logger.info("mortal process interrupted")

    logger.info("mortal process exiting")
    return 0


if __name__ == "__main__":
    sys.exit(main())
