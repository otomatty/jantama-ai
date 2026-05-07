"""認識プロセスのエントリーポイント。

stdin から `{"type": "frame", "id": <int>, "image_b64": "<base64>"}` を受け、
stdout に `{"type": "result", "id": <int>, "tenhou_json": {...}, "confidence": <float>}` を返す。

MVP スケルトン: 実際の画像処理は未実装。
- `--echo`: 受信した内容をそのままエコーバック
- 通常起動: スタブ天鳳 JSON を返却
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

from common import read_request, setup_stderr_logging, write_response

logger = setup_stderr_logging("recognition")


def stub_tenhou_json() -> dict[str, Any]:
    """雛形の天鳳 JSON。手牌・河・場況などを実装中は仮値で返す。"""
    return {
        "hand": ["1m", "2m", "3m", "4m", "5m", "6m", "7p", "8p", "9p", "1z", "2z", "3z", "5m"],
        "river": [],
        "dora_indicators": ["5p"],
        "self_wind": "東",
        "round_wind": "東",
        "turn": 6,
        "scores": [25000, 25000, 25000, 25000],
        "melds": [],
    }


def handle_frame(req: dict[str, Any]) -> dict[str, Any]:
    """1 フレームを処理して結果を返す。"""
    frame_id = req.get("id", -1)
    # TODO(Phase C): image_b64 をデコードして OpenCV で認識する
    return {
        "type": "result",
        "id": frame_id,
        "tenhou_json": stub_tenhou_json(),
        "confidence": 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="雀魂AIアシスタント 認識プロセス")
    parser.add_argument("--echo", action="store_true", help="入力をそのまま返すモード")
    args = parser.parse_args()

    logger.info("recognition process started (echo=%s)", args.echo)

    try:
        for req in read_request():
            req_type = req.get("type")
            if args.echo:
                write_response({"type": "echo", "received": req})
                continue

            if req_type == "frame":
                response = handle_frame(req)
                write_response(response)
            elif req_type == "ping":
                write_response({"type": "pong", "id": req.get("id")})
            else:
                write_response(
                    {"type": "error", "id": req.get("id"), "message": f"unknown type: {req_type}"}
                )
    except KeyboardInterrupt:
        logger.info("recognition process interrupted")

    logger.info("recognition process exiting")
    return 0


if __name__ == "__main__":
    sys.exit(main())
