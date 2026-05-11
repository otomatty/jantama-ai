"""認識プロセスのエントリーポイント。

stdin から `{"type": "frame", "id": <int>, "image_b64": "<base64>",
"roi_calibration": {...}}` を受け、
stdout に `{"type": "result", "id": <int>, "tenhou_json": {...},
"confidence": <float>}` を返す。

issue #11: 手牌領域 (`roi_calibration.hand`) を OpenCV テンプレートマッチング
で 13(+1) 牌に分解する。
issue #12: ドラ表示牌・自風・場風・局名・巡目・点棒を BoardRecognizer がまとめて
認識する (各サブ認識器は graceful degrade で必須フィールドの既定値を返す)。
issue #13: 河 (捨牌) を RiverRecognizer が 4 家分認識する。
副露などはまだ別 issue (#14)。

- `--echo`: 受信した内容をそのままエコーバック (デバッグ用)
"""

from __future__ import annotations

import argparse
import base64
import binascii
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from common import read_request, setup_stderr_logging, write_response
from recognition.board_recognizer import DEFAULT_TENHOU_JSON, BoardRecognizer
from recognition.river_recognizer import RiverRecognizer

logger = setup_stderr_logging("recognition")

_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
_recognizer: BoardRecognizer | None = None
_river_recognizer: RiverRecognizer | None = None


def _get_recognizer() -> BoardRecognizer:
    global _recognizer
    if _recognizer is None:
        _recognizer = BoardRecognizer(_TEMPLATE_DIR)
    return _recognizer


def _get_river_recognizer() -> RiverRecognizer:
    global _river_recognizer
    if _river_recognizer is None:
        _river_recognizer = RiverRecognizer(_TEMPLATE_DIR)
    return _river_recognizer


def stub_tenhou_json() -> dict[str, Any]:
    """全フィールド認識失敗時の最終フォールバック値。

    issue #12 以降、各フィールドは個別に認識される (BoardRecognizer)。
    この関数は「フレームデコード失敗」など全認識が走らなかった例外路で使う。
    """
    return dict(DEFAULT_TENHOU_JSON)


def _decode_frame(image_b64: Any) -> np.ndarray | None:
    if not isinstance(image_b64, str) or not image_b64:
        return None
    try:
        raw = base64.b64decode(image_b64, validate=False)
    except (binascii.Error, ValueError):
        logger.warning("failed to base64-decode image_b64")
        return None
    buf = np.frombuffer(raw, dtype=np.uint8)
    if buf.size == 0:
        return None
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        logger.warning("cv2.imdecode returned None (corrupt PNG?)")
    return img


def handle_frame(req: dict[str, Any]) -> dict[str, Any]:
    """1 フレームを処理して結果を返す。例外時も必ずスキーマを満たすレスポンスを返す。"""
    frame_id = req.get("id", -1)
    tenhou_json = stub_tenhou_json()
    confidence = 0.0

    try:
        bgr = _decode_frame(req.get("image_b64"))
    except Exception:  # noqa: BLE001 — recognition プロセスを落とさない
        logger.warning("frame decode failed for id=%s", frame_id, exc_info=True)
        bgr = None

    if bgr is not None:
        roi_calib = req.get("roi_calibration")
        roi_calib = roi_calib if isinstance(roi_calib, dict) else {}

        # 盤面 (手牌/ドラ/自風/場風/局名/巡目/点棒) と河は独立した try に分け、
        # 片方が失敗してももう片方は試行する。
        try:
            tenhou_json, confidence = _get_recognizer().recognize(bgr, roi_calib)
        except Exception:  # noqa: BLE001 — recognition プロセスを落とさない
            logger.warning("board recognition failed for id=%s", frame_id, exc_info=True)

        try:
            tenhou_json["river"] = _get_river_recognizer().recognize_rivers(
                bgr, roi_calib.get("rivers")
            )
        except Exception:  # noqa: BLE001 — recognition プロセスを落とさない
            logger.warning("river recognition failed for id=%s", frame_id, exc_info=True)

    return {
        "type": "result",
        "id": frame_id,
        "tenhou_json": tenhou_json,
        "confidence": confidence,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="雀魂AIアシスタント 認識プロセス")
    parser.add_argument("--echo", action="store_true", help="入力をそのまま返すモード")
    args = parser.parse_args()

    logger.info("recognition process started (echo=%s)", args.echo)

    if not args.echo:
        # テンプレロードを起動時に走らせ、不在時の警告を早期に出す。
        _get_recognizer()
        _get_river_recognizer()

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
