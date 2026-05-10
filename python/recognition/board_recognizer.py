"""盤面情報 (手牌 / ドラ / 自風 / 場風 / 局 / 巡目 / 点棒) を 1 フレームから
まとめて認識する (issue #12)。

設計方針:
- 各サブ認識器の例外は個別に握り、1 項目の失敗が他項目を巻き込まない
- 認識できなかったフィールドは「Rust 側 build_board_summary がスキーマを通る
  安全な既定値」を使う。フィールド単位の `None` を tenhou_json に出すと
  build_board_summary が GameBoardSummary 全体を None に倒す (= UI が
  「盤面なし」表示) ため、必須フィールドはダミーでも埋める
- ROI 未指定 / Tesseract 不在 / テンプレ未配置の各 graceful degrade は
  サブ認識器側で実装済み。BoardRecognizer は結果を集約するだけ。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from recognition import ocr_recognizer
from recognition.tile_recognizer import RoiRect, TileRecognizer
from recognition.wind_recognizer import WindRecognizer

logger = logging.getLogger("recognition")


# tenhou_json の「フィールドが認識できなかったとき」の既定値。
# Rust 側 `build_board_summary` の必須フィールドを満たすために残す。
DEFAULT_TENHOU_JSON: dict[str, Any] = {
    "hand": [],
    "river": [],
    "dora_indicators": ["5p"],
    "self_wind": "東",
    "round_wind": "東",
    "turn": 1,
    "scores": [25000, 25000, 25000, 25000],
    "melds": [],
}


def _roi(calib: dict[str, Any], key: str) -> RoiRect | None:
    """`roi_calibration` 辞書から RoiRect を取り出す。型不一致は `None`。"""
    if not isinstance(calib, dict):
        return None
    return RoiRect.from_dict(calib.get(key))


class BoardRecognizer:
    """各サブ認識器を保持して 1 フレームから tenhou_json を組み立てる。"""

    def __init__(self, template_dir: Path) -> None:
        self.template_dir = template_dir
        self.tile_recognizer = TileRecognizer(template_dir)
        # winds テンプレは `templates/winds/` 配下に置く規約 (templates/README.md)。
        self.wind_recognizer = WindRecognizer(template_dir / "winds")

    def recognize(
        self,
        bgr_frame: np.ndarray,
        roi_calibration: dict[str, Any] | None,
    ) -> tuple[dict[str, Any], float]:
        """1 フレームを認識して `(tenhou_json, confidence)` を返す。

        `confidence` は手牌 + ドラの最小値 (テンプレマッチの NCC スコア最小)。
        OCR / wind は単一マッチなので confidence 計算には混ぜず、内部ログで
        出すに留める。0.0 は「何も認識できなかった」を意味する。
        """
        tenhou: dict[str, Any] = dict(DEFAULT_TENHOU_JSON)
        confidence = 0.0

        if bgr_frame is None or bgr_frame.size == 0:
            return tenhou, confidence

        calib = roi_calibration if isinstance(roi_calibration, dict) else {}

        # ----- 手牌 (issue #11) -----
        try:
            hand_tiles, hand_conf = self.tile_recognizer.recognize_hand(
                bgr_frame, _roi(calib, "hand")
            )
            if hand_tiles:
                tenhou["hand"] = hand_tiles
                confidence = hand_conf
        except Exception:  # noqa: BLE001
            logger.warning("hand recognition failed", exc_info=True)

        # ----- ドラ表示牌 (issue #12) -----
        try:
            dora_tiles, dora_conf = self.tile_recognizer.recognize_dora(
                bgr_frame, _roi(calib, "doras")
            )
            if dora_tiles:
                tenhou["dora_indicators"] = dora_tiles
                # confidence は手牌・ドラのうち低い方を採用 (= フレーム全体の最低品質)。
                confidence = min(confidence, dora_conf) if confidence > 0 else dora_conf
        except Exception:  # noqa: BLE001
            logger.warning("dora recognition failed", exc_info=True)

        # ----- 自風 (issue #12) -----
        try:
            wind_label, _wind_conf = self.wind_recognizer.recognize(
                bgr_frame, _roi(calib, "self_wind")
            )
            if wind_label is not None:
                tenhou["self_wind"] = wind_label
        except Exception:  # noqa: BLE001
            logger.warning("self_wind recognition failed", exc_info=True)

        # ----- 局名 + 場風 (issue #12) -----
        round_label: str | None = None
        try:
            round_label = ocr_recognizer.recognize_round_label(bgr_frame, _roi(calib, "round_info"))
        except Exception:  # noqa: BLE001
            logger.warning("round_label recognition failed", exc_info=True)
        if round_label is not None:
            tenhou["round_label"] = round_label
            round_wind = ocr_recognizer.round_label_to_wind(round_label)
            if round_wind is not None:
                tenhou["round_wind"] = round_wind

        # ----- 巡目 (issue #12) -----
        try:
            turn = ocr_recognizer.recognize_turn(bgr_frame, _roi(calib, "turn_counter"))
            if turn is not None:
                tenhou["turn"] = turn
        except Exception:  # noqa: BLE001
            logger.warning("turn recognition failed", exc_info=True)

        # ----- 点棒 (issue #12) -----
        try:
            scores = ocr_recognizer.recognize_scores(bgr_frame, _roi(calib, "scores"))
            if scores is not None:
                tenhou["scores"] = scores
        except Exception:  # noqa: BLE001
            logger.warning("scores recognition failed", exc_info=True)

        return tenhou, confidence
