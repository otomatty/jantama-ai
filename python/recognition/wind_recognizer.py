"""自風 / 場風ラベルのテンプレートマッチング認識 (issue #12)。

雀魂 UI では卓上の各家位置に「東/南/西/北」のラベルが表示される。手牌の
1z..4z 牌画像とは見た目が違う (フォント・装飾) ため、別系統のテンプレ
`templates/winds/<east|south|west|north>.png` を用意してマッチさせる。

テンプレが揃わない / ROI 未指定なら `(None, 0.0)` を返し、呼び出し側 (
`BoardRecognizer`) がスタブ値にフォールバックする。
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from recognition.tile_recognizer import RoiRect

logger = logging.getLogger("recognition")

# テンプレファイル名 → tenhou_json で使う日本語ラベル。
_WIND_LABEL_MAP: dict[str, str] = {
    "east": "東",
    "south": "南",
    "west": "西",
    "north": "北",
}

# NCC (cv2.TM_CCOEFF_NORMED) スコアの下限。これ未満なら「自信なし」として `None` を返す。
# 4 種から「相対的に最良」だけで採用すると、ROI が画面の無関係領域 (背景・牌画像など)
# に当たっているフレームでも 0.0〜0.3 程度の弱マッチで仮の風が返ってしまい、
# `BoardRecognizer` 側の scores ゲートを誤って通過させる (Codex P1 on PR #44)。
# 雀魂の固定フォント + ROI が概ね合っていれば実マッチは 0.7+ になる想定。
# 0.5 で「相対最良 + 絶対しきい値」の二重チェックにする。
WIND_MATCH_THRESHOLD = 0.5


class WindRecognizer:
    """4 種の風ラベルテンプレを 1 度だけロードして使い回す。"""

    def __init__(self, template_dir: Path) -> None:
        self.template_dir = template_dir
        self._templates: dict[str, np.ndarray] = {}
        self._tmpl_size: tuple[int, int] | None = None
        self._loaded = False
        self._warned_no_roi = False
        self._load()

    def _load(self) -> None:
        if not self.template_dir.is_dir():
            logger.warning(
                "wind templates dir not found: %s (issue #16 で配置予定)",
                self.template_dir,
            )
            return

        loaded: dict[str, np.ndarray] = {}
        first_size: tuple[int, int] | None = None
        for key, _label in _WIND_LABEL_MAP.items():
            path = self.template_dir / f"{key}.png"
            if not path.is_file():
                continue
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                logger.warning("failed to read wind template: %s", path)
                continue
            if first_size is None:
                first_size = img.shape[:2]
            elif img.shape[:2] != first_size:
                img = cv2.resize(img, (first_size[1], first_size[0]))
            loaded[key] = img

        # 4 種揃わないと「該当しない 1 種が必ず誤分類される」ため、tile_recognizer
        # と同じく partial set は fail-closed にする (Codex P2 on PR #43 のアナロジー)。
        if len(loaded) < len(_WIND_LABEL_MAP):
            missing = [k for k in _WIND_LABEL_MAP if k not in loaded]
            if missing:
                logger.warning(
                    "partial wind template set in %s: missing %s; "
                    "wind recognition disabled until all templates are present",
                    self.template_dir,
                    ", ".join(missing),
                )
            return

        self._templates = loaded
        self._tmpl_size = first_size
        self._loaded = True
        logger.info("loaded %d wind templates from %s", len(loaded), self.template_dir)

    def recognize(
        self,
        bgr_frame: np.ndarray,
        wind_roi: RoiRect | None,
    ) -> tuple[str | None, float]:
        """自風ラベル ROI から「東/南/西/北」のいずれかと信頼度を返す。

        テンプレ未ロード or ROI 未指定なら `(None, 0.0)`。
        """
        if not self._loaded:
            return None, 0.0
        if wind_roi is None:
            if not self._warned_no_roi:
                logger.warning(
                    "self_wind ROI not calibrated; wind recognition returning None. "
                    "Run ROI calibration (issue #10) to enable wind recognition."
                )
                self._warned_no_roi = True
            return None, 0.0
        if bgr_frame is None or bgr_frame.size == 0:
            return None, 0.0

        h, w = bgr_frame.shape[:2]
        x0 = max(0, min(w, int(wind_roi.x * w)))
        y0 = max(0, min(h, int(wind_roi.y * h)))
        x1 = max(0, min(w, int((wind_roi.x + wind_roi.w) * w)))
        y1 = max(0, min(h, int((wind_roi.y + wind_roi.h) * h)))
        if x1 <= x0 or y1 <= y0:
            return None, 0.0

        crop = bgr_frame[y0:y1, x0:x1]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        assert self._tmpl_size is not None
        th, tw = self._tmpl_size
        if gray.shape != (th, tw):
            gray = cv2.resize(gray, (tw, th))

        best_key: str | None = None
        best_score = -1.0
        for key, tmpl in self._templates.items():
            res = cv2.matchTemplate(gray, tmpl, cv2.TM_CCOEFF_NORMED)
            score = float(res[0, 0])
            if score > best_score:
                best_score = score
                best_key = key

        if best_key is None:
            return None, 0.0
        # 信頼度しきい値: 「相対的に一番マシ」だけでは ROI ずれフレームで誤検出する
        # (Codex P1 on PR #44)。NCC < WIND_MATCH_THRESHOLD なら fail-closed。
        if best_score < WIND_MATCH_THRESHOLD:
            return None, best_score
        return _WIND_LABEL_MAP[best_key], best_score
