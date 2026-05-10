"""手牌 (13 + ツモ 1) の OpenCV テンプレートマッチング認識 (issue #11)。

37 種 (1m..9m, 0m, 1p..9p, 0p, 1s..9s, 0s, 1z..7z) のテンプレ PNG を
`templates/<code>.png` として持ち、ROI で切り出した手牌領域を 14 等分して
各セグメントを最大スコアのテンプレに割り当てる。

赤 5 (赤ドラ) は Mortal/天鳳の慣例に合わせ `0m`/`0p`/`0s` で出力する。

テンプレが 1 枚も無い (= issue #16 未完了) 状態でも、
`recognize_hand` は ([], 0.0) を返してプロセスを落とさない。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger("recognition")

TILE_CODES: list[str] = [
    *(f"{i}m" for i in range(1, 10)),
    "0m",
    *(f"{i}p" for i in range(1, 10)),
    "0p",
    *(f"{i}s" for i in range(1, 10)),
    "0s",
    *(f"{i}z" for i in range(1, 8)),
]

# 牌セグメントのグレースケール標準偏差がこれ未満なら「牌なし (空白)」と判定。
# 雀魂の手牌スロット背景は概ね無地で std が小さく、牌画像はテキスト・縁取りで
# std が 30+ になる傾向。8.0 は安全側のマージン込み (要チューニング)。
BLANK_STD_THRESHOLD = 8.0

# テンプレと同サイズへ resize した上で matchTemplate するため、結果は 1x1。
HAND_SLOTS = 14


@dataclass(frozen=True)
class RoiRect:
    """正規化済み ROI 矩形。フィールド名は src-tauri/src/types.rs の RoiRect と同じ。"""

    x: float
    y: float
    w: float
    h: float

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> RoiRect | None:
        if not isinstance(d, dict):
            return None
        try:
            return cls(x=float(d["x"]), y=float(d["y"]), w=float(d["w"]), h=float(d["h"]))
        except (KeyError, TypeError, ValueError):
            return None


class TileRecognizer:
    """37 テンプレを 1 度だけロードして使い回す手牌認識器。"""

    def __init__(self, template_dir: Path) -> None:
        self.template_dir = template_dir
        self._templates: dict[str, np.ndarray] = {}
        self._tmpl_size: tuple[int, int] | None = None  # (h, w)
        self._loaded = False
        self._warned_no_roi = False
        self._load()

    def _load(self) -> None:
        if not self.template_dir.is_dir():
            logger.warning(
                "tile templates dir not found: %s (issue #16 で配置予定)",
                self.template_dir,
            )
            return

        loaded: dict[str, np.ndarray] = {}
        first_size: tuple[int, int] | None = None
        for code in TILE_CODES:
            path = self.template_dir / f"{code}.png"
            if not path.is_file():
                continue
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                logger.warning("failed to read template: %s", path)
                continue
            if first_size is None:
                first_size = img.shape[:2]
            elif img.shape[:2] != first_size:
                img = cv2.resize(img, (first_size[1], first_size[0]))
            loaded[code] = img

        if not loaded:
            logger.warning(
                "no tile templates loaded from %s; recognize_hand will return empty (issue #16)",
                self.template_dir,
            )
            return

        # 部分セットでマッチングを走らせると「欠けた牌種」のセグメントが必ず
        # 残った牌に誤分類される (Codex P2 on PR #43)。安全側に倒し、37 種
        # 全部揃わないと recognize_hand を有効化しない。
        missing = [code for code in TILE_CODES if code not in loaded]
        if missing:
            logger.warning(
                "partial tile template set in %s: missing %d/%d (%s); "
                "recognize_hand disabled until all templates are present",
                self.template_dir,
                len(missing),
                len(TILE_CODES),
                ", ".join(missing),
            )
            return

        self._templates = loaded
        self._tmpl_size = first_size
        self._loaded = True
        logger.info("loaded %d tile templates from %s", len(loaded), self.template_dir)

    def recognize_hand(
        self,
        bgr_frame: np.ndarray,
        hand_roi: RoiRect | None,
    ) -> tuple[list[str], float]:
        """手牌領域から牌コード列と最低信頼度を返す。

        テンプレ未ロード or ROI 未指定なら ([], 0.0)。

        ROI 未指定時は「全画面を 14 等分」というフォールバックは取らない。
        手牌領域が画面のごく一部 (画面下端の細い帯) なので、未キャリブで
        全画面を 14 分割しても意味のある結果が出ず、誤マッチ由来のノイズで
        Mortal を惑わすほうが害が大きい。代わりに 1 度だけ警告ログを出して
        ユーザに ROI キャリブレーションを促す。
        """
        if not self._loaded:
            return [], 0.0
        if hand_roi is None:
            if not self._warned_no_roi:
                logger.warning(
                    "hand ROI not calibrated; recognize_hand returning empty. "
                    "Run ROI calibration (issue #10) to enable hand recognition."
                )
                self._warned_no_roi = True
            return [], 0.0
        if bgr_frame is None or bgr_frame.size == 0:
            return [], 0.0

        h, w = bgr_frame.shape[:2]
        x0 = max(0, min(w, int(hand_roi.x * w)))
        y0 = max(0, min(h, int(hand_roi.y * h)))
        x1 = max(0, min(w, int((hand_roi.x + hand_roi.w) * w)))
        y1 = max(0, min(h, int((hand_roi.y + hand_roi.h) * h)))
        if x1 - x0 < HAND_SLOTS or y1 - y0 <= 0:
            return [], 0.0

        crop = bgr_frame[y0:y1, x0:x1]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        crop_h, crop_w = gray.shape[:2]
        # MVP: 単純に 14 等分する。雀魂 UI では 13 牌目とツモ牌の間に数十 px
        # の隙間があるが、その隙間部分は std がしきい値未満になるので空白判定
        # で自然にスキップされる前提 (Gemini medium on PR #43)。隙間幅まで
        # 込みでスロット境界を補正する高精度版は #16 でテンプレ実画像が
        # 揃ってから精度計測した上で検討する。
        seg_w_base = crop_w // HAND_SLOTS

        tiles: list[str] = []
        scores: list[float] = []
        for i in range(HAND_SLOTS):
            sx = i * seg_w_base
            ex = crop_w if i == HAND_SLOTS - 1 else (i + 1) * seg_w_base
            seg = gray[:, sx:ex]
            if seg.size == 0:
                continue
            if float(seg.std()) < BLANK_STD_THRESHOLD:
                continue
            code, score = self._match_segment(seg)
            if code is None:
                continue
            tiles.append(code)
            scores.append(score)

        if not tiles:
            return [], 0.0
        return tiles, float(min(scores))

    def _match_segment(self, seg_gray: np.ndarray) -> tuple[str | None, float]:
        assert self._tmpl_size is not None
        th, tw = self._tmpl_size
        if seg_gray.shape != (th, tw):
            seg_gray = cv2.resize(seg_gray, (tw, th))

        best_code: str | None = None
        best_score = -1.0
        for code, tmpl in self._templates.items():
            res = cv2.matchTemplate(seg_gray, tmpl, cv2.TM_CCOEFF_NORMED)
            score = float(res[0, 0])
            if score > best_score:
                best_score = score
                best_code = code
        return best_code, best_score
