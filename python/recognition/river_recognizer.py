"""4 家分の河（捨牌列）認識 (issue #13)。

各家の河 ROI を 6 列 × 4 段 = 24 セルのグリッドに分割し、各セルに牌が
あるかを std で判定。ある場合は C2 と同じ NCC テンプレートマッチで牌種を
識別する。横向き牌（リーチ宣言牌）はテンプレを 90 度回転させた版でも
照合し、回転側のスコアが高ければ「横向き」として記録する。

出力形式 (tenhou_json["river"]):

    [{"player": 0, "tile": "1m", "tedashi": true}, ...]

`player` は天鳳座順 (0=self / 自家, 1=right / 下家, 2=across / 対面,
3=left / 上家)。

`tedashi` (手出し) は単一フレームからは判定できないため True 固定で出力
する。時系列で「直前ツモ牌と一致したら tsumogiri」を見るのは後続の課題。

鳴かれた牌の除外は雀魂 UI 側で河から消えるため、空セル判定で自動的に
正しく扱われる。

依存: C1 (ROI キャリブレーション = issue #10), C2 (テンプレマッチング =
issue #11), C7 (テンプレ画像 = issue #16)。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from recognition.tile_recognizer import (
    BLANK_STD_THRESHOLD,
    RoiRect,
    load_tile_templates,
)

logger = logging.getLogger("recognition")

RIVER_COLS = 6
RIVER_ROWS = 4
RIVER_CELLS = RIVER_COLS * RIVER_ROWS  # = 24

# `roi_calibration.rivers` の JSON キーと天鳳座順 (player index) の対応。
# Rust 側 (src-tauri/src/types.rs::RiverRois) は `self_seat` を rename で
# JSON 上 "self" として永続化するため、ここでも "self" を使う。
RIVER_PLAYER_KEYS: tuple[str, ...] = ("self", "right", "across", "left")

# 横向き判定マージン。「縦向きスコアより回転後スコアがこの値以上高い」
# 場合のみ riichi 宣言牌として扱う。NCC スコアはノイズで多少ばらつくため、
# 単純な大小比較だと縦/横の取り違えで信頼度が落ちる。0.05 は安全側マージン
# (要チューニング)。
RIICHI_SCORE_MARGIN = 0.05


@dataclass(frozen=True)
class RiverTile:
    """1 セル分の認識結果。`to_dict()` で tenhou_json["river"] 形式に変換する。"""

    player: int
    tile: str
    tedashi: bool = True
    riichi: bool = False

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "player": self.player,
            "tile": self.tile,
            "tedashi": self.tedashi,
        }
        if self.riichi:
            d["riichi"] = True
        return d


def _extract_river_rois(
    river_rois: dict[str, Any] | None,
) -> dict[str, RoiRect | None]:
    """`roi_calibration.rivers` から各家の RoiRect を取り出す。"""
    out: dict[str, RoiRect | None] = {key: None for key in RIVER_PLAYER_KEYS}
    if not isinstance(river_rois, dict):
        return out
    for key in RIVER_PLAYER_KEYS:
        raw = river_rois.get(key)
        if isinstance(raw, dict):
            out[key] = RoiRect.from_dict(raw)
    return out


class RiverRecognizer:
    """37 テンプレ + その 90 度回転版を 1 度だけロードして使い回す河認識器。"""

    def __init__(self, template_dir: Path) -> None:
        self.template_dir = template_dir
        self._templates_upright: dict[str, np.ndarray] = {}
        self._templates_rotated: dict[str, np.ndarray] = {}
        # (h, w) for upright, (w, h) for rotated.
        self._size_upright: tuple[int, int] | None = None
        self._size_rotated: tuple[int, int] | None = None
        self._loaded = False
        self._warned_no_roi: set[str] = set()
        self._load()

    def _load(self) -> None:
        result = load_tile_templates(self.template_dir)
        if result is None:
            return
        templates, size = result
        # リーチ宣言牌は雀魂 UI 上で 90 度回転して横向きに配置される。
        # 回転方向は CCW でも CW でも matchTemplate のスコアは同等なので
        # 一方向だけ事前計算する。NCC は反転不変ではないため両方は不要。
        rotated = {
            code: cv2.rotate(tmpl, cv2.ROTATE_90_COUNTERCLOCKWISE)
            for code, tmpl in templates.items()
        }
        self._templates_upright = templates
        self._templates_rotated = rotated
        self._size_upright = size  # (h, w)
        self._size_rotated = (size[1], size[0])  # (w, h)
        self._loaded = True

    def recognize_rivers(
        self,
        bgr_frame: np.ndarray | None,
        river_rois: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        """4 家分の河を認識して tenhou_json["river"] 形式の list を返す。

        テンプレ未ロード / フレーム空 / `river_rois` 不正なら `[]`。
        ROI が指定されている家だけを走査する (未キャリブの家は黙ってスキップ
        ＋初回のみ warning ログ)。
        """
        if not self._loaded:
            return []
        if bgr_frame is None or bgr_frame.size == 0:
            return []

        rois = _extract_river_rois(river_rois)
        result: list[dict[str, Any]] = []
        h, w = bgr_frame.shape[:2]
        for player_idx, key in enumerate(RIVER_PLAYER_KEYS):
            roi = rois[key]
            if roi is None:
                if key not in self._warned_no_roi:
                    logger.warning(
                        "river ROI '%s' not calibrated; skipping player %d "
                        "(run ROI calibration to enable)",
                        key,
                        player_idx,
                    )
                    self._warned_no_roi.add(key)
                continue
            tiles = self._recognize_player(bgr_frame, roi, player_idx, (h, w))
            result.extend(t.to_dict() for t in tiles)
        return result

    def _recognize_player(
        self,
        bgr: np.ndarray,
        roi: RoiRect,
        player_idx: int,
        frame_size: tuple[int, int],
    ) -> list[RiverTile]:
        h, w = frame_size
        x0 = max(0, min(w, int(roi.x * w)))
        y0 = max(0, min(h, int(roi.y * h)))
        x1 = max(0, min(w, int((roi.x + roi.w) * w)))
        y1 = max(0, min(h, int((roi.y + roi.h) * h)))
        if x1 - x0 < RIVER_COLS or y1 - y0 < RIVER_ROWS:
            return []

        crop = bgr[y0:y1, x0:x1]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        ch, cw = gray.shape[:2]

        tiles: list[RiverTile] = []
        for row in range(RIVER_ROWS):
            sy = (row * ch) // RIVER_ROWS
            ey = ch if row == RIVER_ROWS - 1 else ((row + 1) * ch) // RIVER_ROWS
            for col in range(RIVER_COLS):
                sx = (col * cw) // RIVER_COLS
                ex = cw if col == RIVER_COLS - 1 else ((col + 1) * cw) // RIVER_COLS
                cell = gray[sy:ey, sx:ex]
                if cell.size == 0:
                    continue
                # 雀魂は鳴かれた牌を河から消す UI のため、空セルとして扱えば
                # 鳴かれた牌が自動的に除外される (issue #13 受け入れ基準)。
                if float(cell.std()) < BLANK_STD_THRESHOLD:
                    continue

                up_code, up_score = self._match(cell, self._templates_upright, self._size_upright)
                rot_code, rot_score = self._match(cell, self._templates_rotated, self._size_rotated)

                if up_code is None and rot_code is None:
                    continue

                if rot_code is not None and rot_score >= up_score + RIICHI_SCORE_MARGIN:
                    tiles.append(RiverTile(player=player_idx, tile=rot_code, riichi=True))
                elif up_code is not None:
                    tiles.append(RiverTile(player=player_idx, tile=up_code))
                elif rot_code is not None:
                    # 縦向きテンプレが 1 つもマッチしなかった (理論上ありえない
                    # が防御的に) ケース: 回転側を採用するが riichi 判定は
                    # 比較対象が無いので付けない。
                    tiles.append(RiverTile(player=player_idx, tile=rot_code))
        return tiles

    def _match(
        self,
        cell_gray: np.ndarray,
        templates: dict[str, np.ndarray],
        tmpl_size: tuple[int, int] | None,
    ) -> tuple[str | None, float]:
        if tmpl_size is None or not templates:
            return None, -1.0
        th, tw = tmpl_size
        cell_resized = cv2.resize(cell_gray, (tw, th)) if cell_gray.shape != (th, tw) else cell_gray
        best_code: str | None = None
        best_score = -1.0
        for code, tmpl in templates.items():
            res = cv2.matchTemplate(cell_resized, tmpl, cv2.TM_CCOEFF_NORMED)
            score = float(res[0, 0])
            if score > best_score:
                best_score = score
                best_code = code
        return best_code, best_score
