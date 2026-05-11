"""4 家分の副露 (鳴き / called melds) 認識 (issue #14)。

各家の副露 ROI から 0〜4 個の副露を検出し、各副露の種別 (chi/pon/minkan/
ankan/kakan) と「どこから鳴いたか」を判定する。

依存: C1 (ROI), C2 (テンプレ), C7 (河と同じ前提)。

アルゴリズム:
  1. プレイヤー別に crop を upright 向きへ正規化 (river と同じ)
  2. グレースケール → 列ごとの std で「副露グループ」を抽出
  3. グループ内で先に「横向き牌」を NCC スライドで 1 つ特定し、残りを
     upright 幅で等分する
  4. 各スロットを NCC で識別 + 横向きの kakan スタック判定
  5. (枚数, 横向き枚数, 横向き位置, スタック有無) で副露種別を決定

出力 (tenhou_json["melds"]):
    [{"player": 0, "type": "pon", "tiles": ["1m","1m","1m"], "from": 1}, ...]

`from` は鳴いた本人から見た相対座順 (Mortal/天鳳 標準):
  0=自家(暗槓のみ), 1=下家, 2=対面, 3=上家
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
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

# `roi_calibration.melds` の JSON キーと天鳳座順 (player index) の対応。
# Rust 側 (src-tauri/src/types.rs::MeldRois) は `self_seat` を rename で
# JSON 上 "self" として永続化するため、ここでも "self" を使う。
MELD_PLAYER_KEYS: tuple[str, ...] = ("self", "right", "across", "left")

# 横向き判定マージン: 「縦向きスコアより回転後スコアがこの値以上高い」
# 場合のみ横向き牌として扱う (river の RIICHI_SCORE_MARGIN と同じ意味)。
HORIZONTAL_SCORE_MARGIN = 0.05

# 暗槓 (ankan) 判定: グループ内全スロットの NCC スコア最大値がこの値未満なら
# ankan と判定する。裏向き牌は表 37 テンプレに対して低 NCC しか出ない前提。
# 専用「裏向き」テンプレを別途用意するのが本命だが MVP は閾値判定で間に合わせる。
ANKAN_NCC_THRESHOLD = 0.4

# 加槓 (kakan) の積み牌判定: 横向きスロットの上にある 2 枚目の横向き牌が
# このスコア以上なら kakan と判定する。
KAKAN_STACK_NCC_THRESHOLD = 0.5

# 副露グループ間の隙間判定 (連続する空白列の最低幅 px)。雀魂 UI では副露と
# 副露の間に明確な隙間 (背景色) が空くため、この幅で run を切り分ける。
MELD_GAP_MIN_COLS = 2

# 1 副露の最低牌枚数 (chi/pon = 3, kan = 4)。これ未満のグループはノイズ扱い。
MIN_TILES_PER_MELD = 3

# 1 副露の最大牌枚数 (槓のみ 4 枚)。これ超えはセグメント誤分割と判定して捨てる。
MAX_TILES_PER_MELD = 4


@dataclass(frozen=True)
class Meld:
    """1 副露の認識結果。`to_dict()` で tenhou_json["melds"] 形式に変換する。

    `tiles` は副露を構成する牌コードのリスト (chi/pon=3, ankan/minkan/kakan=4)。
    `from_` は鳴いた本人から見た相対座順 (0=自家(ankan), 1=下家, 2=対面, 3=上家)。
    """

    player: int
    type: str
    tiles: list[str] = field(default_factory=list)
    from_: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "player": self.player,
            "type": self.type,
            "tiles": list(self.tiles),
            "from": self.from_,
        }


@dataclass
class _TileSlot:
    """グループ内の 1 牌スロット (内部表現)。x_in_group はグループ左端からの px。"""

    code: str
    score: float
    is_horizontal: bool
    x_in_group: int
    width: int


def _extract_meld_rois(
    meld_rois: dict[str, Any] | None,
) -> dict[str, RoiRect | None]:
    """`roi_calibration.melds` から各家の RoiRect を取り出す。"""
    out: dict[str, RoiRect | None] = {key: None for key in MELD_PLAYER_KEYS}
    if not isinstance(meld_rois, dict):
        return out
    for key in MELD_PLAYER_KEYS:
        raw = meld_rois.get(key)
        if isinstance(raw, dict):
            out[key] = RoiRect.from_dict(raw)
    return out


class MeldsRecognizer:
    """37 テンプレ + その 90 度回転版 (CW/CCW) を 1 度だけロードして使い回す副露認識器。"""

    def __init__(self, template_dir: Path) -> None:
        self.template_dir = template_dir
        self._templates_upright: dict[str, np.ndarray] = {}
        # 横向き牌は CW / CCW どちら向きで倒されているか分からないため、
        # 両方向を事前計算して照合する (river と同じ方針)。
        self._templates_rotated_ccw: dict[str, np.ndarray] = {}
        self._templates_rotated_cw: dict[str, np.ndarray] = {}
        self._size_upright: tuple[int, int] | None = None  # (h, w)
        self._size_rotated: tuple[int, int] | None = None  # (w, h) = (h, w) 入替後
        self._loaded = False
        self._warned_no_roi: set[str] = set()
        self._load()

    def _load(self) -> None:
        result = load_tile_templates(self.template_dir)
        if result is None:
            return
        templates, size = result
        rotated_ccw = {
            code: cv2.rotate(tmpl, cv2.ROTATE_90_COUNTERCLOCKWISE)
            for code, tmpl in templates.items()
        }
        rotated_cw = {
            code: cv2.rotate(tmpl, cv2.ROTATE_90_CLOCKWISE) for code, tmpl in templates.items()
        }
        self._templates_upright = templates
        self._templates_rotated_ccw = rotated_ccw
        self._templates_rotated_cw = rotated_cw
        self._size_upright = size  # (h, w)
        self._size_rotated = (size[1], size[0])  # (w, h)
        self._loaded = True

    def recognize_melds(
        self,
        bgr_frame: np.ndarray | None,
        meld_rois: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        """4 家分の副露を認識して tenhou_json["melds"] 形式の list を返す。

        テンプレ未ロード / フレーム空 / `meld_rois` 不正なら `[]`。
        ROI が指定されている家だけを走査する (未キャリブの家は黙ってスキップ
        + 初回のみ warning ログ)。
        """
        if not self._loaded:
            return []
        if bgr_frame is None or bgr_frame.size == 0:
            return []

        rois = _extract_meld_rois(meld_rois)
        result: list[dict[str, Any]] = []
        h, w = bgr_frame.shape[:2]
        for player_idx, key in enumerate(MELD_PLAYER_KEYS):
            roi = rois[key]
            if roi is None:
                if key not in self._warned_no_roi:
                    logger.warning(
                        "meld ROI '%s' not calibrated; skipping player %d "
                        "(run ROI calibration to enable)",
                        key,
                        player_idx,
                    )
                    self._warned_no_roi.add(key)
                continue
            melds = self._recognize_player(bgr_frame, roi, player_idx, (h, w))
            result.extend(m.to_dict() for m in melds)
        return result

    def _recognize_player(
        self,
        bgr: np.ndarray,
        roi: RoiRect,
        player_idx: int,
        frame_size: tuple[int, int],
    ) -> list[Meld]:
        h, w = frame_size
        x0 = max(0, min(w, int(roi.x * w)))
        y0 = max(0, min(h, int(roi.y * h)))
        x1 = max(0, min(w, int((roi.x + roi.w) * w)))
        y1 = max(0, min(h, int((roi.y + roi.h) * h)))
        if x1 - x0 <= 0 or y1 - y0 <= 0:
            return []

        crop = bgr[y0:y1, x0:x1]
        if crop.size == 0:
            return []
        # 雀魂 UI では他家の副露が画面上で回転して描画される (右家=90° CW,
        # 対面=180°, 左家=90° CCW)。crop をプレイヤーごとに逆回転で正規化
        # して「縦向き = upright」に揃え、副露検出を 4 家で共通アルゴで扱う。
        if player_idx == 1:
            crop = cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif player_idx == 2:
            crop = cv2.rotate(crop, cv2.ROTATE_180)
        elif player_idx == 3:
            crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        groups = self._find_meld_groups(gray)
        melds: list[Meld] = []
        for gx0, gx1 in groups:
            slots = self._segment_meld(gray, gx0, gx1)
            if not (MIN_TILES_PER_MELD <= len(slots) <= MAX_TILES_PER_MELD):
                continue
            kakan_stack = self._detect_kakan_stack(gray, gx0, gx1, slots)
            meld = self._classify_meld(slots, kakan_stack, player_idx)
            if meld is not None:
                melds.append(meld)
        return melds

    def _find_meld_groups(self, gray: np.ndarray) -> list[tuple[int, int]]:
        """列 std で active な区間 run を抽出する。

        副露間は MELD_GAP_MIN_COLS 以上の連続空白列で区切られる。空白列は
        雀魂 UI の副露と副露の隙間 (背景色) に対応する。
        """
        if gray.size == 0 or self._size_upright is None:
            return []
        col_std = gray.std(axis=0)
        active = col_std > BLANK_STD_THRESHOLD

        runs: list[tuple[int, int]] = []
        in_run = False
        run_start = 0
        gap_count = 0
        cw = gray.shape[1]
        for x in range(cw):
            if active[x]:
                if not in_run:
                    in_run = True
                    run_start = x
                gap_count = 0
            else:
                if in_run:
                    gap_count += 1
                    if gap_count >= MELD_GAP_MIN_COLS:
                        runs.append((run_start, x - gap_count + 1))
                        in_run = False
                        gap_count = 0
        if in_run:
            runs.append((run_start, cw))

        # 最低幅: テンプレ幅未満なら誤検出 (ノイズ列) として落とす。
        min_width = self._size_upright[1]
        return [(s, e) for s, e in runs if e - s >= min_width]

    def _segment_meld(self, gray: np.ndarray, gx0: int, gx1: int) -> list[_TileSlot]:
        """グループ内を 1 牌ずつのスロットに分割する。

        手順:
          1. グループ最下段で「横向き牌スロット」を NCC スライドで 1 つ最大
             でも特定 (各副露の横向きは 0 個 (ankan) か 1 個)
          2. 横向きスロットがあれば、その左右を upright 幅で等分する
          3. 横向きスロットが無ければ、グループ全幅を upright 幅で等分する

        upright と horizontal の幅は異なる (W vs H) ため、横向きを先に決めると
        残りの等分が綺麗に揃う (greedy 左右等分よりカスケード誤差が出にくい)。
        """
        if self._size_upright is None or self._size_rotated is None:
            return []
        h_up, w_up = self._size_upright  # (H, W)
        h_rot, w_rot = self._size_rotated  # (W, H): 横向き牌の (高さ, 幅)

        group = gray[:, gx0:gx1]
        gh, gw = group.shape
        if gh < h_rot or gw < w_up:
            return []

        # 横向きスロットを strip 全幅に対して 1 度だけ NCC スキャンして特定する
        # (gemini-code-assist Medium on PR #46): 元実装の x×37×2 個別呼び出しを
        # 37×2 個の cv2.matchTemplate に置換 (各 call が strip 全幅の score map を返す)。
        best_h_x, best_h_score, best_h_code = self._find_best_horizontal(
            group, h_up, w_up, h_rot, w_rot
        )
        # 横向き候補は絶対 NCC が ANKAN_NCC_THRESHOLD 以上ある場合のみ採用する
        # (coderabbitai Major on PR #46)。ガードがないと、4 枚とも裏向き
        # (ankan 候補) の strip でも、ノイズで「横向きが upright より僅かに高い」
        # x が偶発的に選ばれ、h_count=1 → minkan 判定で ankan 検出が潰される。
        slots: list[_TileSlot] = []
        if best_h_x >= 0 and best_h_code is not None and best_h_score >= ANKAN_NCC_THRESHOLD:
            left_slots = self._segment_upright_run(group, 0, best_h_x)
            right_slots = self._segment_upright_run(group, best_h_x + w_rot, gw)
            slots.extend(left_slots)
            slots.append(
                _TileSlot(
                    code=best_h_code,
                    score=best_h_score,
                    is_horizontal=True,
                    x_in_group=best_h_x,
                    width=w_rot,
                )
            )
            slots.extend(right_slots)
            slots.sort(key=lambda s: s.x_in_group)
        else:
            slots = self._segment_upright_run(group, 0, gw)
        return slots

    def _find_best_horizontal(
        self,
        group: np.ndarray,
        h_up: int,
        w_up: int,
        h_rot: int,
        w_rot: int,
    ) -> tuple[int, float, str | None]:
        """horizontal slot を 1 つだけ特定し `(x, score, code)` を返す。

        テンプレ 1 つにつき `cv2.matchTemplate` 1 回で strip 全幅の score map を
        取り、x ごとの最大値テーブルを構築する。元実装の per-x ループ + per-call
        cell resize を排して、長い strip でも O(templates) 回で済むようにする
        (gemini-code-assist Medium)。

        upright としても高スコアになる x は除外する (元実装と同じ「横向き優位」
        判定)。一致なしなら `(-1, -inf, None)`。
        """
        gh, gw = group.shape
        n_x_h = gw - w_rot + 1
        if n_x_h <= 0:
            return -1, float("-inf"), None

        # 横向きテンプレ (CCW + CW) で strip 最下段の score map を構築する。
        bottom_strip = group[gh - h_rot : gh, :]
        h_max_scores = np.full(n_x_h, -np.inf, dtype=np.float64)
        h_max_codes: list[str | None] = [None] * n_x_h
        for templates in (self._templates_rotated_ccw, self._templates_rotated_cw):
            for code, tmpl in templates.items():
                res = cv2.matchTemplate(bottom_strip, tmpl, cv2.TM_CCOEFF_NORMED)
                scores = res[0]  # shape (n_x_h,)
                improved = scores > h_max_scores
                if not improved.any():
                    continue
                h_max_scores = np.where(improved, scores, h_max_scores)
                for idx in np.flatnonzero(improved):
                    h_max_codes[int(idx)] = code

        # 同じ x 起点で「もし upright 牌だったら」の score map も 1 度ずつ取る。
        # upright と horizontal で幅 (w_up vs w_rot) が違うため n_x_up > n_x_h だが、
        # 比較は x ∈ [0, n_x_h) のみで行う (x_h と x_up を「左端を揃えて比較」)。
        upright_strip = group[max(0, gh - h_up) : gh, :]
        n_x_up = max(0, gw - w_up + 1)
        up_max_scores = np.full(n_x_up, -np.inf, dtype=np.float64)
        if upright_strip.shape[0] >= h_up and n_x_up > 0:
            for tmpl in self._templates_upright.values():
                res = cv2.matchTemplate(upright_strip, tmpl, cv2.TM_CCOEFF_NORMED)
                up_max_scores = np.maximum(up_max_scores, res[0])

        best_x = -1
        best_score = float("-inf")
        best_code: str | None = None
        for x in range(n_x_h):
            h_score = h_max_scores[x]
            if not np.isfinite(h_score):
                continue
            up_score = up_max_scores[x] if x < n_x_up else -np.inf
            if h_score > up_score + HORIZONTAL_SCORE_MARGIN and h_score > best_score:
                best_score = float(h_score)
                best_x = x
                best_code = h_max_codes[x]
        return best_x, best_score, best_code

    def _segment_upright_run(self, group: np.ndarray, x0: int, x1: int) -> list[_TileSlot]:
        """指定範囲 [x0, x1) を upright 牌幅で等分してマッチする。

        テンプレ 1 つにつき `cv2.matchTemplate` 1 回で run 全幅の score map を取り、
        固定スロット位置 (`local_x = i * w_up`) の列を抜き出して各スロットの
        最良コードを決める (gemini-code-assist Medium on PR #46)。
        """
        if self._size_upright is None:
            return []
        h_up, w_up = self._size_upright
        gh = group.shape[0]
        run_w = x1 - x0
        if run_w < w_up:
            return []
        n = run_w // w_up
        if n == 0:
            return []
        y0 = max(0, gh - h_up)
        upright_strip = group[y0:gh, x0:x1]
        if upright_strip.shape[0] < h_up:
            return []
        n_x = upright_strip.shape[1] - w_up + 1
        if n_x <= 0:
            return []

        # 全テンプレの score map を (37, n_x) 行列にまとめ、列ごとの argmax を取る。
        codes = list(self._templates_upright.keys())
        score_map = np.empty((len(codes), n_x), dtype=np.float64)
        for i, code in enumerate(codes):
            res = cv2.matchTemplate(
                upright_strip, self._templates_upright[code], cv2.TM_CCOEFF_NORMED
            )
            score_map[i] = res[0]

        slots: list[_TileSlot] = []
        for i in range(n):
            sx = x0 + i * w_up
            ex = x1 if i == n - 1 else x0 + (i + 1) * w_up
            local_x = i * w_up
            if local_x >= n_x:
                continue
            # 空白セルはスキップ (= 鳴かれた牌の隙間や帯端のはみ出し)。
            cell = upright_strip[:, local_x : local_x + w_up]
            if cell.size == 0 or float(cell.std()) < BLANK_STD_THRESHOLD:
                continue
            scores_at_x = score_map[:, local_x]
            best_idx = int(np.argmax(scores_at_x))
            slots.append(
                _TileSlot(
                    code=codes[best_idx],
                    score=float(scores_at_x[best_idx]),
                    is_horizontal=False,
                    x_in_group=sx,
                    width=ex - sx,
                )
            )
        return slots

    def _detect_kakan_stack(
        self,
        gray: np.ndarray,
        gx0: int,
        gx1: int,
        slots: list[_TileSlot],
    ) -> bool:
        """横向きスロットの上に 2 枚目の横向き牌が積まれているか判定 (加槓判定)。

        ROI の縦余白が足りず stack 領域が strip 上端を超える場合は False を返す
        (= minkan にフォールバック)。
        """
        if self._size_rotated is None:
            return False
        h_horiz, _ = self._size_rotated  # 横向き牌の高さ = W
        h_slot = next((s for s in slots if s.is_horizontal), None)
        if h_slot is None:
            return False
        group = gray[:, gx0:gx1]
        gh, _ = group.shape
        # 下段の横向き牌は y ∈ [gh - W, gh]、その上の積み牌は y ∈ [gh - 2W, gh - W]
        stack_y0 = gh - 2 * h_horiz
        stack_y1 = gh - h_horiz
        if stack_y0 < 0 or stack_y1 <= 0:
            return False
        cell = group[stack_y0:stack_y1, h_slot.x_in_group : h_slot.x_in_group + h_slot.width]
        if cell.size == 0 or float(cell.std()) < BLANK_STD_THRESHOLD:
            return False
        _, ccw_score = self._match_at_size(cell, self._templates_rotated_ccw, self._size_rotated)
        _, cw_score = self._match_at_size(cell, self._templates_rotated_cw, self._size_rotated)
        return max(ccw_score, cw_score) >= KAKAN_STACK_NCC_THRESHOLD

    def _classify_meld(
        self, slots: list[_TileSlot], kakan_stack: bool, player_idx: int
    ) -> Meld | None:
        """(枚数, 横向き枚数, 横向き位置, スタック有無) で副露種別を決める。

        pon / minkan / kakan は「全 3 (or 4) 牌が同一」が必須条件 (麻雀ルール)。
        誤認識で 1 枚でも異なると `["1m","1m","2m"]` のような不正 meld を
        tenhou_json に出して downstream の状態を壊すため、同一でなければ
        None を返して該当グループを無視する (coderabbitai Major on PR #46)。
        """
        n = len(slots)
        h_indices = [i for i, s in enumerate(slots) if s.is_horizontal]
        h_count = len(h_indices)
        tiles = [s.code for s in slots]
        all_same = bool(tiles) and len(set(tiles)) == 1

        if n == 4:
            if h_count == 0:
                # 4 枚すべて表牌スコアが低ければ ankan (= 全て裏向き)。
                max_score = max(s.score for s in slots) if slots else -1.0
                if max_score < ANKAN_NCC_THRESHOLD:
                    # 裏向き牌からは具体的な牌種を識別できないため `tiles` は空にする。
                    # ランダム低スコアの best-effort code を 4 つ並べると Mortal を惑わす
                    # (gemini-code-assist High on PR #46)。自家 ankan のみ face-up
                    # 検出する追補と裏向き専用テンプレ整備は #16 follow-up。
                    return Meld(player=player_idx, type="ankan", tiles=[], from_=0)
                return None
            if h_count == 1 and all_same:
                from_ = self._horizontal_position_to_from(h_indices[0], n)
                return Meld(player=player_idx, type="minkan", tiles=tiles, from_=from_)
            return None

        if n == 3:
            if h_count != 1:
                return None
            from_ = self._horizontal_position_to_from(h_indices[0], n)
            if kakan_stack and all_same:
                # 加槓: 4 枚目はポンと同種なので、横向き牌の code を複製して 4 枚にする。
                tiles_with_stack = list(tiles)
                tiles_with_stack.append(slots[h_indices[0]].code)
                return Meld(player=player_idx, type="kakan", tiles=tiles_with_stack, from_=from_)
            # チー判定は「横向きが左端 (= 上家由来) かつ 3 牌が同一スーツの連続数」のとき。
            if from_ == 3 and self._is_chi_sequence(tiles):
                return Meld(player=player_idx, type="chi", tiles=tiles, from_=3)
            if all_same:
                return Meld(player=player_idx, type="pon", tiles=tiles, from_=from_)
            return None

        return None

    @staticmethod
    def _horizontal_position_to_from(h_index: int, total: int) -> int:
        """横向き牌の位置から「鳴いた相手」の相対座順を返す。

        upright 正規化済みの座標で左から 0, 1, ..., total-1。
        - 左端 (h_index == 0) → 上家 (3)
        - 右端 (h_index == total - 1) → 下家 (1)
        - それ以外 → 対面 (2)
        """
        if h_index == 0:
            return 3
        if h_index == total - 1:
            return 1
        return 2

    @staticmethod
    def _is_chi_sequence(tiles: list[str]) -> bool:
        """3 牌が同一スーツ (m/p/s) の連続数 (例: 1m, 2m, 3m) か判定。

        赤 5 (0m/0p/0s) は数字 5 として扱う (4m, 0m, 6m はチー成立)。
        字牌 (1z..7z) は連続数の概念がないので必ず False。
        """
        if len(tiles) != 3:
            return False
        suits = {t[-1] for t in tiles}
        if len(suits) != 1:
            return False
        suit = next(iter(suits))
        if suit not in ("m", "p", "s"):
            return False
        try:
            nums = [int(t[0]) for t in tiles]
        except ValueError:
            return False
        nums_normalized = sorted(5 if n == 0 else n for n in nums)
        return all(
            nums_normalized[i] + 1 == nums_normalized[i + 1]
            for i in range(len(nums_normalized) - 1)
        )

    def _match_at_size(
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
