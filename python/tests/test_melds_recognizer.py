"""melds_recognizer の単体テスト (issue #14)。

実テンプレ画像 (issue #16) は未配備のため、test_river_recognizer と同様に
ランダム合成テンプレを一時ディレクトリに書き出して挙動を検証する。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from recognition.melds_recognizer import (
    ANKAN_NCC_THRESHOLD,
    MELD_PLAYER_KEYS,
    Meld,
    MeldsRecognizer,
)
from recognition.tile_recognizer import TILE_CODES

TMPL_H = 32  # upright 牌の高さ (= 横向き牌の幅)
TMPL_W = 24  # upright 牌の幅 (= 横向き牌の高さ)
# kakan の積み牌が乗るため、副露の strip 高さは 2*W 以上必要。テストでは max(H, 2W) に揃える。
STRIP_H = max(TMPL_H, 2 * TMPL_W)


def _make_tile_template(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(TMPL_H, TMPL_W), dtype=np.uint8)


def _write_all_templates(dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for i, code in enumerate(TILE_CODES):
        cv2.imwrite(str(dst / f"{code}.png"), _make_tile_template(i))


def _facedown_cell() -> np.ndarray:
    """裏向き牌セル: 表テンプレ 37 種いずれにも低 NCC しか出ないランダム模様。"""
    rng = np.random.default_rng(999)
    return rng.integers(0, 256, size=(TMPL_H, TMPL_W), dtype=np.uint8)


def _make_blank_canvas(width: int, height: int = STRIP_H, bg: int = 200) -> np.ndarray:
    return np.full((height, width), bg, dtype=np.uint8)


def _draw_upright(canvas: np.ndarray, x: int, code: str) -> int:
    """upright 牌を canvas[strip_h - TMPL_H : strip_h, x : x + TMPL_W] に描画。"""
    sh = canvas.shape[0]
    tmpl = _make_tile_template(TILE_CODES.index(code))
    canvas[sh - TMPL_H : sh, x : x + TMPL_W] = tmpl
    return TMPL_W


def _draw_horizontal(canvas: np.ndarray, x: int, code: str, *, ccw: bool = True) -> int:
    """横向き牌を canvas[strip_h - TMPL_W : strip_h, x : x + TMPL_H] に描画。"""
    sh = canvas.shape[0]
    tmpl = _make_tile_template(TILE_CODES.index(code))
    rotated = cv2.rotate(tmpl, cv2.ROTATE_90_COUNTERCLOCKWISE if ccw else cv2.ROTATE_90_CLOCKWISE)
    canvas[sh - TMPL_W : sh, x : x + TMPL_H] = rotated
    return TMPL_H


def _draw_kakan_stack(canvas: np.ndarray, x: int, code: str, *, ccw: bool = True) -> None:
    """加槓の積み牌 (横向き) を bottom 横向き牌の上に描画。"""
    sh = canvas.shape[0]
    tmpl = _make_tile_template(TILE_CODES.index(code))
    rotated = cv2.rotate(tmpl, cv2.ROTATE_90_COUNTERCLOCKWISE if ccw else cv2.ROTATE_90_CLOCKWISE)
    # 積み牌は bottom 横向き (y ∈ [sh - W, sh]) の上 (y ∈ [sh - 2W, sh - W])。
    canvas[sh - 2 * TMPL_W : sh - TMPL_W, x : x + TMPL_H] = rotated


def _draw_facedown(canvas: np.ndarray, x: int) -> int:
    """裏向き牌を upright スロットと同じサイズで描画。"""
    sh = canvas.shape[0]
    canvas[sh - TMPL_H : sh, x : x + TMPL_W] = _facedown_cell()
    return TMPL_W


def _to_screen_orientation(upright_panel: np.ndarray, player_idx: int) -> np.ndarray:
    """upright 向きで組んだパネルを「画面上で見える向き」へ pre-rotate する。

    melds_recognizer は他家の crop を player_idx 別に逆回転して upright へ
    正規化するため、テスト入力は逆方向 (upright → 画面向き) を仕込む。
    """
    if player_idx == 0:
        return upright_panel
    if player_idx == 1:  # 下家: 画面で 90° CW
        return cv2.rotate(upright_panel, cv2.ROTATE_90_CLOCKWISE)
    if player_idx == 2:  # 対面: 画面で 180°
        return cv2.rotate(upright_panel, cv2.ROTATE_180)
    if player_idx == 3:  # 上家: 画面で 90° CCW
        return cv2.rotate(upright_panel, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError(player_idx)


# ============================================================================
# Meld dataclass
# ============================================================================


def test_meld_to_dict_contains_player_type_tiles_from() -> None:
    m = Meld(player=2, type="pon", tiles=["5p", "5p", "5p"], from_=1)
    assert m.to_dict() == {
        "player": 2,
        "type": "pon",
        "tiles": ["5p", "5p", "5p"],
        "from": 1,
    }


def test_meld_to_dict_returns_fresh_tiles_list() -> None:
    m = Meld(player=0, type="ankan", tiles=["1z", "1z", "1z", "1z"], from_=0)
    d = m.to_dict()
    d["tiles"].append("X")
    assert m.tiles == ["1z", "1z", "1z", "1z"]  # 元 dataclass は副作用を受けない


def test_meld_to_dict_includes_called_index_when_set() -> None:
    """called_index が設定されている場合は to_dict に含まれる (チー用)。"""
    m = Meld(player=0, type="chi", tiles=["3m", "4m", "5m"], from_=3, called_index=1)
    d = m.to_dict()
    assert d == {
        "player": 0,
        "type": "chi",
        "tiles": ["3m", "4m", "5m"],
        "from": 3,
        "called_index": 1,
    }


def test_meld_to_dict_omits_called_index_when_none() -> None:
    """called_index が None (ポン/ミンカン/アンカン/カカン) では出力に含めない。"""
    m = Meld(player=2, type="pon", tiles=["5p", "5p", "5p"], from_=1)
    d = m.to_dict()
    assert "called_index" not in d


# ============================================================================
# Graceful degradation
# ============================================================================


def test_returns_empty_when_templates_missing(tmp_path: Path) -> None:
    rec = MeldsRecognizer(tmp_path)
    bgr = np.zeros((100, 100, 3), dtype=np.uint8)
    melds = {"self": {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0}}
    assert rec.recognize_melds(bgr, melds) == []


def test_returns_empty_when_meld_rois_missing(tmp_path: Path) -> None:
    _write_all_templates(tmp_path)
    rec = MeldsRecognizer(tmp_path)
    bgr = np.zeros((100, 100, 3), dtype=np.uint8)
    assert rec.recognize_melds(bgr, None) == []
    assert rec.recognize_melds(bgr, "not a dict") == []  # type: ignore[arg-type]


def test_returns_empty_when_frame_is_empty(tmp_path: Path) -> None:
    _write_all_templates(tmp_path)
    rec = MeldsRecognizer(tmp_path)
    melds = {key: {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0} for key in MELD_PLAYER_KEYS}
    assert rec.recognize_melds(None, melds) == []
    assert rec.recognize_melds(np.zeros((0, 0, 3), dtype=np.uint8), melds) == []


def test_no_roi_warning_logged_once_per_player(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _write_all_templates(tmp_path)
    rec = MeldsRecognizer(tmp_path)
    bgr = np.zeros((100, 100, 3), dtype=np.uint8)
    with caplog.at_level("WARNING", logger="recognition"):
        for _ in range(5):
            rec.recognize_melds(bgr, {})
    no_roi_warnings = [
        r
        for r in caplog.records
        if "meld ROI" in r.getMessage() and "not calibrated" in r.getMessage()
    ]
    assert len(no_roi_warnings) == 4


@pytest.mark.parametrize(
    "rect",
    [
        {"x": 0.0, "y": 0.0, "w": 0.0, "h": 1.0},
        {"x": 0.0, "y": 0.0, "w": 1.0, "h": 0.0},
    ],
)
def test_recognize_handles_degenerate_roi(tmp_path: Path, rect: dict) -> None:
    _write_all_templates(tmp_path)
    rec = MeldsRecognizer(tmp_path)
    bgr = np.zeros((200, 200, 3), dtype=np.uint8)
    assert rec.recognize_melds(bgr, {"self": rect}) == []


def test_partial_template_set_fails_closed(tmp_path: Path) -> None:
    """部分セット (9/37) では recognize_melds が無効化される。"""
    for code in TILE_CODES[:9]:
        cv2.imwrite(
            str(tmp_path / f"{code}.png"),
            _make_tile_template(TILE_CODES.index(code)),
        )
    rec = MeldsRecognizer(tmp_path)
    bgr = np.full((STRIP_H, 200, 3), 128, dtype=np.uint8)
    melds = {"self": {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0}}
    assert rec.recognize_melds(bgr, melds) == []


# ============================================================================
# 副露種別の認識
# ============================================================================


def _build_and_recognize(
    tmp_path: Path,
    strip_layout,
    strip_w: int = 200,
    player_key: str = "self",
    player_idx: int = 0,
) -> list[dict]:
    """1 家分の strip を canvas に貼り、recognize_melds を呼んで結果を返す。"""
    _write_all_templates(tmp_path)
    rec = MeldsRecognizer(tmp_path)

    canvas = _make_blank_canvas(strip_w, STRIP_H)
    strip_layout(canvas)

    # 他家でテストする場合は画面向きに pre-rotate する。
    if player_idx != 0:
        canvas = _to_screen_orientation(canvas, player_idx)
    bgr = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

    rois = {player_key: {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0}}
    return rec.recognize_melds(bgr, rois)


def test_recognize_chi_from_kamicha(tmp_path: Path) -> None:
    """連続スーツ 3 牌 + 左横向き → チー、from=3 (上家)、called_index=0。"""

    def layout(c: np.ndarray) -> None:
        x = 4
        x += _draw_horizontal(c, x, "2m")  # 鳴いた牌 (左)
        x += _draw_upright(c, x, "3m")
        x += _draw_upright(c, x, "4m")

    result = _build_and_recognize(tmp_path, layout)
    assert len(result) == 1
    m = result[0]
    assert m["type"] == "chi"
    assert m["from"] == 3
    assert m["tiles"] == ["2m", "3m", "4m"]
    assert m["player"] == 0
    # called_index は鳴いた牌の位置 (CodeRabbit Critical on PR #51)。
    assert m["called_index"] == 0


def test_recognize_chi_with_middle_horizontal(tmp_path: Path) -> None:
    """連続スーツ 3 牌 + 中央横向き (= 中央の牌を鳴いた) → チー、from=3。

    雀魂は副露牌を昇順表示するため、4m+5m+6m の連続で 5m を鳴いた場合、
    中央が横向きになる。チーは上家からしか鳴けないので、横向きの位置に
    関わらず from=3 になるのが正しい (chatgpt-codex P1 on PR #46)。
    """

    def layout(c: np.ndarray) -> None:
        x = 4
        x += _draw_upright(c, x, "4m")
        x += _draw_horizontal(c, x, "5m")  # 鳴いた牌 (中央)
        x += _draw_upright(c, x, "6m")

    result = _build_and_recognize(tmp_path, layout)
    assert len(result) == 1
    m = result[0]
    assert m["type"] == "chi"
    assert m["from"] == 3
    assert m["tiles"] == ["4m", "5m", "6m"]
    assert m["called_index"] == 1


def test_recognize_chi_with_right_horizontal(tmp_path: Path) -> None:
    """連続スーツ 3 牌 + 右横向き (= 最大の牌を鳴いた) → チー、from=3、called_index=2。

    7p+8p+9p で 9p を鳴いた場合は右端が横向きになる。
    """

    def layout(c: np.ndarray) -> None:
        x = 4
        x += _draw_upright(c, x, "7p")
        x += _draw_upright(c, x, "8p")
        x += _draw_horizontal(c, x, "9p")  # 鳴いた牌 (右)

    result = _build_and_recognize(tmp_path, layout)
    assert len(result) == 1
    m = result[0]
    assert m["type"] == "chi"
    assert m["from"] == 3
    assert m["tiles"] == ["7p", "8p", "9p"]
    assert m["called_index"] == 2


def test_recognize_pon_from_kamicha(tmp_path: Path) -> None:
    """同一 3 牌 + 左横向き → ポン、from=3 (上家)。"""

    def layout(c: np.ndarray) -> None:
        x = 4
        x += _draw_horizontal(c, x, "5p")
        x += _draw_upright(c, x, "5p")
        x += _draw_upright(c, x, "5p")

    result = _build_and_recognize(tmp_path, layout)
    assert len(result) == 1
    m = result[0]
    assert m["type"] == "pon"
    assert m["from"] == 3
    assert m["tiles"] == ["5p", "5p", "5p"]


def test_recognize_pon_from_toimen(tmp_path: Path) -> None:
    """中央横向き → ポン、from=2 (対面)。"""

    def layout(c: np.ndarray) -> None:
        x = 4
        x += _draw_upright(c, x, "5p")
        x += _draw_horizontal(c, x, "5p")
        x += _draw_upright(c, x, "5p")

    result = _build_and_recognize(tmp_path, layout)
    assert len(result) == 1
    m = result[0]
    assert m["type"] == "pon"
    assert m["from"] == 2


def test_recognize_pon_from_shimocha(tmp_path: Path) -> None:
    """右横向き → ポン、from=1 (下家)。"""

    def layout(c: np.ndarray) -> None:
        x = 4
        x += _draw_upright(c, x, "5p")
        x += _draw_upright(c, x, "5p")
        x += _draw_horizontal(c, x, "5p")

    result = _build_and_recognize(tmp_path, layout)
    assert len(result) == 1
    m = result[0]
    assert m["type"] == "pon"
    assert m["from"] == 1


@pytest.mark.parametrize(
    "horiz_index,expected_from",
    [(0, 3), (1, 2), (3, 1)],
    ids=["from_kamicha_left", "from_toimen_index1", "from_shimocha_right"],
)
def test_recognize_minkan_from_various_sources(
    tmp_path: Path, horiz_index: int, expected_from: int
) -> None:
    """4 牌 + 横向き 1 = 大明槓。横向き位置で from を決定。"""

    def layout(c: np.ndarray) -> None:
        x = 4
        for i in range(4):
            if i == horiz_index:
                x += _draw_horizontal(c, x, "1z")
            else:
                x += _draw_upright(c, x, "1z")

    result = _build_and_recognize(tmp_path, layout)
    assert len(result) == 1
    m = result[0]
    assert m["type"] == "minkan"
    assert m["from"] == expected_from
    assert len(m["tiles"]) == 4


def test_recognize_ankan(tmp_path: Path) -> None:
    """4 セルすべて裏向き (低 NCC) → 暗槓、from=0。

    `tiles` は空 list で返る: 裏向き牌からは具体的な牌種を識別できないため、
    ランダム低スコアの best-effort code を出力すると Mortal を惑わす
    (gemini-code-assist High on PR #46)。
    """

    def layout(c: np.ndarray) -> None:
        x = 4
        for _ in range(4):
            x += _draw_facedown(c, x)

    result = _build_and_recognize(tmp_path, layout)
    assert len(result) == 1
    m = result[0]
    assert m["type"] == "ankan"
    assert m["from"] == 0
    assert m["tiles"] == []


def test_ankan_threshold_constant_consistency() -> None:
    """ANKAN_NCC_THRESHOLD は 0.4 (副露種別判定の境界値)。

    値を変える際は test_recognize_ankan の裏向きセル設計も合わせて確認すべき
    なので、固定値レグレッションを置く。
    """
    assert pytest.approx(0.4) == ANKAN_NCC_THRESHOLD


def test_recognize_kakan(tmp_path: Path) -> None:
    """1 横向き + 上スタック + 2 upright → 加槓。"""
    code = "7z"

    def layout(c: np.ndarray) -> None:
        x = 4
        # 横向き牌 + その上に積み牌
        h_x = x
        x += _draw_horizontal(c, x, code)
        _draw_kakan_stack(c, h_x, code)
        x += _draw_upright(c, x, code)
        x += _draw_upright(c, x, code)

    result = _build_and_recognize(tmp_path, layout)
    assert len(result) == 1
    m = result[0]
    assert m["type"] == "kakan"
    assert m["from"] == 3
    assert m["tiles"] == [code, code, code, code]  # 4 枚目は元ポンと同種


def test_recognize_kakan_falls_back_to_pon_when_strip_too_short(tmp_path: Path) -> None:
    """ROI の縦余白が足りず stack 領域を含まない strip では kakan 検出は失敗し
    pon として返る (graceful degrade)。"""
    _write_all_templates(tmp_path)
    rec = MeldsRecognizer(tmp_path)

    # strip 高さ = TMPL_H (= 32) のみ。stack 領域 (y ∈ [-16, 8]) は範囲外で kakan 検出不可。
    canvas = _make_blank_canvas(200, height=TMPL_H)
    code = "1m"
    x = 4
    x += _draw_horizontal(canvas, x, code)
    x += _draw_upright(canvas, x, code)
    x += _draw_upright(canvas, x, code)

    bgr = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    rois = {"self": {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0}}
    result = rec.recognize_melds(bgr, rois)
    assert len(result) == 1
    assert result[0]["type"] == "pon"  # stack 検出できないため pon に倒れる


def test_recognize_multiple_melds_same_player(tmp_path: Path) -> None:
    """1 帯に 2 副露 (隙間あり) を仕込むと 2 件返る。"""

    def layout(c: np.ndarray) -> None:
        # 1 副露目: ポン (5p × 3, 左横向き)
        x = 4
        x += _draw_horizontal(c, x, "5p")
        x += _draw_upright(c, x, "5p")
        x += _draw_upright(c, x, "5p")
        # 隙間
        x += 8
        # 2 副露目: チー (1m, 2m, 3m, 左横向き)
        x += _draw_horizontal(c, x, "1m")
        x += _draw_upright(c, x, "2m")
        x += _draw_upright(c, x, "3m")

    result = _build_and_recognize(tmp_path, layout, strip_w=240)
    assert len(result) == 2
    # 左 → 右の順に並ぶ
    assert result[0]["type"] == "pon"
    assert result[0]["tiles"] == ["5p", "5p", "5p"]
    assert result[1]["type"] == "chi"
    assert result[1]["tiles"] == ["1m", "2m", "3m"]


def test_pon_with_mismatched_tiles_returns_none(tmp_path: Path) -> None:
    """3 牌 + 左横向き でも全牌が同一でなければ meld を出さない (fail-closed)。

    `["1m", "3m", "5m"]` は同スーツだが連続ではないので chi にもならず、
    また同一でもないので pon にもしない (coderabbitai Major on PR #46)。
    """

    def layout(c: np.ndarray) -> None:
        x = 4
        x += _draw_horizontal(c, x, "1m")
        x += _draw_upright(c, x, "3m")
        x += _draw_upright(c, x, "5m")

    result = _build_and_recognize(tmp_path, layout)
    assert result == []


def test_minkan_with_mismatched_tiles_returns_none(tmp_path: Path) -> None:
    """4 牌 + 横向き 1 でも全牌が同一でなければ minkan を出さない (fail-closed)。"""

    def layout(c: np.ndarray) -> None:
        x = 4
        x += _draw_horizontal(c, x, "1m")
        x += _draw_upright(c, x, "1m")
        x += _draw_upright(c, x, "2m")  # 異なる牌種を混ぜる
        x += _draw_upright(c, x, "1m")

    result = _build_and_recognize(tmp_path, layout)
    assert result == []


def test_low_confidence_horizontal_does_not_break_ankan_detection(tmp_path: Path) -> None:
    """4 セル裏向きの strip でノイズの「横向き候補」が出ても ankan 判定が
    優先される (coderabbitai Major on PR #46)。

    `_find_best_horizontal` は ANKAN_NCC_THRESHOLD 未満の候補を採用しないため、
    全テンプレに対し低 NCC のグループは h_count=0 として ankan branch に流れる。
    """

    def layout(c: np.ndarray) -> None:
        x = 4
        for _ in range(4):
            x += _draw_facedown(c, x)

    result = _build_and_recognize(tmp_path, layout)
    assert len(result) == 1
    assert result[0]["type"] == "ankan"
    assert result[0]["tiles"] == []


def test_chi_vs_pon_disambiguation(tmp_path: Path) -> None:
    """同レイアウト (横向き左端 + 縦 2) で、コードが連続 → chi、同一 → pon に分かれる。"""

    def chi_layout(c: np.ndarray) -> None:
        x = 4
        x += _draw_horizontal(c, x, "1m")
        x += _draw_upright(c, x, "2m")
        x += _draw_upright(c, x, "3m")

    def pon_layout(c: np.ndarray) -> None:
        x = 4
        x += _draw_horizontal(c, x, "1m")
        x += _draw_upright(c, x, "1m")
        x += _draw_upright(c, x, "1m")

    chi_result = _build_and_recognize(tmp_path, chi_layout)
    assert len(chi_result) == 1
    assert chi_result[0]["type"] == "chi"

    pon_result = _build_and_recognize(tmp_path, pon_layout)
    assert len(pon_result) == 1
    assert pon_result[0]["type"] == "pon"


def test_recognize_all_four_players_with_one_meld_each(tmp_path: Path) -> None:
    """4 家分の ROI を別々に切って与えると、それぞれ player 0..3 で返る。

    他家のパネルは画面上の向き (右家=90° CW, 対面=180°, 左家=90° CCW) へ
    pre-rotate して仕込む。melds recognizer が player_idx で逆回転して upright
    に正規化することを検証する。
    """
    _write_all_templates(tmp_path)
    rec = MeldsRecognizer(tmp_path)

    def _one_pon_panel(code: str) -> np.ndarray:
        # ポン (左横向き = from kamicha) を 1 つだけ含む小さな strip
        panel_w = TMPL_H + 2 * TMPL_W + 8  # 左端マージン込みでぴったり
        panel = _make_blank_canvas(panel_w, STRIP_H)
        x = 4
        x += _draw_horizontal(panel, x, code)
        x += _draw_upright(panel, x, code)
        x += _draw_upright(panel, x, code)
        return panel

    upright_codes = {"self": "1m", "right": "2p", "across": "3s", "left": "1z"}
    panels_screen = {
        key: _to_screen_orientation(_one_pon_panel(code), player_idx)
        for player_idx, (key, code) in enumerate(upright_codes.items())
    }

    # CW/CCW 後は (h, w) が入れ替わるため、どの向きでも収まる正方形セルへ
    # 各家を左上寄せで配置し、ROI は実パネル寸に合わせて切り出す。
    cell_size = max(p.shape[0] for p in panels_screen.values()) + max(
        p.shape[1] for p in panels_screen.values()
    )
    full_h = cell_size * 2
    full_w = cell_size * 2
    full = np.full((full_h, full_w), 128, dtype=np.uint8)
    offsets = {
        "self": (0, 0),
        "right": (0, cell_size),
        "across": (cell_size, 0),
        "left": (cell_size, cell_size),
    }
    rois: dict[str, dict[str, float]] = {}
    for key, (oy, ox) in offsets.items():
        panel = panels_screen[key]
        ph, pw = panel.shape
        full[oy : oy + ph, ox : ox + pw] = panel
        rois[key] = {
            "x": ox / full_w,
            "y": oy / full_h,
            "w": pw / full_w,
            "h": ph / full_h,
        }
    bgr = cv2.cvtColor(full, cv2.COLOR_GRAY2BGR)

    result = rec.recognize_melds(bgr, rois)
    assert len(result) == 4
    by_player = {r["player"]: r for r in result}
    assert by_player[0]["tiles"] == ["1m", "1m", "1m"]
    assert by_player[1]["tiles"] == ["2p", "2p", "2p"]
    assert by_player[2]["tiles"] == ["3s", "3s", "3s"]
    assert by_player[3]["tiles"] == ["1z", "1z", "1z"]
    for m in result:
        assert m["type"] == "pon"
        assert m["from"] == 3  # 全て上家由来 (左横向き) で揃えた


# ============================================================================
# 内部ロジック (_is_chi_sequence) の単体検証
# ============================================================================


@pytest.mark.parametrize(
    "tiles,expected",
    [
        (["1m", "2m", "3m"], True),
        (["7p", "8p", "9p"], True),
        (["3s", "4s", "5s"], True),
        (["4m", "0m", "6m"], True),  # 赤 5 を 5 として扱う
        (["1m", "1m", "1m"], False),  # 同一 → pon
        (["1m", "2p", "3s"], False),  # 異スーツ
        (["1z", "2z", "3z"], False),  # 字牌は連続不可
        (["1m", "3m", "5m"], False),  # 飛び
        (["1m", "2m"], False),  # 枚数不一致
        (["9m", "8m", "7m"], True),  # 順不同
    ],
)
def test_is_chi_sequence(tiles: list[str], expected: bool) -> None:
    assert MeldsRecognizer._is_chi_sequence(tiles) is expected


@pytest.mark.parametrize(
    "h_index,total,expected_from",
    [
        (0, 3, 3),
        (1, 3, 2),
        (2, 3, 1),
        (0, 4, 3),
        (1, 4, 2),
        (2, 4, 2),
        (3, 4, 1),
    ],
)
def test_horizontal_position_to_from(h_index: int, total: int, expected_from: int) -> None:
    assert MeldsRecognizer._horizontal_position_to_from(h_index, total) == expected_from
