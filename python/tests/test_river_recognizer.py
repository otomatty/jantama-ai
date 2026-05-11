"""river_recognizer の単体テスト (issue #13)。

実テンプレ画像 (issue #16) は未配備のため、test_tile_recognizer と同様に
ランダム合成テンプレを一時ディレクトリに書き出して挙動を検証する。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from recognition.river_recognizer import (
    RIVER_COLS,
    RIVER_PLAYER_KEYS,
    RIVER_ROWS,
    RiverRecognizer,
    RiverTile,
)
from recognition.tile_recognizer import BLANK_STD_THRESHOLD, TILE_CODES

TMPL_H = 32
TMPL_W = 24


def _make_tile_template(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(TMPL_H, TMPL_W), dtype=np.uint8)


def _write_all_templates(dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for i, code in enumerate(TILE_CODES):
        cv2.imwrite(str(dst / f"{code}.png"), _make_tile_template(i))


def _blank_cell() -> np.ndarray:
    cell = np.full((TMPL_H, TMPL_W), 200, dtype=np.uint8)
    assert float(cell.std()) < BLANK_STD_THRESHOLD
    return cell


def _make_player_canvas(codes_grid: list[list[str | None]]) -> np.ndarray:
    """`codes_grid[row][col]` を upright で 6 列 × 4 段の河画像に並べる。

    `None` は空セル (平坦) を意味する。
    """
    assert len(codes_grid) == RIVER_ROWS
    rows: list[np.ndarray] = []
    for row in codes_grid:
        assert len(row) == RIVER_COLS
        cells = [
            _make_tile_template(TILE_CODES.index(c)) if c is not None else _blank_cell()
            for c in row
        ]
        rows.append(np.concatenate(cells, axis=1))
    return np.concatenate(rows, axis=0)


def test_river_tile_to_dict_default_no_riichi() -> None:
    t = RiverTile(player=2, tile="3p")
    assert t.to_dict() == {"player": 2, "tile": "3p", "tedashi": True}


def test_river_tile_to_dict_includes_riichi_when_set() -> None:
    t = RiverTile(player=1, tile="5z", riichi=True)
    assert t.to_dict() == {
        "player": 1,
        "tile": "5z",
        "tedashi": True,
        "riichi": True,
    }


def test_returns_empty_when_templates_missing(tmp_path: Path) -> None:
    rec = RiverRecognizer(tmp_path)
    bgr = np.zeros((100, 100, 3), dtype=np.uint8)
    rivers = {"self": {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0}}
    assert rec.recognize_rivers(bgr, rivers) == []


def test_returns_empty_when_rivers_dict_missing(tmp_path: Path) -> None:
    _write_all_templates(tmp_path)
    rec = RiverRecognizer(tmp_path)
    bgr = np.zeros((100, 100, 3), dtype=np.uint8)
    assert rec.recognize_rivers(bgr, None) == []
    assert rec.recognize_rivers(bgr, "not a dict") == []  # type: ignore[arg-type]


def test_returns_empty_when_frame_is_empty(tmp_path: Path) -> None:
    _write_all_templates(tmp_path)
    rec = RiverRecognizer(tmp_path)
    rivers = {key: {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0} for key in RIVER_PLAYER_KEYS}
    assert rec.recognize_rivers(None, rivers) == []
    assert rec.recognize_rivers(np.zeros((0, 0, 3), dtype=np.uint8), rivers) == []


def test_no_roi_warning_logged_once_per_player(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _write_all_templates(tmp_path)
    rec = RiverRecognizer(tmp_path)
    bgr = np.zeros((100, 100, 3), dtype=np.uint8)
    # 4 家全てキャリブ無しで 5 サイクル走らせても、各家 1 回ずつしか warn しない。
    with caplog.at_level("WARNING", logger="recognition"):
        for _ in range(5):
            rec.recognize_rivers(bgr, {})
    no_roi_warnings = [
        r
        for r in caplog.records
        if "river ROI" in r.getMessage() and "not calibrated" in r.getMessage()
    ]
    assert len(no_roi_warnings) == 4


def test_recognize_self_player_full_river(tmp_path: Path) -> None:
    """自家の 6×4 = 24 牌すべて埋まったケースで、全 24 牌を正しく識別する。"""
    _write_all_templates(tmp_path)
    rec = RiverRecognizer(tmp_path)
    # 24 牌分を TILE_CODES から循環ピック
    codes_flat = [TILE_CODES[i % len(TILE_CODES)] for i in range(RIVER_ROWS * RIVER_COLS)]
    grid: list[list[str | None]] = [
        codes_flat[r * RIVER_COLS : (r + 1) * RIVER_COLS] for r in range(RIVER_ROWS)
    ]
    canvas_gray = _make_player_canvas(grid)
    bgr = cv2.cvtColor(canvas_gray, cv2.COLOR_GRAY2BGR)

    rivers = {"self": {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0}}
    result = rec.recognize_rivers(bgr, rivers)
    assert len(result) == RIVER_ROWS * RIVER_COLS
    assert all(r["player"] == 0 for r in result)
    assert all(r["tedashi"] is True for r in result)
    # 横向き判定は付かない (全て upright テンプレを並べたため)。
    assert all("riichi" not in r for r in result)
    assert [r["tile"] for r in result] == codes_flat


def test_recognize_skips_blank_cells(tmp_path: Path) -> None:
    """途中で鳴かれて空になったセルはスキップされ、残りの牌だけ返る。"""
    _write_all_templates(tmp_path)
    rec = RiverRecognizer(tmp_path)

    # 1 段目: 1m, 2m, _, 3m, _, _
    # 残り 3 段は空
    grid: list[list[str | None]] = [
        ["1m", "2m", None, "3m", None, None],
        [None] * RIVER_COLS,
        [None] * RIVER_COLS,
        [None] * RIVER_COLS,
    ]
    canvas_gray = _make_player_canvas(grid)
    bgr = cv2.cvtColor(canvas_gray, cv2.COLOR_GRAY2BGR)

    rivers = {"right": {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0}}
    result = rec.recognize_rivers(bgr, rivers)
    tiles = [r["tile"] for r in result]
    assert tiles == ["1m", "2m", "3m"]
    assert all(r["player"] == 1 for r in result)


def test_recognize_detects_riichi_horizontal_tile(tmp_path: Path) -> None:
    """90 度 CCW 回転したテンプレを 1 枚混ぜると riichi=True で識別される。"""
    _write_all_templates(tmp_path)
    rec = RiverRecognizer(tmp_path)

    riichi_code = "5m"
    riichi_gray_upright = _make_tile_template(TILE_CODES.index(riichi_code))
    riichi_gray_horizontal = cv2.rotate(riichi_gray_upright, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # 横向き牌は素のままだと (W, H) = (TMPL_W, TMPL_H) 違いになるためセルへ
    # フィットさせる: ここでは upright と同サイズへ resize して並べる。
    # river_recognizer 側は cell 形状を tmpl 形状へ resize するため、
    # アスペクト不一致でも回転テンプレ側のスコアが勝てば riichi 判定される。
    riichi_gray_horizontal = cv2.resize(riichi_gray_horizontal, (TMPL_W, TMPL_H))

    grid: list[list[str | None]] = [
        ["1m", "2m", "3m", "4m", None, None],
        [None] * RIVER_COLS,
        [None] * RIVER_COLS,
        [None] * RIVER_COLS,
    ]
    canvas_gray = _make_player_canvas(grid)
    # 1 段目 col=4 を riichi 横向き牌に差し替え
    sx = (4 * canvas_gray.shape[1]) // RIVER_COLS
    ex = (5 * canvas_gray.shape[1]) // RIVER_COLS
    sy = 0
    ey = canvas_gray.shape[0] // RIVER_ROWS
    canvas_gray[sy:ey, sx:ex] = riichi_gray_horizontal
    bgr = cv2.cvtColor(canvas_gray, cv2.COLOR_GRAY2BGR)

    rivers = {"self": {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0}}
    result = rec.recognize_rivers(bgr, rivers)
    tiles = [r["tile"] for r in result]
    assert tiles == ["1m", "2m", "3m", "4m", riichi_code]
    riichi_entries = [r for r in result if r.get("riichi")]
    assert len(riichi_entries) == 1
    assert riichi_entries[0]["tile"] == riichi_code


def test_recognize_all_four_players_have_correct_player_index(tmp_path: Path) -> None:
    """4 家分の ROI を別々に切って与えると、それぞれ player 0..3 で返る。"""
    _write_all_templates(tmp_path)
    rec = RiverRecognizer(tmp_path)

    # 1 牌だけを置いた小さい河キャンバスを 2x2 グリッドに並べる:
    # 上段左=self, 上段右=right, 下段左=across, 下段右=left
    def _one_tile_canvas(code: str) -> np.ndarray:
        grid: list[list[str | None]] = [
            [code, None, None, None, None, None],
            [None] * RIVER_COLS,
            [None] * RIVER_COLS,
            [None] * RIVER_COLS,
        ]
        return _make_player_canvas(grid)

    panels = {
        "self": _one_tile_canvas("1m"),
        "right": _one_tile_canvas("2p"),
        "across": _one_tile_canvas("3s"),
        "left": _one_tile_canvas("1z"),
    }
    ph, pw = next(iter(panels.values())).shape
    full = np.full((ph * 2, pw * 2), 128, dtype=np.uint8)
    full[0:ph, 0:pw] = panels["self"]
    full[0:ph, pw : 2 * pw] = panels["right"]
    full[ph : 2 * ph, 0:pw] = panels["across"]
    full[ph : 2 * ph, pw : 2 * pw] = panels["left"]
    bgr = cv2.cvtColor(full, cv2.COLOR_GRAY2BGR)

    rivers = {
        "self": {"x": 0.0, "y": 0.0, "w": 0.5, "h": 0.5},
        "right": {"x": 0.5, "y": 0.0, "w": 0.5, "h": 0.5},
        "across": {"x": 0.0, "y": 0.5, "w": 0.5, "h": 0.5},
        "left": {"x": 0.5, "y": 0.5, "w": 0.5, "h": 0.5},
    }
    result = rec.recognize_rivers(bgr, rivers)
    by_player = {r["player"]: r for r in result}
    assert by_player[0]["tile"] == "1m"
    assert by_player[1]["tile"] == "2p"
    assert by_player[2]["tile"] == "3s"
    assert by_player[3]["tile"] == "1z"


@pytest.mark.parametrize(
    "rect",
    [
        {"x": 0.0, "y": 0.0, "w": 0.0, "h": 1.0},
        {"x": 0.0, "y": 0.0, "w": 1.0, "h": 0.0},
    ],
)
def test_recognize_handles_degenerate_roi(tmp_path: Path, rect: dict) -> None:
    _write_all_templates(tmp_path)
    rec = RiverRecognizer(tmp_path)
    bgr = np.zeros((200, 200, 3), dtype=np.uint8)
    assert rec.recognize_rivers(bgr, {"self": rect}) == []


def test_partial_template_set_fails_closed(tmp_path: Path) -> None:
    """部分セット (9/37) では recognize_rivers が無効化される。"""
    for code in TILE_CODES[:9]:
        cv2.imwrite(
            str(tmp_path / f"{code}.png"),
            _make_tile_template(TILE_CODES.index(code)),
        )
    rec = RiverRecognizer(tmp_path)
    bgr = np.full((TMPL_H * RIVER_ROWS, TMPL_W * RIVER_COLS, 3), 128, dtype=np.uint8)
    rivers = {"self": {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0}}
    assert rec.recognize_rivers(bgr, rivers) == []
