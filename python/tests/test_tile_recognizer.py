"""tile_recognizer の単体テスト (issue #11)。

テンプレ画像 (issue #16) は本物がまだ無いので、テスト内で合成テンプレを
一時ディレクトリに書き出して挙動を検証する。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from recognition.tile_recognizer import (
    BLANK_STD_THRESHOLD,
    HAND_SLOTS,
    TILE_CODES,
    RoiRect,
    TileRecognizer,
)

TMPL_H = 32
TMPL_W = 24


def _make_tile_template(seed: int) -> np.ndarray:
    """牌コードごとに異なるパターンを生成 (matchTemplate で区別できる程度に違う)。"""
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 256, size=(TMPL_H, TMPL_W), dtype=np.uint8)
    return img


def _write_all_templates(dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for i, code in enumerate(TILE_CODES):
        cv2.imwrite(str(dst / f"{code}.png"), _make_tile_template(i))


def test_tile_codes_are_37_unique() -> None:
    assert len(TILE_CODES) == 37
    assert len(set(TILE_CODES)) == 37
    # Mortal/天鳳慣例: 赤 5 は 0m / 0p / 0s
    assert {"0m", "0p", "0s"}.issubset(set(TILE_CODES))


def test_roi_rect_from_dict_handles_bad_input() -> None:
    assert RoiRect.from_dict(None) is None
    assert RoiRect.from_dict({}) is None
    assert RoiRect.from_dict({"x": 0.1}) is None
    assert RoiRect.from_dict("not a dict") is None  # type: ignore[arg-type]
    r = RoiRect.from_dict({"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4})
    assert r is not None
    assert (r.x, r.y, r.w, r.h) == (0.1, 0.2, 0.3, 0.4)


def test_recognize_returns_empty_when_templates_missing(tmp_path: Path) -> None:
    rec = TileRecognizer(tmp_path)
    bgr = np.zeros((100, 100, 3), dtype=np.uint8)
    tiles, conf = rec.recognize_hand(bgr, RoiRect(0.0, 0.0, 1.0, 1.0))
    assert tiles == []
    assert conf == 0.0


def test_recognize_returns_empty_when_roi_none(tmp_path: Path) -> None:
    _write_all_templates(tmp_path)
    rec = TileRecognizer(tmp_path)
    bgr = np.zeros((100, 100, 3), dtype=np.uint8)
    tiles, conf = rec.recognize_hand(bgr, None)
    assert tiles == []
    assert conf == 0.0


def test_no_roi_warning_logged_only_once(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    _write_all_templates(tmp_path)
    rec = TileRecognizer(tmp_path)
    bgr = np.zeros((100, 100, 3), dtype=np.uint8)
    with caplog.at_level("WARNING", logger="recognition"):
        for _ in range(5):
            rec.recognize_hand(bgr, None)
    no_roi_warnings = [r for r in caplog.records if "ROI not calibrated" in r.getMessage()]
    assert len(no_roi_warnings) == 1


def test_partial_template_set_fails_closed(tmp_path: Path) -> None:
    """Codex P2 on PR #43: 部分セットで誤認識を防ぐため、37 種揃わなければ無効化。"""
    # 1m から 9m まで (9 枚) だけ書き出す → 28/37 不足
    for code in TILE_CODES[:9]:
        cv2.imwrite(str(tmp_path / f"{code}.png"), _make_tile_template(TILE_CODES.index(code)))
    rec = TileRecognizer(tmp_path)
    # 一見テンプレが「ある」状態でも、未ロード扱いで empty を返す
    bgr = np.zeros((TMPL_H, TMPL_W * HAND_SLOTS, 3), dtype=np.uint8) + 100
    bgr[5:25, 5:200] = np.random.default_rng(0).integers(0, 256, (20, 195, 3), dtype=np.uint8)
    tiles, conf = rec.recognize_hand(bgr, RoiRect(0.0, 0.0, 1.0, 1.0))
    assert tiles == []
    assert conf == 0.0


def test_recognize_returns_empty_when_all_segments_blank(tmp_path: Path) -> None:
    _write_all_templates(tmp_path)
    rec = TileRecognizer(tmp_path)
    # 完全にフラットな画像 → std=0 で全セグが空白判定される
    bgr = np.full((TMPL_H, TMPL_W * HAND_SLOTS, 3), 128, dtype=np.uint8)
    tiles, conf = rec.recognize_hand(bgr, RoiRect(0.0, 0.0, 1.0, 1.0))
    assert tiles == []
    assert conf == 0.0


def test_recognize_matches_known_tiles(tmp_path: Path) -> None:
    """合成手牌画像 (テンプレを横に並べたもの) を入れて、各セグメントが
    元のテンプレ牌コードに正しくマッチすることを確認。"""
    _write_all_templates(tmp_path)
    rec = TileRecognizer(tmp_path)

    # 14 種の異なる牌をテンプレからピックアップして横並び
    expected = TILE_CODES[:HAND_SLOTS]
    grays = [_make_tile_template(TILE_CODES.index(c)) for c in expected]
    canvas_gray = np.concatenate(grays, axis=1)
    bgr = cv2.cvtColor(canvas_gray, cv2.COLOR_GRAY2BGR)

    tiles, conf = rec.recognize_hand(bgr, RoiRect(0.0, 0.0, 1.0, 1.0))
    assert tiles == expected
    # ランダムノイズが完全一致するはずなので NCC は 1.0 近辺。
    assert conf > 0.99


def test_recognize_skips_blank_slot_in_middle(tmp_path: Path) -> None:
    """13 牌 + 1 ブランクの並びで、ブランクを除いた 13 牌が返る。"""
    _write_all_templates(tmp_path)
    rec = TileRecognizer(tmp_path)

    expected = TILE_CODES[: HAND_SLOTS - 1]
    grays: list[np.ndarray] = [_make_tile_template(TILE_CODES.index(c)) for c in expected]
    # 14 番目スロットを std がしきい値未満になる平坦画像にする
    blank = np.full((TMPL_H, TMPL_W), 200, dtype=np.uint8)
    grays.append(blank)
    assert float(blank.std()) < BLANK_STD_THRESHOLD
    canvas_gray = np.concatenate(grays, axis=1)
    bgr = cv2.cvtColor(canvas_gray, cv2.COLOR_GRAY2BGR)

    tiles, conf = rec.recognize_hand(bgr, RoiRect(0.0, 0.0, 1.0, 1.0))
    assert tiles == expected
    assert conf > 0.99


def test_recognize_uses_roi_subregion(tmp_path: Path) -> None:
    """ROI 外の領域は無視される。"""
    _write_all_templates(tmp_path)
    rec = TileRecognizer(tmp_path)

    expected = TILE_CODES[:HAND_SLOTS]
    grays = [_make_tile_template(TILE_CODES.index(c)) for c in expected]
    hand_strip = np.concatenate(grays, axis=1)
    hand_bgr = cv2.cvtColor(hand_strip, cv2.COLOR_GRAY2BGR)
    h, w = hand_bgr.shape[:2]

    # 大きいキャンバスの下半分・中央寄りに hand_strip を貼る
    full_h, full_w = h * 4, w * 2
    canvas = np.full((full_h, full_w, 3), 255, dtype=np.uint8)
    y0 = full_h // 2
    x0 = (full_w - w) // 2
    canvas[y0 : y0 + h, x0 : x0 + w] = hand_bgr

    roi = RoiRect(x=x0 / full_w, y=y0 / full_h, w=w / full_w, h=h / full_h)
    tiles, conf = rec.recognize_hand(canvas, roi)
    assert tiles == expected
    assert conf > 0.99


@pytest.mark.parametrize("rect", [RoiRect(0.0, 0.0, 0.0, 1.0), RoiRect(0.0, 0.0, 1.0, 0.0)])
def test_recognize_handles_degenerate_roi(tmp_path: Path, rect: RoiRect) -> None:
    _write_all_templates(tmp_path)
    rec = TileRecognizer(tmp_path)
    bgr = np.zeros((100, 100, 3), dtype=np.uint8)
    tiles, conf = rec.recognize_hand(bgr, rect)
    assert tiles == []
    assert conf == 0.0
