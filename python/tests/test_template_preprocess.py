"""crop_roi_to_gray / fit_to_template_size の単体テスト (issue #16)。

`tile_recognizer` / `river_recognizer` / `wind_recognizer` の重複していた
ROI クランプ + grayscale + resize を共通化したヘルパーの挙動を直接検証する。
"""

from __future__ import annotations

import numpy as np

from recognition.tile_recognizer import (
    RoiRect,
    crop_roi_to_gray,
    fit_to_template_size,
)


def _make_bgr(h: int, w: int) -> np.ndarray:
    """値が座標で一意に決まる BGR フレーム (テスト検証用)。"""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 0] = 50
    img[:, :, 1] = 100
    img[:, :, 2] = 200
    return img


def test_crop_roi_to_gray_returns_grayscale_2d() -> None:
    frame = _make_bgr(200, 300)
    roi = RoiRect(x=0.1, y=0.2, w=0.5, h=0.4)
    out = crop_roi_to_gray(frame, roi)
    assert out is not None
    assert out.ndim == 2
    # 期待サイズ: (200*0.4, 300*0.5) = (80, 150)
    assert out.shape == (80, 150)
    assert out.dtype == np.uint8


def test_crop_roi_to_gray_clamps_negative_and_over_one() -> None:
    """ROI が画面外に飛び出していても画素座標は 0..frame_size にクランプされる。"""
    frame = _make_bgr(100, 200)
    # x が負・幅が画面端を超える ROI
    roi = RoiRect(x=-0.1, y=-0.5, w=2.0, h=2.0)
    out = crop_roi_to_gray(frame, roi)
    assert out is not None
    # クランプ結果: (0..100, 0..200) フル領域
    assert out.shape == (100, 200)


def test_crop_roi_to_gray_returns_none_for_degenerate_roi() -> None:
    frame = _make_bgr(100, 200)
    # 幅が 0 になる ROI
    assert crop_roi_to_gray(frame, RoiRect(x=0.5, y=0.5, w=0.0, h=0.1)) is None
    # 高さが 0 になる ROI
    assert crop_roi_to_gray(frame, RoiRect(x=0.5, y=0.5, w=0.1, h=0.0)) is None


def test_crop_roi_to_gray_respects_min_size() -> None:
    """`min_w` / `min_h` を下回る ROI は None を返す。"""
    frame = _make_bgr(100, 200)
    # 幅は 200*0.05 = 10 ピクセル
    roi = RoiRect(x=0.0, y=0.0, w=0.05, h=0.5)
    assert crop_roi_to_gray(frame, roi, min_w=10) is not None
    assert crop_roi_to_gray(frame, roi, min_w=11) is None


def test_crop_roi_to_gray_none_for_empty_frame() -> None:
    roi = RoiRect(x=0.0, y=0.0, w=1.0, h=1.0)
    assert crop_roi_to_gray(None, roi) is None
    empty = np.zeros((0, 0, 3), dtype=np.uint8)
    assert crop_roi_to_gray(empty, roi) is None


def test_crop_roi_to_gray_passes_through_grayscale_input() -> None:
    """既にグレースケール (2D) な入力は cvtColor を通さずそのまま切り出される。"""
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 256, size=(100, 200), dtype=np.uint8)
    roi = RoiRect(x=0.0, y=0.0, w=0.5, h=0.5)
    out = crop_roi_to_gray(frame, roi)
    assert out is not None
    assert out.ndim == 2
    assert out.shape == (50, 100)
    assert np.array_equal(out, frame[:50, :100])


def test_fit_to_template_size_noop_when_matching() -> None:
    seg = np.zeros((32, 24), dtype=np.uint8)
    out = fit_to_template_size(seg, (32, 24))
    assert out is seg  # no-op = 同一オブジェクトを返す


def test_fit_to_template_size_resizes_when_mismatched() -> None:
    seg = np.zeros((64, 48), dtype=np.uint8)
    out = fit_to_template_size(seg, (32, 24))
    assert out.shape == (32, 24)
    assert out.dtype == np.uint8
