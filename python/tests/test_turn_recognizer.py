"""TurnRecognizer の単体テスト (issue #15)。

合成テンプレ + 合成入力画像でカバレッジを取る (実画像 templates は issue #16
以降の系列で配備予定)。タイマー検出は HSV 色判定なので、決め打ちの色を持つ
ROI を作って入出力を検証する。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from recognition.tile_recognizer import RoiRect
from recognition.turn_recognizer import (
    ACTION_KEYS,
    ACTION_MATCH_THRESHOLD,
    TIMER_OCCUPANCY_THRESHOLD,
    TurnRecognizer,
)

TMPL_H = 24
TMPL_W = 64


def _make_pattern(seed: int) -> np.ndarray:
    """テンプレートとして区別可能なグレースケールノイズパターン。"""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(TMPL_H, TMPL_W), dtype=np.uint8)


def _write_all_action_templates(dst: Path) -> dict[str, np.ndarray]:
    """全 7 種のテンプレを書き出して、テスト側で照合に使えるよう dict を返す。"""
    dst.mkdir(parents=True, exist_ok=True)
    templates: dict[str, np.ndarray] = {}
    for i, key in enumerate(ACTION_KEYS):
        # シード空間は wind / tile と重ならないように 2000+ から開始。
        img = _make_pattern(2000 + i)
        cv2.imwrite(str(dst / f"{key}.png"), img)
        templates[key] = img
    return templates


# ----------------------- detect_buttons ------------------------------------


def test_detect_buttons_returns_empty_when_templates_missing(tmp_path: Path) -> None:
    rec = TurnRecognizer(tmp_path / "actions")
    bgr = np.zeros((50, 200, 3), dtype=np.uint8)
    assert rec.detect_buttons(bgr, RoiRect(0.0, 0.0, 1.0, 1.0)) == []


def test_detect_buttons_partial_set_fails_closed(tmp_path: Path) -> None:
    """4/7 種だけのテンプレでは fail-closed で全無効化される。"""
    actions = tmp_path / "actions"
    actions.mkdir()
    for i, key in enumerate(ACTION_KEYS[:4]):
        cv2.imwrite(str(actions / f"{key}.png"), _make_pattern(2000 + i))
    rec = TurnRecognizer(actions)
    bgr = np.zeros((TMPL_H + 10, TMPL_W + 50, 3), dtype=np.uint8) + 100
    assert rec.detect_buttons(bgr, RoiRect(0.0, 0.0, 1.0, 1.0)) == []


def test_detect_buttons_detects_pon_at_arbitrary_position(tmp_path: Path) -> None:
    """ROI 内の任意位置にテンプレを埋め込んでも検出できる (sliding match の妥当性)。"""
    actions = tmp_path / "actions"
    templates = _write_all_action_templates(actions)
    rec = TurnRecognizer(actions)

    # 1 段だけのキャンバスを横に長く取り、「pon」テンプレを中央付近に貼り付ける。
    # それ以外のピクセルは中間色で std を作って、テンプレと無相関にする。
    canvas_h = TMPL_H
    canvas_w = TMPL_W * 4
    canvas_gray = np.full((canvas_h, canvas_w), 128, dtype=np.uint8)
    paste_x = TMPL_W * 2  # 中央付近
    canvas_gray[:, paste_x : paste_x + TMPL_W] = templates["pon"]
    bgr = cv2.cvtColor(canvas_gray, cv2.COLOR_GRAY2BGR)

    detected = rec.detect_buttons(bgr, RoiRect(0.0, 0.0, 1.0, 1.0))
    assert "pon" in detected


def test_detect_buttons_rejects_unrelated_noise(tmp_path: Path) -> None:
    """NCC が ACTION_MATCH_THRESHOLD 未満のフレームでは何も検出しない (FP <5% 受け入れ基準)。"""
    assert ACTION_MATCH_THRESHOLD > 0.0  # 閾値ゼロ退行防止
    actions = tmp_path / "actions"
    _write_all_action_templates(actions)
    rec = TurnRecognizer(actions)

    # 全テンプレと無相関のノイズ (seed=9999) で検出されないことを確認。
    unrelated = _make_pattern(9999)
    bgr = cv2.cvtColor(unrelated, cv2.COLOR_GRAY2BGR)
    detected = rec.detect_buttons(bgr, RoiRect(0.0, 0.0, 1.0, 1.0))
    assert detected == []


def test_detect_buttons_no_roi_warns_once(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    actions = tmp_path / "actions"
    _write_all_action_templates(actions)
    rec = TurnRecognizer(actions)
    bgr = np.zeros((TMPL_H, TMPL_W, 3), dtype=np.uint8)
    with caplog.at_level("WARNING", logger="recognition"):
        for _ in range(3):
            assert rec.detect_buttons(bgr, None) == []
    msgs = [
        r.getMessage()
        for r in caplog.records
        if "action_buttons ROI not calibrated" in r.getMessage()
    ]
    assert len(msgs) == 1


# ----------------------- detect_timer_active -------------------------------


def test_detect_timer_active_returns_false_when_no_roi(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    rec = TurnRecognizer(tmp_path / "actions")
    bgr = np.zeros((50, 50, 3), dtype=np.uint8)
    with caplog.at_level("WARNING", logger="recognition"):
        for _ in range(3):
            assert rec.detect_timer_active(bgr, None) is False
    msgs = [
        r.getMessage() for r in caplog.records if "turn_timer ROI not calibrated" in r.getMessage()
    ]
    assert len(msgs) == 1


def test_detect_timer_active_true_for_yellow_patch(tmp_path: Path) -> None:
    """雀魂のタイマー色 (黄系 HSV) で塗りつぶされた ROI は active。"""
    rec = TurnRecognizer(tmp_path / "actions")
    # HSV (20, 200, 220) ≒ 鮮やかな黄色 → BGR に変換して 1 ROI に敷き詰める
    hsv_patch = np.full((50, 50, 3), (20, 200, 220), dtype=np.uint8)
    bgr = cv2.cvtColor(hsv_patch, cv2.COLOR_HSV2BGR)
    assert rec.detect_timer_active(bgr, RoiRect(0.0, 0.0, 1.0, 1.0)) is True


def test_detect_timer_active_false_for_neutral_background(tmp_path: Path) -> None:
    """色味のない背景 (グレー) はタイマー色域に入らないので非アクティブ。"""
    rec = TurnRecognizer(tmp_path / "actions")
    # 完全な無彩色グレー → HSV (?, 0, V)。S=0 なので inRange (S>=100) に入らない。
    bgr = np.full((50, 50, 3), 128, dtype=np.uint8)
    assert rec.detect_timer_active(bgr, RoiRect(0.0, 0.0, 1.0, 1.0)) is False


def test_detect_timer_active_threshold_boundary(tmp_path: Path) -> None:
    """占有率がしきい値未満 (= タイマーがほぼ消えている) なら False。"""
    rec = TurnRecognizer(tmp_path / "actions")
    # 1% だけ黄色 → TIMER_OCCUPANCY_THRESHOLD (5%) 未満
    bgr = np.full((100, 100, 3), 128, dtype=np.uint8)
    hsv_yellow = cv2.cvtColor(
        np.full((10, 10, 3), (20, 200, 220), dtype=np.uint8),
        cv2.COLOR_HSV2BGR,
    )
    bgr[0:10, 0:10] = hsv_yellow
    occupancy = 100 / (100 * 100)
    assert occupancy < TIMER_OCCUPANCY_THRESHOLD
    assert rec.detect_timer_active(bgr, RoiRect(0.0, 0.0, 1.0, 1.0)) is False
