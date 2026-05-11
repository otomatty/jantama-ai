"""board_recognizer / wind_recognizer / ocr_recognizer の単体テスト (issue #12)。

テンプレ実画像はまだ無いので、tile_recognizer のテストと同じく合成画像で
カバレッジを確保する。Tesseract は monkeypatch で差し替えてバイナリ依存を回避。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from recognition import ocr_recognizer
from recognition.board_recognizer import DEFAULT_TENHOU_JSON, BoardRecognizer
from recognition.tile_recognizer import (
    BLANK_STD_THRESHOLD,
    DORA_SLOTS,
    HAND_SLOTS,
    TILE_CODES,
    RoiRect,
    TileRecognizer,
)
from recognition.wind_recognizer import WIND_MATCH_THRESHOLD, WindRecognizer

TMPL_H = 32
TMPL_W = 24


def _make_pattern(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(TMPL_H, TMPL_W), dtype=np.uint8)


def _write_all_tile_templates(dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for i, code in enumerate(TILE_CODES):
        cv2.imwrite(str(dst / f"{code}.png"), _make_pattern(i))


def _write_all_wind_templates(dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    # tile テンプレと違うシード空間で書く (= マッチ済みの牌テンプレと混同しない)。
    for i, key in enumerate(["east", "south", "west", "north"]):
        cv2.imwrite(str(dst / f"{key}.png"), _make_pattern(1000 + i))


def _reset_pytesseract_ref() -> None:
    """ocr_recognizer のグローバル `_PYTESSERACT` を各テスト前にリセット。"""
    ocr_recognizer._PYTESSERACT.module = None
    ocr_recognizer._PYTESSERACT.tried = False
    ocr_recognizer._PYTESSERACT.error = None


@pytest.fixture(autouse=True)
def _autoreset_pytesseract() -> None:
    _reset_pytesseract_ref()


# ----------------------- recognize_dora ------------------------------------


def test_recognize_dora_returns_empty_when_templates_missing(tmp_path: Path) -> None:
    rec = TileRecognizer(tmp_path)
    bgr = np.zeros((100, 100, 3), dtype=np.uint8)
    tiles, conf = rec.recognize_dora(bgr, RoiRect(0.0, 0.0, 1.0, 1.0))
    assert tiles == []
    assert conf == 0.0


def test_recognize_dora_matches_first_n_tiles(tmp_path: Path) -> None:
    """合成ドラ ROI 画像 (= 連結テンプレ) を入れて先頭から N 枚マッチすることを確認。"""
    _write_all_tile_templates(tmp_path)
    rec = TileRecognizer(tmp_path)

    expected = TILE_CODES[:DORA_SLOTS]
    grays = [_make_pattern(TILE_CODES.index(c)) for c in expected]
    canvas_gray = np.concatenate(grays, axis=1)
    bgr = cv2.cvtColor(canvas_gray, cv2.COLOR_GRAY2BGR)

    tiles, conf = rec.recognize_dora(bgr, RoiRect(0.0, 0.0, 1.0, 1.0))
    assert tiles == expected
    assert conf > 0.99


def test_recognize_dora_skips_trailing_blank_slots(tmp_path: Path) -> None:
    """3 枚 + 2 ブランクの並びで、ブランクを除いた 3 枚だけ返る。"""
    _write_all_tile_templates(tmp_path)
    rec = TileRecognizer(tmp_path)

    expected = TILE_CODES[:3]
    grays = [_make_pattern(TILE_CODES.index(c)) for c in expected]
    for _ in range(DORA_SLOTS - 3):
        blank = np.full((TMPL_H, TMPL_W), 200, dtype=np.uint8)
        assert float(blank.std()) < BLANK_STD_THRESHOLD
        grays.append(blank)
    canvas_gray = np.concatenate(grays, axis=1)
    bgr = cv2.cvtColor(canvas_gray, cv2.COLOR_GRAY2BGR)

    tiles, conf = rec.recognize_dora(bgr, RoiRect(0.0, 0.0, 1.0, 1.0))
    assert tiles == expected
    assert conf > 0.99


def test_recognize_dora_no_roi_returns_empty(tmp_path: Path) -> None:
    _write_all_tile_templates(tmp_path)
    rec = TileRecognizer(tmp_path)
    bgr = np.zeros((100, 100, 3), dtype=np.uint8)
    tiles, conf = rec.recognize_dora(bgr, None)
    assert tiles == []
    assert conf == 0.0


def test_hand_and_dora_no_roi_warnings_independent(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """hand / dora で「ROI 未キャリブ」警告がそれぞれ 1 回ずつ出る (重複しない)。"""
    _write_all_tile_templates(tmp_path)
    rec = TileRecognizer(tmp_path)
    bgr = np.zeros((100, 100, 3), dtype=np.uint8)
    with caplog.at_level("WARNING", logger="recognition"):
        for _ in range(3):
            rec.recognize_hand(bgr, None)
        for _ in range(3):
            rec.recognize_dora(bgr, None)
    msgs = [r.getMessage() for r in caplog.records if "ROI not calibrated" in r.getMessage()]
    # hand 用 1 件 + dora 用 1 件 = 2 件
    assert sum("hand" in m for m in msgs) == 1
    assert sum("dora" in m for m in msgs) == 1


# ----------------------- WindRecognizer ------------------------------------


def test_wind_recognizer_returns_none_when_templates_missing(tmp_path: Path) -> None:
    rec = WindRecognizer(tmp_path / "winds")
    bgr = np.zeros((100, 100, 3), dtype=np.uint8)
    label, conf = rec.recognize(bgr, RoiRect(0.0, 0.0, 1.0, 1.0))
    assert label is None
    assert conf == 0.0


def test_wind_recognizer_partial_set_fails_closed(tmp_path: Path) -> None:
    winds = tmp_path / "winds"
    winds.mkdir()
    # 2 枚だけ → fail-closed で disabled
    cv2.imwrite(str(winds / "east.png"), _make_pattern(1000))
    cv2.imwrite(str(winds / "south.png"), _make_pattern(1001))
    rec = WindRecognizer(winds)
    bgr = np.zeros((TMPL_H, TMPL_W, 3), dtype=np.uint8) + 100
    bgr[5:25, 5:20] = np.random.default_rng(0).integers(0, 256, (20, 15, 3), dtype=np.uint8)
    label, conf = rec.recognize(bgr, RoiRect(0.0, 0.0, 1.0, 1.0))
    assert label is None
    assert conf == 0.0


def test_wind_recognizer_matches_known_label(tmp_path: Path) -> None:
    winds = tmp_path / "winds"
    _write_all_wind_templates(winds)
    rec = WindRecognizer(winds)

    # 「south.png」のテンプレと完全一致する画像を投げ、「南」が返ることを確認。
    south_gray = _make_pattern(1001)
    bgr = cv2.cvtColor(south_gray, cv2.COLOR_GRAY2BGR)
    label, conf = rec.recognize(bgr, RoiRect(0.0, 0.0, 1.0, 1.0))
    assert label == "南"
    assert conf > 0.99


def test_wind_recognizer_rejects_low_confidence_match(tmp_path: Path) -> None:
    """Codex P1 on PR #44: NCC < WIND_MATCH_THRESHOLD なら None を返す。

    全テンプレと無相関な画像 (= 全テンプレで違うシードのノイズ) を投げて、
    どのテンプレとも 0.5+ で一致せず最良スコアでも閾値未満になることを確認。
    """
    assert WIND_MATCH_THRESHOLD > 0.0  # 「閾値ゼロ」の退行を防ぐ
    winds = tmp_path / "winds"
    _write_all_wind_templates(winds)
    rec = WindRecognizer(winds)

    # _make_pattern(1000..1003) がテンプレ。9999 は全テンプレと無相関のはず。
    unrelated_gray = _make_pattern(9999)
    bgr = cv2.cvtColor(unrelated_gray, cv2.COLOR_GRAY2BGR)
    label, conf = rec.recognize(bgr, RoiRect(0.0, 0.0, 1.0, 1.0))
    assert label is None, f"low-confidence match (conf={conf}) should return None"
    # 失敗扱いだが、スコアは返している (デバッグ用途)。
    assert conf < WIND_MATCH_THRESHOLD


def test_wind_recognizer_no_roi_warns_once(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    winds = tmp_path / "winds"
    _write_all_wind_templates(winds)
    rec = WindRecognizer(winds)
    bgr = np.zeros((100, 100, 3), dtype=np.uint8)
    with caplog.at_level("WARNING", logger="recognition"):
        for _ in range(5):
            rec.recognize(bgr, None)
    warnings = [r for r in caplog.records if "self_wind ROI not calibrated" in r.getMessage()]
    assert len(warnings) == 1


# ----------------------- ocr_recognizer (with mocked Tesseract) -------------


class _FakePytesseract:
    """`pytesseract.image_to_string` を返却値テーブルで差し替えるためのスタブ。"""

    def __init__(self, returns: dict[str, str] | str | Exception) -> None:
        self._returns = returns
        self.calls: list[tuple[Any, str, str]] = []

    def image_to_string(
        self, img: np.ndarray, lang: str = "eng", config: str = "", **kwargs: Any
    ) -> str:  # noqa: ARG002
        self.calls.append((img.shape, lang, config))
        if isinstance(self._returns, Exception):
            raise self._returns
        if isinstance(self._returns, dict):
            # lang 別に返却 (round_label=jpn, scores/turn=eng)
            return self._returns.get(lang, "")
        return self._returns


def _install_fake_pytesseract(fake: object) -> None:
    """`_get_pytesseract` がインポート試行する前に `_PYTESSERACT` を埋める。

    `fake` は `image_to_string` を持つ任意のオブジェクト (`_FakePytesseract` や
    stateful `_Side` クラス等)。`error = None` のリセットも込みで、各テストが
    helper 経由で統一的に扱えるようにする (CodeRabbit nit on PR #44)。
    """
    ocr_recognizer._PYTESSERACT.module = fake
    ocr_recognizer._PYTESSERACT.tried = True
    ocr_recognizer._PYTESSERACT.error = None


def test_recognize_round_label_parses_japanese_output(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakePytesseract({"jpn": "東1局\n"})
    _install_fake_pytesseract(fake)
    bgr = np.full((40, 80, 3), 128, dtype=np.uint8)
    bgr[5:35, 5:75] = 255  # OTSU の閾値計算が回るよう適度なバラつき
    label = ocr_recognizer.recognize_round_label(bgr, RoiRect(0.0, 0.0, 1.0, 1.0))
    assert label == "東1局"


def test_recognize_round_label_handles_missing_kyoku(monkeypatch: pytest.MonkeyPatch) -> None:
    """OCR が「局」を読み損ねても、「南3」だけで局名を組み立てる。"""
    fake = _FakePytesseract({"jpn": "南3"})
    _install_fake_pytesseract(fake)
    bgr = np.full((40, 80, 3), 128, dtype=np.uint8)
    bgr[5:35, 5:75] = 255
    label = ocr_recognizer.recognize_round_label(bgr, RoiRect(0.0, 0.0, 1.0, 1.0))
    assert label == "南3局"


def test_recognize_round_label_returns_none_on_garbage(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakePytesseract({"jpn": "????"})
    _install_fake_pytesseract(fake)
    bgr = np.full((40, 80, 3), 128, dtype=np.uint8)
    bgr[5:35, 5:75] = 255
    label = ocr_recognizer.recognize_round_label(bgr, RoiRect(0.0, 0.0, 1.0, 1.0))
    assert label is None


def test_recognize_scores_returns_all_four() -> None:
    # OCR は呼ばれる順に決めたい (4 セグで違う値) → side_effect 風に。
    values = iter(["25000", "30000", "20000", "25000"])

    class _Side:
        def __init__(self) -> None:
            self.calls = 0

        def image_to_string(
            self, img: np.ndarray, lang: str = "eng", config: str = "", **kwargs: Any
        ) -> str:  # noqa: ARG002
            self.calls += 1
            return next(values)

    _install_fake_pytesseract(_Side())

    bgr = np.full((40, 160, 3), 128, dtype=np.uint8)
    bgr[5:35, 5:155] = 255
    scores = ocr_recognizer.recognize_scores(bgr, RoiRect(0.0, 0.0, 1.0, 1.0))
    assert scores == [25000, 30000, 20000, 25000]


def test_recognize_scores_partial_failure_returns_none() -> None:
    """4 セグのうち 1 つでも読み取れなければ all-or-nothing で `None`。"""
    values = iter(["25000", "", "20000", "25000"])

    class _Side:
        def image_to_string(
            self, img: np.ndarray, lang: str = "eng", config: str = "", **kwargs: Any
        ) -> str:  # noqa: ARG002
            return next(values)

    _install_fake_pytesseract(_Side())
    bgr = np.full((40, 160, 3), 128, dtype=np.uint8)
    bgr[5:35, 5:155] = 255
    scores = ocr_recognizer.recognize_scores(bgr, RoiRect(0.0, 0.0, 1.0, 1.0))
    assert scores is None


def test_recognize_scores_negative_value() -> None:
    """飛び (マイナス点) も読み取れる。"""
    values = iter(["50000", "-5000", "30000", "25000"])

    class _Side:
        def image_to_string(
            self, img: np.ndarray, lang: str = "eng", config: str = "", **kwargs: Any
        ) -> str:  # noqa: ARG002
            return next(values)

    _install_fake_pytesseract(_Side())
    bgr = np.full((40, 160, 3), 128, dtype=np.uint8)
    bgr[5:35, 5:155] = 255
    scores = ocr_recognizer.recognize_scores(bgr, RoiRect(0.0, 0.0, 1.0, 1.0))
    assert scores == [50000, -5000, 30000, 25000]


def test_recognize_scores_out_of_range_rejected() -> None:
    """Codex P1 on PR #44: 1 桁多く読まれた (25000 → 250000) ケースを fail-closed。

    各家 100000 超は ありえない (上限) ため、合計整合性に達する前のレンジ
    チェックで `None` に倒れる。
    """
    values = iter(["250000", "25000", "25000", "25000"])

    class _Side:
        def image_to_string(
            self, img: np.ndarray, lang: str = "eng", config: str = "", **kwargs: Any
        ) -> str:  # noqa: ARG002
            return next(values)

    _install_fake_pytesseract(_Side())
    bgr = np.full((40, 160, 3), 128, dtype=np.uint8)
    bgr[5:35, 5:155] = 255
    assert ocr_recognizer.recognize_scores(bgr, RoiRect(0.0, 0.0, 1.0, 1.0)) is None


def test_recognize_scores_total_mismatch_rejected() -> None:
    """各家は範囲内でも合計が 100000 から大きくずれたら all-or-nothing で `None`。"""
    # 25000 + 25000 + 25000 + 10000 = 85000 → 100000 から 15000 ずれ (tolerance 5000)
    values = iter(["25000", "25000", "25000", "10000"])

    class _Side:
        def image_to_string(
            self, img: np.ndarray, lang: str = "eng", config: str = "", **kwargs: Any
        ) -> str:  # noqa: ARG002
            return next(values)

    _install_fake_pytesseract(_Side())
    bgr = np.full((40, 160, 3), 128, dtype=np.uint8)
    bgr[5:35, 5:155] = 255
    assert ocr_recognizer.recognize_scores(bgr, RoiRect(0.0, 0.0, 1.0, 1.0)) is None


def test_recognize_scores_within_tolerance_accepted() -> None:
    """供託リーチ棒 3 本ぶん (-3000) の合計ずれは tolerance 内で採用される。"""
    # 25000 + 25000 + 25000 + 22000 = 97000 → 100000 から 3000 ずれ (tolerance 5000)
    values = iter(["25000", "25000", "25000", "22000"])

    class _Side:
        def image_to_string(
            self, img: np.ndarray, lang: str = "eng", config: str = "", **kwargs: Any
        ) -> str:  # noqa: ARG002
            return next(values)

    _install_fake_pytesseract(_Side())
    bgr = np.full((40, 160, 3), 128, dtype=np.uint8)
    bgr[5:35, 5:155] = 255
    assert ocr_recognizer.recognize_scores(bgr, RoiRect(0.0, 0.0, 1.0, 1.0)) == [
        25000,
        25000,
        25000,
        22000,
    ]


def test_ocr_string_passes_timeout_to_pytesseract() -> None:
    """CodeRabbit Major on PR #44: image_to_string にハング保護 timeout を渡している。"""
    captured: dict[str, Any] = {}

    class _CapturingFake:
        def image_to_string(
            self, img: np.ndarray, lang: str = "eng", config: str = "", **kwargs: Any
        ) -> str:  # noqa: ARG002
            captured.update(kwargs)
            return "東1局"

    _install_fake_pytesseract(_CapturingFake())
    bgr = np.full((40, 80, 3), 128, dtype=np.uint8)
    bgr[5:35, 5:75] = 255
    ocr_recognizer.recognize_round_label(bgr, RoiRect(0.0, 0.0, 1.0, 1.0))
    assert "timeout" in captured, "timeout kwarg must be forwarded to pytesseract"
    assert isinstance(captured["timeout"], (int, float))
    assert captured["timeout"] > 0


def test_recognize_turn_in_range(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakePytesseract({"eng": "7"})
    _install_fake_pytesseract(fake)
    bgr = np.full((40, 40, 3), 128, dtype=np.uint8)
    bgr[5:35, 5:35] = 255
    assert ocr_recognizer.recognize_turn(bgr, RoiRect(0.0, 0.0, 1.0, 1.0)) == 7


def test_recognize_turn_out_of_range_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakePytesseract({"eng": "99"})
    _install_fake_pytesseract(fake)
    bgr = np.full((40, 40, 3), 128, dtype=np.uint8)
    bgr[5:35, 5:35] = 255
    assert ocr_recognizer.recognize_turn(bgr, RoiRect(0.0, 0.0, 1.0, 1.0)) is None


@pytest.mark.parametrize("h", [10, 15, 20, 24, 31])
def test_preprocess_for_ocr_upscales_low_height_crops_to_target(h: int) -> None:
    """Codex P2 on PR #44: 整数倍スケーリングだと h=20 や h=24 が拡大されないバグの退行防止。

    32 // h の floor 演算では h ∈ [17, 31] でスケール 1 となり拡大が走らなかった。
    現在は直接ターゲット高さに合わせる実装なので、h < TARGET なら必ず TARGET 以上の
    高さに引き上がる。
    """
    crop = np.full((h, 40, 3), 128, dtype=np.uint8)
    crop[1:-1, 1:-1] = 255
    out = ocr_recognizer._preprocess_for_ocr(crop)
    assert out.shape[0] >= ocr_recognizer._OCR_TARGET_HEIGHT, (
        f"h={h} was not upscaled to >= {ocr_recognizer._OCR_TARGET_HEIGHT} px "
        f"(got {out.shape[0]})"
    )


def test_round_label_to_wind() -> None:
    assert ocr_recognizer.round_label_to_wind("東1局") == "東"
    assert ocr_recognizer.round_label_to_wind("南3局") == "南"
    assert ocr_recognizer.round_label_to_wind("") is None
    assert ocr_recognizer.round_label_to_wind(None) is None
    assert ocr_recognizer.round_label_to_wind("X1局") is None


def test_ocr_short_circuits_after_tesseract_not_found(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """CodeRabbit Major on PR #44: TesseractNotFoundError 確定後は image_to_string を再呼びしない。

    監視ループで毎フレーム 3 OCR 関数が例外コストを払うのを避けるための短絡。
    """

    class _TesseractNotFoundError(Exception):
        """pytesseract.TesseractNotFoundError と同じクラス名のスタブ例外。"""

    class _NotFoundFake:
        def __init__(self) -> None:
            self.calls = 0

        def image_to_string(
            self, img: np.ndarray, lang: str = "eng", config: str = "", **kwargs: Any
        ) -> str:  # noqa: ARG002
            self.calls += 1
            raise _TesseractNotFoundError("simulated")

    _TesseractNotFoundError.__name__ = "TesseractNotFoundError"

    fake = _NotFoundFake()
    _install_fake_pytesseract(fake)

    bgr = np.full((40, 40, 3), 128, dtype=np.uint8)
    bgr[5:35, 5:35] = 255

    with caplog.at_level("WARNING", logger="recognition"):
        # 1 回目: image_to_string が呼ばれ TesseractNotFoundError → エラー状態が記録される
        assert ocr_recognizer.recognize_turn(bgr, RoiRect(0.0, 0.0, 1.0, 1.0)) is None
        first_call_count = fake.calls
        assert first_call_count == 1
        # 2..N 回目: 上の早期 return が効いて image_to_string は呼ばれない
        for _ in range(5):
            assert ocr_recognizer.recognize_turn(bgr, RoiRect(0.0, 0.0, 1.0, 1.0)) is None
            assert ocr_recognizer.recognize_round_label(bgr, RoiRect(0.0, 0.0, 1.0, 1.0)) is None
        assert fake.calls == first_call_count, "image_to_string must not be re-invoked"

    # 警告は初回 1 回のみ。
    warns = [r for r in caplog.records if "Tesseract binary not found" in r.getMessage()]
    assert len(warns) == 1


def test_ocr_disabled_when_pytesseract_missing(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """`import pytesseract` が ImportError を吐く環境で警告 1 回 + None 返却。

    pytesseract が CI 上にインストール済みかどうかに依存しないよう、
    sys.modules キャッシュを抜いた上で builtins.__import__ を差し替える。
    """
    import builtins
    import sys

    monkeypatch.delitem(sys.modules, "pytesseract", raising=False)
    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "pytesseract":
            raise ImportError("simulated missing pytesseract")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    bgr = np.full((40, 40, 3), 128, dtype=np.uint8)
    bgr[5:35, 5:35] = 255
    with caplog.at_level("WARNING", logger="recognition"):
        for _ in range(3):
            assert ocr_recognizer.recognize_turn(bgr, RoiRect(0.0, 0.0, 1.0, 1.0)) is None
    warns = [r for r in caplog.records if "pytesseract not importable" in r.getMessage()]
    assert len(warns) == 1


# ----------------------- BoardRecognizer (end-to-end-ish) -------------------


def test_board_recognizer_returns_defaults_when_no_templates(tmp_path: Path) -> None:
    """テンプレ 0 件 + OCR off で BoardRecognizer は既定値の tenhou_json を返す。"""
    rec = BoardRecognizer(tmp_path)
    bgr = np.zeros((100, 100, 3), dtype=np.uint8)
    tenhou, conf = rec.recognize(bgr, {})
    # build_board_summary が必須とするフィールドが全部 type 一致で入っていること。
    assert isinstance(tenhou["hand"], list)
    assert tenhou["self_wind"] in {"東", "南", "西", "北"}
    assert tenhou["round_wind"] in {"東", "南", "西", "北"}
    assert isinstance(tenhou["turn"], int)
    assert isinstance(tenhou["dora_indicators"], list)
    assert isinstance(tenhou["scores"], list) and len(tenhou["scores"]) == 4
    assert conf == 0.0
    # DEFAULT_TENHOU_JSON を破壊的に変更していないこと
    assert DEFAULT_TENHOU_JSON["self_wind"] == "東"


def test_board_recognizer_handles_empty_frame(tmp_path: Path) -> None:
    rec = BoardRecognizer(tmp_path)
    tenhou, conf = rec.recognize(np.empty((0, 0, 3), dtype=np.uint8), {})
    assert tenhou == DEFAULT_TENHOU_JSON
    assert conf == 0.0


def test_board_recognizer_fills_hand_when_templates_present(tmp_path: Path) -> None:
    _write_all_tile_templates(tmp_path)
    rec = BoardRecognizer(tmp_path)

    # 14 等分するハンド画像を準備。
    grays = [_make_pattern(TILE_CODES.index(c)) for c in TILE_CODES[:HAND_SLOTS]]
    canvas_gray = np.concatenate(grays, axis=1)
    bgr = cv2.cvtColor(canvas_gray, cv2.COLOR_GRAY2BGR)

    tenhou, conf = rec.recognize(bgr, {"hand": {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0}})
    assert tenhou["hand"] == TILE_CODES[:HAND_SLOTS]
    assert conf > 0.99


def test_board_recognizer_fills_round_label_when_ocr_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _FakePytesseract({"jpn": "南2局", "eng": ""})
    _install_fake_pytesseract(fake)
    rec = BoardRecognizer(tmp_path)
    bgr = np.full((80, 200, 3), 128, dtype=np.uint8)
    bgr[5:75, 5:195] = 255

    tenhou, _ = rec.recognize(bgr, {"round_info": {"x": 0.0, "y": 0.0, "w": 0.5, "h": 1.0}})
    assert tenhou["round_label"] == "南2局"
    assert tenhou["round_wind"] == "南"


def test_board_recognizer_field_failures_are_isolated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """1 つのフィールド (例: scores) が失敗しても他フィールドは埋まる。"""
    _write_all_tile_templates(tmp_path)
    # OCR は round_label だけ成功し、scores/turn は失敗 (空文字)
    values: dict[str, str] = {"jpn": "東4局", "eng": ""}
    fake = _FakePytesseract(values)
    _install_fake_pytesseract(fake)

    rec = BoardRecognizer(tmp_path)
    grays = [_make_pattern(TILE_CODES.index(c)) for c in TILE_CODES[:HAND_SLOTS]]
    canvas_gray = np.concatenate(grays, axis=1)
    bgr = cv2.cvtColor(canvas_gray, cv2.COLOR_GRAY2BGR)

    tenhou, conf = rec.recognize(
        bgr,
        {
            "hand": {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0},
            "round_info": {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0},
            "scores": {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0},
            "turn_counter": {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0},
        },
    )
    assert tenhou["hand"] == TILE_CODES[:HAND_SLOTS]
    assert tenhou["round_label"] == "東4局"
    assert tenhou["round_wind"] == "東"
    # scores / turn は失敗してもスタブ既定値で埋まる
    assert tenhou["scores"] == DEFAULT_TENHOU_JSON["scores"]
    assert tenhou["turn"] == DEFAULT_TENHOU_JSON["turn"]
    assert conf > 0.99


def test_board_recognizer_drops_scores_when_self_wind_not_recognized(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Codex P1 on PR #44: scores OCR が通っても self_wind が未認識なら scores は採用しない。

    scores は座順 (東→南→西→北) で並ぶため、self_wind が既定 "東" のままだと、
    build_board_summary (Rust) は scores[0] を引いて非起家局面で他家の点数を
    自分の持ち点として UI に流す。これを防ぐため self_wind 実認識が無いフレームでは
    scores を既定値のままに残し、初回のみ警告を出す。
    """
    # wind テンプレなし (templates/winds/ 未作成) で scores OCR は成功させる。
    values = iter(["25000", "30000", "20000", "25000"] * 2)  # 複数フレーム分

    class _Side:
        def image_to_string(
            self, img: np.ndarray, lang: str = "eng", config: str = "", **kwargs: Any
        ) -> str:  # noqa: ARG002
            return next(values)

    _install_fake_pytesseract(_Side())

    rec = BoardRecognizer(tmp_path)
    bgr = np.full((40, 160, 3), 128, dtype=np.uint8)
    bgr[5:35, 5:155] = 255

    with caplog.at_level("WARNING", logger="recognition"):
        for _ in range(2):
            tenhou, _ = rec.recognize(bgr, {"scores": {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0}})
            # scores 採用されず既定値 [25000, 25000, 25000, 25000] のまま
            assert tenhou["scores"] == DEFAULT_TENHOU_JSON["scores"]
            assert tenhou["self_wind"] == "東"  # 既定値

    # 警告は対局中 1 回のみ (毎フレーム監視ループでスパムしない)
    warns = [r for r in caplog.records if "scores OCR succeeded but self_wind" in r.getMessage()]
    assert len(warns) == 1


def test_board_recognizer_accepts_scores_when_both_recognized(tmp_path: Path) -> None:
    """self_wind と scores の両方が認識できれば scores はそのまま採用される。"""
    _write_all_wind_templates(tmp_path / "winds")

    values = iter(["25000", "30000", "20000", "25000"])

    class _Side:
        def image_to_string(
            self, img: np.ndarray, lang: str = "eng", config: str = "", **kwargs: Any
        ) -> str:  # noqa: ARG002
            return next(values)

    _install_fake_pytesseract(_Side())

    rec = BoardRecognizer(tmp_path)
    # 南風テンプレと完全一致する画像を self_wind ROI に流す → "南" が返る
    south_gray = _make_pattern(1001)
    bgr = cv2.cvtColor(south_gray, cv2.COLOR_GRAY2BGR)

    tenhou, _ = rec.recognize(
        bgr,
        {
            "self_wind": {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0},
            "scores": {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0},
        },
    )
    assert tenhou["self_wind"] == "南"
    assert tenhou["scores"] == [25000, 30000, 20000, 25000]
