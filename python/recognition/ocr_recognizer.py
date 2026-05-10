"""Tesseract OCR を使ったテキスト・数字認識 (issue #12)。

- `recognize_round_label`: 局名 (例「東1局」「南3局」) を jpn 言語で OCR
- `recognize_scores`: 点棒 ROI を 4 等分し、各家の持ち点を eng で OCR
- `recognize_turn`: 巡目カウンタ ROI を eng で OCR

`pytesseract` import 失敗 / Tesseract バイナリ不在は warning を 1 度だけ出して
以降は `None` を返す graceful degrade パターン (issue #11 の TileRecognizer と同様)。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import cv2
import numpy as np

from recognition.tile_recognizer import RoiRect

logger = logging.getLogger("recognition")


@dataclass
class _PytesseractRef:
    """`pytesseract` モジュールを 1 度だけインポート試行して保持する。"""

    module: object | None = None
    tried: bool = False
    error: str | None = None


_PYTESSERACT = _PytesseractRef()


def _get_pytesseract():  # type: ignore[no-untyped-def]
    """pytesseract モジュールを返す。未インストール時は `None` + 警告 1 回。"""
    if _PYTESSERACT.tried:
        return _PYTESSERACT.module
    _PYTESSERACT.tried = True
    try:
        # pytesseract は py.typed なし → mypy stubs 無視 (型情報は使わない動的呼び出し)。
        import pytesseract  # type: ignore[import-untyped]  # noqa: PLC0415
    except ImportError as exc:
        _PYTESSERACT.error = str(exc)
        logger.warning(
            "pytesseract not importable (%s); OCR-based fields (round_label, "
            "scores, turn) will be skipped. Install with `uv add pytesseract` "
            "and Tesseract binary separately.",
            exc,
        )
        return None
    _PYTESSERACT.module = pytesseract
    return pytesseract


# 雀魂局名で出現する文字のみを許可。OCR 誤検出 (例: 漢字「東」を「束」と誤読) を抑える。
# 局番号は 1-4 のみ (東1局〜東4局 / 南1局〜南4局)、5 以上は存在しない (CodeRabbit nit on PR #44)。
_ROUND_LABEL_WHITELIST = "東南西北1234局"
_DIGIT_WHITELIST = "0123456789"
_SCORE_WHITELIST = "0123456789-"

# OCR 前処理で低解像 ROI を拡大する目標高さ (px)。Tesseract は x-height が
# 30〜35 px 程度のときに精度が出やすい (公式 FAQ より)。
_OCR_TARGET_HEIGHT = 32

# `pytesseract.image_to_string` の TesseractNotFoundError を例外名で識別 (型 import を避ける)。
# 重要: `TesseractError` (非ゼロ終了, e.g. `jpn` 言語データ欠落) はここに含めない。
# 含めると「設定不備」を「binary 未導入」と誤誘導し、ユーザに `jpn` パック不足の
# 診断機会を奪う (CodeRabbit on PR #44)。`TesseractError` は毎フレーム単発で
# 警告を出し続け、ユーザに修正を促す。
_TESSERACT_NOT_FOUND_NAMES = {"TesseractNotFoundError"}


def _crop_roi(bgr_frame: np.ndarray, roi: RoiRect) -> np.ndarray | None:
    """ROI 比率を実ピクセルに直して切り出し。サイズ 0 になったら `None`。"""
    if bgr_frame is None or bgr_frame.size == 0:
        return None
    h, w = bgr_frame.shape[:2]
    x0 = max(0, min(w, int(roi.x * w)))
    y0 = max(0, min(h, int(roi.y * h)))
    x1 = max(0, min(w, int((roi.x + roi.w) * w)))
    y1 = max(0, min(h, int((roi.y + roi.h) * h)))
    if x1 <= x0 or y1 <= y0:
        return None
    return bgr_frame[y0:y1, x0:x1]


def _preprocess_for_ocr(crop_bgr: np.ndarray) -> np.ndarray:
    """OCR 前処理: 軽く拡大 → グレースケール化 → 2 値化 (gemini medium on PR #44)。

    リサイズは 2 値化前に行う。2 値化後にリサイズすると階段状のジャギーが
    残って Tesseract の細い線の読み取り精度が落ちるため。
    """
    h, w = crop_bgr.shape[:2]
    # 直接ターゲット高さに合わせる。整数倍 (32 // h) スケーリングだと h=20 で
    # scale=1 (拡大なし) や h=15 で scale=2 (30px = 目標未達) になり、低解像 ROI
    # での精度劣化が黙って発生していた (Codex P2 on PR #44)。
    if 0 < h < _OCR_TARGET_HEIGHT:
        new_w = max(1, round(w * _OCR_TARGET_HEIGHT / h))
        crop_bgr = cv2.resize(crop_bgr, (new_w, _OCR_TARGET_HEIGHT), interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    # 雀魂 UI は概ね白文字 / 黒背景 or 暗文字 / 明背景。`THRESH_OTSU` で自動 2 値化。
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return binary


def _ocr_string(img: np.ndarray, lang: str, whitelist: str) -> str | None:
    """1 ROI に対する OCR 呼び出し。Tesseract 不在は `None`。"""
    pyt = _get_pytesseract()
    if pyt is None:
        return None
    # binary 未導入と一度確定したら、以降は例外 throw/catch のコストを払わず即 None。
    # 監視ループで毎フレーム 3 関数 × image_to_string が同じ TesseractNotFoundError を
    # 投げ続けるのを避ける (CodeRabbit Major on PR #44)。
    if _PYTESSERACT.error in _TESSERACT_NOT_FOUND_NAMES:
        return None
    config = f"--psm 7 -c tessedit_char_whitelist={whitelist}"
    try:
        # `image_to_string` の戻り値型は実装上 str。型注釈なしの動的呼び出し。
        text: str = pyt.image_to_string(img, lang=lang, config=config)  # type: ignore[attr-defined]
    except Exception as exc:  # noqa: BLE001 — Tesseract 不在 / 内部エラーを統一して握る
        name = type(exc).__name__
        if name in _TESSERACT_NOT_FOUND_NAMES:
            # 警告は初回のみ。`_PYTESSERACT.error` への記録は毎回行い、上の早期 return
            # が以降のフレームで確実に効くようにする (CodeRabbit Major on PR #44)。
            if _PYTESSERACT.error is None:
                logger.warning(
                    "Tesseract binary not found (%s); OCR-based fields disabled. "
                    "Install Tesseract from https://github.com/UB-Mannheim/tesseract/wiki "
                    "(Windows) and ensure it's on PATH.",
                    exc,
                )
            _PYTESSERACT.error = name
        else:
            logger.warning("Tesseract OCR failed (%s): %s", name, exc)
        return None
    return text.strip() if isinstance(text, str) else None


def recognize_round_label(
    bgr_frame: np.ndarray,
    round_roi: RoiRect | None,
) -> str | None:
    """局名 (「東1局」「南3局」等) を返す。失敗時は `None`。"""
    if round_roi is None:
        return None
    crop = _crop_roi(bgr_frame, round_roi)
    if crop is None:
        return None
    img = _preprocess_for_ocr(crop)
    text = _ocr_string(img, lang="jpn", whitelist=_ROUND_LABEL_WHITELIST)
    if not text:
        return None
    # OCR 結果から最初の「(東|南|西|北)\d局?」を抽出。「局」が読めなくても許す。
    m = re.search(r"([東南西北])(\d)局?", text)
    if not m:
        return None
    wind, num = m.group(1), m.group(2)
    if num not in {"1", "2", "3", "4"}:
        return None
    return f"{wind}{num}局"


def recognize_scores(
    bgr_frame: np.ndarray,
    scores_roi: RoiRect | None,
) -> list[int] | None:
    """点棒 ROI を 4 等分して 4 家分の持ち点を返す (座順: 東→南→西→北)。

    レイアウト前提: 1 つの矩形に 4 家分が水平に並んだ「点棒帯」を想定
    (gemini medium on PR #44)。雀魂の標準対戦画面はスコア表示が卓中央の
    十字配置 (上下左右に 1 つずつ) なので、その場合はユーザがオーバーレイ
    UI 経由で別途並べて表示するスキンを使うか、将来 issue で 4 個独立した
    ROI (`score_self/right/across/left`) に分割する。

    どれか 1 つでも OCR 失敗 / パース失敗があれば `None` を返す
    (一部だけ欠けた scores 配列を tenhou_json に流すと self_wind_index で
    異常値を引いて Mortal が誤動作する恐れがあるため、all-or-nothing)。
    """
    if scores_roi is None:
        return None
    crop = _crop_roi(bgr_frame, scores_roi)
    if crop is None:
        return None

    h, w = crop.shape[:2]
    if w < 4:
        return None
    seg_w = w // 4
    scores: list[int] = []
    for i in range(4):
        sx = i * seg_w
        ex = w if i == 3 else (i + 1) * seg_w
        seg = crop[:, sx:ex]
        if seg.size == 0:
            return None
        img = _preprocess_for_ocr(seg)
        text = _ocr_string(img, lang="eng", whitelist=_SCORE_WHITELIST)
        if not text:
            return None
        # 「-」が先頭にだけ来る場合のみ符号として許容。それ以外は除去。
        sign = -1 if text.startswith("-") else 1
        digits = re.sub(r"[^0-9]", "", text)
        if not digits:
            return None
        try:
            value = sign * int(digits)
        except ValueError:
            return None
        # 雀魂の点棒は 100 点単位。OCR 誤読 (25000 → 2500 / 2500 → 250 など桁落ち)
        # の sanity check として 100 で割り切れない値は失敗扱い (gemini medium on PR #44)。
        if value % 100 != 0:
            return None
        scores.append(value)
    return scores


def recognize_turn(
    bgr_frame: np.ndarray,
    turn_roi: RoiRect | None,
) -> int | None:
    """巡目カウンタ ROI を読み整数を返す。範囲外 / 失敗時は `None`。

    上限は流局時の最大巡数 + 副露でツモ順がずれて伸びるケースを考慮して 25
    までを許容 (gemini medium on PR #44)。理論最大は配牌後 70 山牌 / 4 家 ≈
    17-18 巡だが、リンシャン牌・カンドラ複数発動・副露によるツモ飛ばし等で
    25 程度までは現実的に到達し得る。
    """
    if turn_roi is None:
        return None
    crop = _crop_roi(bgr_frame, turn_roi)
    if crop is None:
        return None
    img = _preprocess_for_ocr(crop)
    text = _ocr_string(img, lang="eng", whitelist=_DIGIT_WHITELIST)
    if not text:
        return None
    digits = re.sub(r"[^0-9]", "", text)
    if not digits:
        return None
    try:
        value = int(digits)
    except ValueError:
        return None
    if not 1 <= value <= 25:
        return None
    return value


def round_label_to_wind(round_label: str | None) -> str | None:
    """局名から場風 (1 文字) を導出。「東1局」→「東」。"""
    if not round_label:
        return None
    head = round_label[0]
    if head in {"東", "南", "西", "北"}:
        return head
    return None
