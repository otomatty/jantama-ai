"""スクショから牌 / 風 / アクションのテンプレート PNG を切り出す CLI (issue #16)。

雀魂のスクショと牌コード (`1m`〜`7z` / `winds/east` / `actions/chi` 等) と
ピクセル矩形 (`x y w h`) を指定すると、

1. グレースケール変換、
2. `--size` 指定時はテンプレ用サイズへ `cv2.resize`、
3. `<out>/<code>.png` に保存、

までを行う。`load_tile_templates` / `wind_recognizer._load` /
`turn_recognizer._load` が期待する命名規則 (`{code}.png`,
`winds/{key}.png`, `actions/{key}.png`) に合わせて出力する。

使い方:

    uv run python -m recognition.tools.extract_template \\
        screenshot.png 1m 540 940 60 90 \\
        --out python/recognition/templates --size 64x96
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2

from recognition.tile_recognizer import TILE_CODES
from recognition.turn_recognizer import ACTION_KEYS

logger = logging.getLogger("recognition.tools.extract_template")

WIND_KEYS: tuple[str, ...] = ("east", "south", "west", "north")


def _valid_codes() -> set[str]:
    """切り出し先として受け付けるコードの全集合。"""
    codes: set[str] = set(TILE_CODES)
    codes.update(f"winds/{k}" for k in WIND_KEYS)
    codes.update(f"actions/{k}" for k in ACTION_KEYS)
    return codes


def _parse_size(spec: str) -> tuple[int, int]:
    """`64x96` 形式の `--size` 引数を `(width, height)` に変換する。"""
    try:
        w_str, h_str = spec.lower().split("x", 1)
        return int(w_str), int(h_str)
    except (ValueError, AttributeError) as exc:
        raise argparse.ArgumentTypeError(
            f"invalid --size '{spec}', expected WxH (e.g. 64x96)"
        ) from exc


def extract(
    screenshot: Path,
    code: str,
    rect: tuple[int, int, int, int],
    out_dir: Path,
    resize: tuple[int, int] | None,
) -> Path:
    """スクショから 1 枚切り出して保存し、書き出し先パスを返す。"""
    valid = _valid_codes()
    if code not in valid:
        raise ValueError(
            f"unknown code '{code}'. Expected one of TILE_CODES ({len(TILE_CODES)} entries), "
            f"winds/{{{','.join(WIND_KEYS)}}}, or actions/{{{','.join(ACTION_KEYS)}}}."
        )

    img = cv2.imread(str(screenshot))
    if img is None:
        raise FileNotFoundError(f"failed to read screenshot: {screenshot}")

    x, y, w, h = rect
    if w <= 0 or h <= 0:
        raise ValueError(f"width/height must be positive, got w={w} h={h}")
    img_h, img_w = img.shape[:2]
    if x < 0 or y < 0 or x + w > img_w or y + h > img_h:
        raise ValueError(f"rect ({x},{y},{w},{h}) is outside screenshot bounds ({img_w}x{img_h})")

    crop = img[y : y + h, x : x + w]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    if resize is not None:
        gray = cv2.resize(gray, resize)

    out_path = out_dir / f"{code}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_path), gray):
        raise OSError(f"cv2.imwrite failed: {out_path}")
    return out_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m recognition.tools.extract_template",
        description="雀魂スクショから 1 枚分のテンプレ PNG を切り出して保存する。",
    )
    parser.add_argument("screenshot", type=Path, help="入力スクショ画像 (PNG/JPEG)")
    parser.add_argument(
        "code",
        help="牌コード: 1m..9m,0m,1p..0p,1s..0s,1z..7z / winds/<east|south|west|north> / actions/<key>",
    )
    parser.add_argument("x", type=int, help="切り出し左上 x (ピクセル)")
    parser.add_argument("y", type=int, help="切り出し左上 y (ピクセル)")
    parser.add_argument("w", type=int, help="切り出し幅 (ピクセル)")
    parser.add_argument("h", type=int, help="切り出し高さ (ピクセル)")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("python/recognition/templates"),
        help="書き出し先のテンプレートディレクトリ (デフォルト: python/recognition/templates)",
    )
    parser.add_argument(
        "--size",
        type=_parse_size,
        default=None,
        help="保存前にリサイズするサイズ (例: 64x96)。省略時はクロップサイズのまま。",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        out = extract(
            screenshot=args.screenshot,
            code=args.code,
            rect=(args.x, args.y, args.w, args.h),
            out_dir=args.out,
            resize=args.size,
        )
    except (ValueError, FileNotFoundError, OSError) as exc:
        logger.error("%s", exc)
        return 2
    logger.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
