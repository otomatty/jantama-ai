"""extract_template CLI の引数バリデーション (issue #16)。

`_parse_size` が 0 や負値を弾いて argparse 経由でエラー化することを確認する
(Codex P2 on PR #48: cv2.resize に 0/負値を渡すと cv2.error が投げられ
main の except では捕捉できないため、入力時点で弾く)。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from recognition.tools.extract_template import _DEFAULT_OUT_DIR, _parse_size


def test_parse_size_accepts_positive_dimensions() -> None:
    assert _parse_size("64x96") == (64, 96)
    assert _parse_size("1x1") == (1, 1)


def test_parse_size_rejects_zero_or_negative() -> None:
    for spec in ("0x96", "64x0", "-1x96", "64x-1", "0x0"):
        with pytest.raises(argparse.ArgumentTypeError, match="must be positive"):
            _parse_size(spec)


def test_parse_size_rejects_malformed() -> None:
    for spec in ("64", "64x", "xx", "abc", "64x96x32"):
        with pytest.raises(argparse.ArgumentTypeError, match="invalid --size"):
            _parse_size(spec)


def test_default_out_dir_is_recognition_templates_regardless_of_cwd() -> None:
    """`--out` のデフォルトは cwd 非依存で `recognition/templates` を指す。

    Codex P2 on PR #48: 既定パスが相対だと `cd python` 後に
    `python/python/recognition/templates` に書かれてサイレントに recognition.main
    が読む場所からズレる。`__file__` 起点で絶対パス化することで防ぐ。
    """
    assert _DEFAULT_OUT_DIR.is_absolute()
    # ソース位置: python/recognition/tools/extract_template.py
    # 期待デフォルト: python/recognition/templates/
    expected = Path(__file__).resolve().parent.parent / "recognition" / "templates"
    assert expected == _DEFAULT_OUT_DIR
