"""issue #17: vendor 取り込んだ Mortal が import できるかを検証する。

`torch` (= `[mortal]` extras) が未インストールの環境では skip する。
受け入れ基準: `from vendor.mortal.mortal.engine import MortalEngine` が通り、
`MortalEngine` クラスが参照できる。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

# python/mortal/__init__.py の副作用で python/ が sys.path に乗る。
import mortal  # noqa: F401

VENDOR_ROOT = Path(__file__).resolve().parents[1] / "vendor" / "mortal"


def test_vendor_submodule_present() -> None:
    """submodule が clone されていること (.git ファイル or LICENSE で判定)。"""
    assert VENDOR_ROOT.exists(), "submodule python/vendor/mortal が未取得"
    assert (VENDOR_ROOT / "LICENSE").exists()
    assert (VENDOR_ROOT / "mortal" / "engine.py").exists()


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="torch (= [mortal] extras) が未インストール",
)
def test_mortal_engine_importable() -> None:
    """namespace package 経由で MortalEngine が import できる。"""
    from vendor.mortal.mortal.engine import MortalEngine

    assert isinstance(MortalEngine, type)
    assert MortalEngine.__name__ == "MortalEngine"
