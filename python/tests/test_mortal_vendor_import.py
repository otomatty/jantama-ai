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
# submodule が未初期化 (= 浅い clone や `git submodule update --init` をしていない
# 環境) の場合は `vendor/mortal/` が空ディレクトリだけ存在することがある。
# LICENSE ファイルの有無で「実体が clone 済みか」を判定する。
_SUBMODULE_INITIALIZED = (VENDOR_ROOT / "LICENSE").exists()
_SKIP_REASON_NO_SUBMODULE = (
    "python/vendor/mortal が未初期化。`git submodule update --init --recursive` で取得してください"
)


@pytest.mark.skipif(not _SUBMODULE_INITIALIZED, reason=_SKIP_REASON_NO_SUBMODULE)
def test_vendor_submodule_present() -> None:
    """submodule が clone されていれば期待するファイル構造であることを検証する。"""
    assert (VENDOR_ROOT / "LICENSE").exists()
    assert (VENDOR_ROOT / "mortal" / "engine.py").exists()


@pytest.mark.skipif(not _SUBMODULE_INITIALIZED, reason=_SKIP_REASON_NO_SUBMODULE)
@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="torch (= [mortal] extras) が未インストール",
)
def test_mortal_engine_importable() -> None:
    """namespace package 経由で MortalEngine が import できる。"""
    from vendor.mortal.mortal.engine import MortalEngine

    assert isinstance(MortalEngine, type)
    assert MortalEngine.__name__ == "MortalEngine"
