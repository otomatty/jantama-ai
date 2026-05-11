"""Mortal 推論プロセス。

PRD §8.2: PyTorch (ROCm 7.2.1+ on Windows / フォールバック CPU) で
Mortal 本体 (https://github.com/Equim-chan/Mortal) を実行する。

Mortal 本体は `python/vendor/mortal/` に git submodule で取り込み、
namespace package として `vendor.mortal.mortal.<module>` でアクセスする。
直接スクリプト実行 (`python mortal/main.py`) されたケースでも import が
通るよう、本パッケージのロード時に `python/` を sys.path に追加する。
"""

from __future__ import annotations

import sys
from pathlib import Path

_PYTHON_ROOT = Path(__file__).resolve().parent.parent
if str(_PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(_PYTHON_ROOT))
