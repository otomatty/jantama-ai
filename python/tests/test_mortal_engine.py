"""issue #18: MortalEngine ロード / スタブ動作の検証。

vendor submodule / torch は重い依存なので、本テストは
- スタブモード (依存ゼロ)
- ファイル不在エラー (依存ゼロ)
- 不正ファイルでの ModelLoadError (torch がある場合のみ)
の観点で確認する。
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import pytest

import mortal  # noqa: F401 — sys.path 調整副作用
from mortal.main import _build_engine
from mortal.mortal_engine import ModelLoadError, MortalEngine


def _make_args(*, model: str | None, backend: str) -> argparse.Namespace:
    ns = argparse.Namespace()
    ns.model = model
    ns.backend = backend
    return ns


def test_stub_engine_is_ready() -> None:
    engine = MortalEngine.stub()
    assert engine.is_ready() is True
    assert engine.name == "jantama-mortal-stub"


def test_stub_engine_infer_shape() -> None:
    """issue #20 受け入れ基準: スタブで dummy mjai event → candidates 5 件 + primary_label。"""
    engine = MortalEngine.stub()
    result = engine.infer([{"type": "tsumo", "actor": 0, "pai": "5m"}])
    # InferenceResult TS 型 (src/types/index.ts:187) 必須キー。
    assert "recommended" in result
    assert "candidates" in result
    assert "timestamp" in result
    assert "primary_label" in result
    # 整形ロジック (action_formatter.format_inference_result) の挙動を検証。
    assert isinstance(result["candidates"], list)
    assert len(result["candidates"]) == 5
    rec = result["recommended"]
    assert rec["action_type"] == "discard"
    assert rec["tile"] == "6m"
    assert result["primary_label"] == "6m を切る"
    # 各 candidate に probability / expected_value が付与される。
    for c in result["candidates"]:
        assert "action_type" in c
        assert "probability" in c
        assert "expected_value" in c
    # raw policy/value/q_values (デバッグ用) も同梱されている。
    assert "policy" in result
    assert "value" in result
    assert "q_values" in result


def test_engine_default_not_ready() -> None:
    engine = MortalEngine()
    assert engine.is_ready() is False
    with pytest.raises(RuntimeError, match="not ready"):
        engine.infer([])


def test_from_pretrained_missing_file_raises(tmp_path: Path) -> None:
    missing = tmp_path / "no_such_model.pth"
    with pytest.raises(ModelLoadError, match="not found"):
        MortalEngine.from_pretrained(missing, backend="cpu")


def test_load_missing_file_does_not_require_torch(tmp_path: Path) -> None:
    """ファイル存在チェックは torch import 前に行われる (依存ゼロで失敗する)。"""
    engine = MortalEngine()
    with pytest.raises(ModelLoadError, match="not found"):
        engine.load(tmp_path / "no_such.pth", backend="cpu")


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="torch (= [mortal] extras) が未インストール",
)
def test_load_unknown_backend_raises(tmp_path: Path) -> None:
    """ファイル存在チェックは通っても backend が不正なら ValueError。"""
    fake = tmp_path / "fake.pth"
    fake.write_bytes(b"not a real pth")
    engine = MortalEngine()
    with pytest.raises(ValueError, match="unknown backend"):
        engine.load(fake, backend="tpu")


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="torch (= [mortal] extras) が未インストール",
)
def test_load_invalid_file_raises(tmp_path: Path) -> None:
    """空ファイルでも `ModelLoadError` にラップされる (素の例外を漏らさない)。"""
    bad = tmp_path / "bad.pth"
    bad.write_bytes(b"not a real pth")
    engine = MortalEngine()
    with pytest.raises(ModelLoadError):
        engine.load(bad, backend="cpu")


def test_build_engine_stub_via_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """`JANTAMA_STUB=1` で main 側がスタブにフォールバックする。"""
    monkeypatch.setenv("JANTAMA_STUB", "1")
    engine = _build_engine(_make_args(model=None, backend="cpu"))
    assert engine.is_ready()
    assert engine.name.endswith("stub")


def test_build_engine_missing_model_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    """`JANTAMA_STUB` 無し + `--model` 無し → SystemExit(2)。"""
    monkeypatch.delenv("JANTAMA_STUB", raising=False)
    with pytest.raises(SystemExit) as excinfo:
        _build_engine(_make_args(model=None, backend="cpu"))
    assert excinfo.value.code == 2


def test_build_engine_stub_env_empty_string_not_triggered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`JANTAMA_STUB=""` は stub モードを発火しない (== "1" 厳密一致が契約)。

    Rust 側 `spawn_mortal` が、実モデル指定時に親プロセスから継承された
    `JANTAMA_STUB=1` を空文字列で上書きすることで stub を抑止する設計
    (PR #50 codex P1)。本テストはその上書き契約 (= 空文字列なら stub に
    落ちない) が Python 側で保証されていることを検証する。
    """
    monkeypatch.setenv("JANTAMA_STUB", "")
    # `--model` 未指定なので stub にならなければ SystemExit(2) になるはず。
    with pytest.raises(SystemExit) as excinfo:
        _build_engine(_make_args(model=None, backend="cpu"))
    assert excinfo.value.code == 2


def test_stub_flag_is_removed_from_main_parser() -> None:
    """旧 `--stub` フラグは廃止済み (main.py のソースに残っていない)。"""
    src = Path(mortal.__file__).parent / "main.py"
    text = src.read_text(encoding="utf-8")
    assert (
        '"--stub"' not in text and "'--stub'" not in text
    ), "--stub フラグが main.py に残っています (issue #18 で廃止)"


def test_mortal_engine_module_has_no_top_level_torch_import() -> None:
    """`mortal_engine` の import で torch が即時引き込まれないこと。

    torch (= [mortal] extras) が未インストールの環境でも `from mortal.mortal_engine
    import MortalEngine` が成功する必要があるため、`import torch` は遅延 import
    に限定する。
    """
    src = Path(mortal.__file__).parent / "mortal_engine.py"
    text = src.read_text(encoding="utf-8")
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0]  # コメント除去
        if line.startswith("import torch") or line.startswith("from torch"):
            raise AssertionError(
                f"top-level torch import 検出: {raw_line!r}; 遅延 import にしてください"
            )
