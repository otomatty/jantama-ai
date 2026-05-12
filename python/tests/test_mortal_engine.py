"""issue #18: MortalEngine ロード / スタブ動作の検証。

vendor submodule / torch は重い依存なので、本テストは
- スタブモード (依存ゼロ)
- ファイル不在エラー (依存ゼロ)
- 不正ファイルでの ModelLoadError (torch がある場合のみ)
- CUDA RuntimeError → CPU フォールバック (issue #21, torch 不要 / MagicMock)
の観点で確認する。
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import mortal  # noqa: F401 — sys.path 調整副作用
from mortal.main import _build_engine
from mortal.mortal_engine import (
    ModelLoadError,
    MortalEngine,
    _is_cuda_runtime_error,
)


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


# --- issue #21: ROCm/CUDA → CPU 動的フォールバック ------------------------


def test_is_cuda_runtime_error_matches_cuda_markers() -> None:
    """`_is_cuda_runtime_error` が CUDA / ROCm 由来 RuntimeError を判定する。"""
    assert _is_cuda_runtime_error(RuntimeError("CUDA error: device-side assert triggered"))
    assert _is_cuda_runtime_error(RuntimeError("HIP error: invalid device function"))
    assert _is_cuda_runtime_error(RuntimeError("CUDA out of memory."))
    assert _is_cuda_runtime_error(RuntimeError("no CUDA-capable device is detected"))
    assert _is_cuda_runtime_error(
        RuntimeError("CUDA driver version is insufficient for CUDA runtime version")
    )
    assert _is_cuda_runtime_error(RuntimeError("CUDA initialization failed"))


def test_is_cuda_runtime_error_rejects_non_cuda() -> None:
    """state_dict shape mismatch 等の非 CUDA RuntimeError は False。"""
    assert not _is_cuda_runtime_error(RuntimeError("size mismatch for layer.weight"))
    assert not _is_cuda_runtime_error(RuntimeError("Error(s) in loading state_dict for Brain"))
    assert not _is_cuda_runtime_error(ValueError("not a RuntimeError"))
    assert not _is_cuda_runtime_error(None)


def _install_fake_torch(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """`mortal.mortal_engine` の `importlib.import_module("torch")` 経路を
    MagicMock 製の擬似 torch に差し替える。CUDA 利用可と装い、`device(...)` は
    `.type` 属性に渡された文字列を持つ MagicMock を返す。
    """
    fake_torch = MagicMock(name="torch")
    fake_torch.cuda.is_available.return_value = True

    def _make_device(spec: str) -> MagicMock:
        dev = MagicMock(name=f"device({spec!r})")
        dev.type = spec
        return dev

    fake_torch.device.side_effect = _make_device

    real_import_module = importlib.import_module

    def fake_import_module(name: str, *args: object, **kwargs: object) -> object:
        if name == "torch":
            return fake_torch
        return real_import_module(name, *args, **kwargs)

    monkeypatch.setattr("mortal.mortal_engine.importlib.import_module", fake_import_module)
    return fake_torch


def test_load_falls_back_to_cpu_on_cuda_runtime_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """`backend=rocm` で `_load_with_device` が CUDA error を上げたら CPU で再試行。

    issue #21 受け入れ基準: `torch.cuda.is_available()==True` でも実際の
    GPU 操作で `RuntimeError: CUDA error: ...` が飛ぶケース (ROCm 不安定環境)
    を検知して、警告ログを出しながら CPU で 1 回だけリトライする。
    """
    _install_fake_torch(monkeypatch)

    fake = tmp_path / "fake.pth"
    fake.write_bytes(b"x")

    calls: list[str] = []

    def fake_load_with_device(
        self_inner: MortalEngine,
        torch_mod: object,
        path: Path,
        device: object,
        *,
        backend: str,
    ) -> None:
        del torch_mod, path, backend
        dev_type = getattr(device, "type", None)
        calls.append(str(dev_type))
        if dev_type == "cuda":
            raise ModelLoadError(f"failed to move model to {device}") from RuntimeError(
                "CUDA error: device-side assert triggered"
            )
        # CPU 経路は成功扱い (実 vendor を呼ばずに ready 状態へ遷移)
        self_inner._device = device
        self_inner._version = 1
        self_inner._ready = True

    monkeypatch.setattr(MortalEngine, "_load_with_device", fake_load_with_device)

    engine = MortalEngine()
    with caplog.at_level("WARNING", logger="mortal.engine"):
        engine.load(fake, backend="rocm")

    assert calls == ["cuda", "cpu"], "GPU → CPU の順で 1 回ずつ呼ばれる"
    assert engine.is_ready()
    assert engine.device.type == "cpu"
    assert any(
        "CPU にフォールバック" in rec.message and "ROCm/CUDA 起動失敗" in rec.message
        for rec in caplog.records
    ), "フォールバック警告ログ (B7: python-log Tauri Event 経由で UI に届く) が出ていること"


def test_load_does_not_retry_on_non_cuda_runtime_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """state_dict shape mismatch 等の非 CUDA RuntimeError は CPU 再試行しない。

    CPU で再試行しても直らないので、原例外を `ModelLoadError` のままユーザに
    返してプロセスを失敗させる方が誤魔化しがなく良い。
    """
    _install_fake_torch(monkeypatch)

    fake = tmp_path / "fake.pth"
    fake.write_bytes(b"x")

    calls: list[str] = []

    def fake_load_with_device(
        self_inner: MortalEngine,
        torch_mod: object,
        path: Path,
        device: object,
        *,
        backend: str,
    ) -> None:
        del self_inner, torch_mod, path, backend
        calls.append(str(getattr(device, "type", None)))
        raise ModelLoadError("state_dict load failed") from RuntimeError(
            "size mismatch for layer.weight"
        )

    monkeypatch.setattr(MortalEngine, "_load_with_device", fake_load_with_device)

    engine = MortalEngine()
    with pytest.raises(ModelLoadError, match="state_dict load failed"):
        engine.load(fake, backend="rocm")
    assert calls == ["cuda"], "CPU リトライは起きない"


def test_load_no_retry_when_backend_is_cpu(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """`backend=cpu` 指定時は `device.type=="cpu"` なのでリトライ条件に入らない。"""
    _install_fake_torch(monkeypatch)

    fake = tmp_path / "fake.pth"
    fake.write_bytes(b"x")

    calls: list[str] = []

    def fake_load_with_device(
        self_inner: MortalEngine,
        torch_mod: object,
        path: Path,
        device: object,
        *,
        backend: str,
    ) -> None:
        del self_inner, torch_mod, path, backend
        calls.append(str(getattr(device, "type", None)))
        # CPU 経路で起きた "CUDA error" は本来あり得ないが、防御的に
        # それでも GPU リトライが走らないこと (= 同じ device で無限再試行
        # にならないこと) を確認する。
        raise ModelLoadError("simulated") from RuntimeError(
            "CUDA error: should not trigger retry on cpu backend"
        )

    monkeypatch.setattr(MortalEngine, "_load_with_device", fake_load_with_device)

    engine = MortalEngine()
    with pytest.raises(ModelLoadError, match="simulated"):
        engine.load(fake, backend="cpu")
    assert calls == ["cpu"], "CPU 指定では 1 回しか呼ばれない"
