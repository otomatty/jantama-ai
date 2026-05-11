"""Mortal モデルロード / 推論ラッパ。

`torch.load` で `.pth` を読み込み、vendor Mortal の `Brain` / `DQN` を
構築して state_dict をロードする。推論は vendor 側 `MortalEngine.react_batch`
にデリゲートする想定だが、`tenhou_json -> obs/mask` 変換は Phase D3/D4
で実装するため本クラスからは現状スタブ応答を返す。

vendor submodule (`python/vendor/mortal/`) が未取得、または `torch`
([mortal] extras) が未インストールの環境では `ModelLoadError` を送出する。
後方互換用に `JANTAMA_STUB=1` を main.py 側で見て本クラスの `stub()`
コンストラクタにフォールバックできるようにしている (本クラス自体は環境変数
を読まない — 上位プロセス責務)。
"""

from __future__ import annotations

import importlib
import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger("mortal.engine")


class ModelLoadError(RuntimeError):
    """Mortal モデルロード失敗を表す例外。

    ファイル不在 / state_dict のキー不整合 (= モデルバージョン不一致) /
    GPU OOM / vendor submodule 未取得 / torch 未インストール などを
    本例外にラップする。`__cause__` に元例外を保持する。
    """


_STUB_CANDIDATES: list[dict[str, Any]] = [
    {"tile": "6m", "action_type": "discard", "expected_value": 0.32},
    {"tile": "9p", "action_type": "discard", "expected_value": 0.18},
    {"tile": "1z", "action_type": "discard", "expected_value": -0.05},
]


def _stub_result() -> dict[str, Any]:
    return {
        "recommended": _STUB_CANDIDATES[0],
        "candidates": list(_STUB_CANDIDATES),
        "timestamp": datetime.now(UTC).isoformat(),
    }


def _resolve_device(torch: Any, backend: str) -> Any:
    """`--backend` 引数 (rocm|cuda|cpu) から `torch.device` を返す。

    ROCm 環境でも PyTorch 上は `torch.cuda.is_available()` が True を返す
    ため、`rocm` と `cuda` を同一視する。GPU が利用不可なら CPU にフォール
    バックして警告を出す。
    """
    backend_lower = backend.lower()
    if backend_lower in {"rocm", "cuda"}:
        if not torch.cuda.is_available():
            logger.warning(
                "backend=%s が指定されたが CUDA/ROCm が利用不可。CPU にフォールバックします",
                backend,
            )
            return torch.device("cpu")
        return torch.device("cuda")
    if backend_lower == "cpu":
        return torch.device("cpu")
    raise ValueError(f"unknown backend: {backend!r} (expected 'rocm' | 'cuda' | 'cpu')")


class MortalEngine:
    """Mortal モデルのロード・推論をカプセル化するラッパ。

    使い方:

    >>> engine = MortalEngine.from_pretrained("/path/to/mortal.pth", backend="cpu")
    >>> engine.is_ready()
    True
    >>> engine.infer(tenhou_json)
    {"recommended": ..., "candidates": [...], ...}

    スタブ用途:

    >>> engine = MortalEngine.stub()
    >>> engine.is_ready()
    True
    """

    def __init__(self, *, stub: bool = False, name: str = "jantama-mortal") -> None:
        self._stub = stub
        self._name = name
        self._device: Any = None
        self._engine: Any = None  # vendor.mortal.mortal.engine.MortalEngine
        self._version: int | None = None
        # スタブモードはロード不要で常に ready 扱い
        self._ready = stub

    @classmethod
    def stub(cls, *, name: str = "jantama-mortal-stub") -> MortalEngine:
        logger.info("MortalEngine: stub mode (no model loaded)")
        return cls(stub=True, name=name)

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Path,
        *,
        backend: str = "cpu",
        name: str = "jantama-mortal",
    ) -> MortalEngine:
        engine = cls(stub=False, name=name)
        engine.load(model_path, backend=backend)
        return engine

    def load(self, model_path: str | Path, *, backend: str = "cpu") -> None:
        """`.pth` をロードして Brain/DQN を構築する。

        Raises:
            ModelLoadError: ファイル不在 / torch 未インストール /
                vendor 未取得 / state_dict 不整合 / OOM 時。
        """
        path = Path(model_path)
        # ファイル存在チェックは torch import 前に行うことで、エラーメッセージを
        # 利用者にとってわかりやすくする (依存欠落より「パス間違い」の方が頻度
        # が高いため)。
        if not path.is_file():
            raise ModelLoadError(f"model file not found: {path}")

        try:
            torch = importlib.import_module("torch")
        except ImportError as exc:
            raise ModelLoadError("torch is not installed; run `uv sync --extra mortal`") from exc

        device = _resolve_device(torch, backend)
        logger.info(
            "MortalEngine: loading model from %s (device=%s, backend=%s)",
            path,
            device,
            backend,
        )
        t0 = time.perf_counter()

        state = self._torch_load(torch, path, device)

        cfg = state.get("config") or {}
        control = cfg.get("control") or {}
        resnet = cfg.get("resnet") or {}
        version = int(control.get("version", 1))
        num_blocks = int(resnet.get("num_blocks", 40))
        conv_channels = int(resnet.get("conv_channels", 192))

        try:
            model_module = importlib.import_module("vendor.mortal.mortal.model")
            engine_module = importlib.import_module("vendor.mortal.mortal.engine")
        except ImportError as exc:
            raise ModelLoadError(
                "vendor mortal not available; run `git submodule update --init --recursive`"
            ) from exc

        Brain = model_module.Brain  # noqa: N806 — vendor のクラス名をそのまま参照
        DQN = model_module.DQN  # noqa: N806
        VendorEngine = engine_module.MortalEngine  # noqa: N806

        try:
            brain = Brain(
                version=version, num_blocks=num_blocks, conv_channels=conv_channels
            ).eval()
            dqn = DQN(version=version).eval()
            brain.load_state_dict(state["mortal"])
            dqn.load_state_dict(state["current_dqn"])
        except KeyError as exc:
            raise ModelLoadError(
                f"state_dict missing expected key {exc}; model version mismatch?"
            ) from exc
        except RuntimeError as exc:
            # load_state_dict の shape mismatch も RuntimeError
            raise ModelLoadError(f"state_dict load failed: {exc}") from exc

        try:
            brain = brain.to(device)
            dqn = dqn.to(device)
        except RuntimeError as exc:
            # torch.cuda.OutOfMemoryError は torch>=2.4 で RuntimeError サブ
            # クラスなので、ここで一括して捕捉してメッセージを付け替える。
            raise ModelLoadError(f"failed to move model to {device}: {exc}") from exc

        self._engine = VendorEngine(
            brain=brain,
            dqn=dqn,
            is_oracle=False,
            version=version,
            device=device,
            enable_amp=False,
            enable_quick_eval=True,
            enable_rule_based_agari_guard=True,
            name=self._name,
        )
        self._device = device
        self._version = version
        self._ready = True

        elapsed = time.perf_counter() - t0
        logger.info(
            "MortalEngine: load complete in %.2f s (version=%d, num_blocks=%d, conv_channels=%d)",
            elapsed,
            version,
            num_blocks,
            conv_channels,
        )

    @staticmethod
    def _torch_load(torch: Any, path: Path, device: Any) -> dict[str, Any]:
        """`torch.load` を `weights_only=True` 優先で試し、失敗したら False で再試行。

        torch>=2.6 で `weights_only` デフォルトが True に変わったが、
        Mortal の `.pth` は `state['config']` に dict を含むため、古い保存
        フォーマットだと True で拒否される場合がある。利用者が指定したファイル
        なので False フォールバックは許容する (信頼前提)。
        """
        try:
            return torch.load(str(path), map_location=device, weights_only=True)
        except FileNotFoundError as exc:
            raise ModelLoadError(f"model file not found: {path}") from exc
        except Exception as exc_safe:
            logger.warning(
                "torch.load(weights_only=True) failed (%s); retrying with weights_only=False",
                exc_safe,
            )
            try:
                return torch.load(str(path), map_location=device, weights_only=False)
            except FileNotFoundError as exc:
                raise ModelLoadError(f"model file not found: {path}") from exc
            except Exception as exc:
                raise ModelLoadError(f"failed to torch.load: {exc}") from exc

    def is_ready(self) -> bool:
        return self._ready

    @property
    def device(self) -> Any:
        return self._device

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> int | None:
        return self._version

    def infer(self, tenhou_json: dict[str, Any]) -> dict[str, Any]:
        """`tenhou_json` から推奨アクションを返す。

        Phase D3/D4 で `tenhou_json -> obs/mask` 変換 + `react_batch`
        呼び出しを実装する。それまではモデルがロード済みでもスタブ応答を返す
        (recommended / candidates / timestamp の形状は本実装と互換)。
        """
        if not self._ready:
            raise RuntimeError("MortalEngine is not ready; call load() first")
        # 引数は将来の変換実装で使う。現状はスタブのため未使用。
        del tenhou_json
        return _stub_result()
