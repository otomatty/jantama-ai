"""Mortal モデルロード / 推論ラッパ。

`torch.load` で `.pth` を読み込み、vendor Mortal の `Brain` / `DQN` を
構築して state_dict をロードする。推論は vendor 側 `MortalEngine.react_batch`
にデリゲートする想定だが、libriichi `mjai.Bot` ↔ vendor `react_batch`
の配線は Phase D5 (後続 issue) に分離する。本クラスの `infer()` は
スタブモードでは `action_formatter.format_inference_result()` を経由した
整形済みレスポンスを返し、リアルモードでは `_infer_real()` で
`NotImplementedError` を上げて後続実装に明示的に委譲する。

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
from pathlib import Path
from typing import Any

from mortal.action_formatter import format_inference_result

logger = logging.getLogger("mortal.engine")


class ModelLoadError(RuntimeError):
    """Mortal モデルロード失敗を表す例外。

    ファイル不在 / state_dict のキー不整合 (= モデルバージョン不一致) /
    GPU OOM / vendor submodule 未取得 / torch 未インストール などを
    本例外にラップする。`__cause__` に元例外を保持する。
    """


# スタブモード用の合成 policy/value/q_values。`infer()` が
# `action_formatter.format_inference_result` を通すための入力で、
# - discard 4 種 + reach + pon の 6 アクションを含む (top-5 抽出と
#   action_type マッピングが網羅的に検証できる最小セット)
# - policy 合計はおおむね 1.0 に揃える (現実の softmax 出力に近い形)
_STUB_POLICY: dict[str, float] = {
    "discard:1m": 0.05,
    "discard:6m": 0.61,
    "discard:9p": 0.18,
    "discard:1z": 0.04,
    "reach": 0.10,
    "pon:5m5m": 0.02,
}
_STUB_Q_VALUES: dict[str, float] = {
    "discard:1m": -0.10,
    "discard:6m": 0.32,
    "discard:9p": 0.18,
    "discard:1z": -0.05,
    "reach": 0.40,
    "pon:5m5m": -0.20,
}
_STUB_VALUE: float = 0.32


def _stub_result() -> dict[str, Any]:
    """スタブモード時の整形済みレスポンスを返す。

    本実装も同じ `format_inference_result` を経由するため、
    shape は完全に互換となる。
    """
    return format_inference_result(
        policy=_STUB_POLICY,
        value=_STUB_VALUE,
        q_values=_STUB_Q_VALUES,
    )


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

        # `state` が想定外の型 (list / Tensor 等) で来たり、`version` 等が
        # 整数化できない値で保存されていた場合に AttributeError / ValueError
        # が漏れないよう ModelLoadError にラップする (PR #50 gemini review)。
        try:
            cfg = state.get("config") or {}
            control = cfg.get("control") or {}
            resnet = cfg.get("resnet") or {}
            version = int(control.get("version", 1))
            num_blocks = int(resnet.get("num_blocks", 40))
            conv_channels = int(resnet.get("conv_channels", 192))
        except (AttributeError, TypeError, ValueError) as exc:
            raise ModelLoadError(f"failed to parse model config: {exc}") from exc

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

        # VendorEngine の __init__ で引数不整合 / リソース不足が起きても
        # ModelLoadError にラップする (PR #50 gemini review)。
        try:
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
        except Exception as exc:
            raise ModelLoadError(f"failed to construct vendor MortalEngine: {exc}") from exc

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

    def infer(self, mjai_events: list[dict[str, Any]]) -> dict[str, Any]:
        """mjai event 列から推奨アクションを返す。

        スタブモードでは引数を無視して合成 policy を `action_formatter` に
        通した整形済みレスポンスを返す。実モデルが load 済みの場合は
        `_infer_real(mjai_events)` を呼ぶが、libriichi `mjai.Bot` への
        配線は Phase D5 (後続 issue) で実装するため現状は
        `NotImplementedError` を上げる。

        返値の shape は frontend `InferenceResult` 型 (`src/types/index.ts`)
        と互換: `recommended`, `candidates`, `primary_label`, `timestamp` を
        必ず含み、デバッグ用に raw `policy` / `value` / `q_values` も同梱する。
        """
        if not self._ready:
            raise RuntimeError("MortalEngine is not ready; call load() first")
        if self._stub:
            del mjai_events
            return _stub_result()
        return self._infer_real(mjai_events)

    def _infer_real(self, mjai_events: list[dict[str, Any]]) -> dict[str, Any]:
        """vendor `MortalEngine.react_batch` への配線 (Phase D5)。

        実装ステップ (後続 issue):
        1. libriichi の `mjai.Bot` を `oya`/`bakaze`/`kyoku` 等で初期化し、
           受け取った mjai_events を 1 件ずつ feed して内部状態を構築。
        2. Bot から `(obs, mask)` を取り出し `self._engine.react_batch` に渡す。
        3. 返ってきた logits / Q values から `policy` (softmax)、`q_values`、
           `value` を抽出。
        4. `action_formatter.format_inference_result()` に渡して整形して返す。

        現時点で配線できない理由: vendor submodule
        (`python/vendor/mortal/`) が CI 環境では未初期化のため、
        `react_batch` の正確な戻り値型を検証する手段がない。明示的に
        `NotImplementedError` を上げ、フォールバックで黙ってスタブを返す
        ことは避ける (本番でユーザに気付かれない劣化が起きるため)。
        """
        del mjai_events
        raise NotImplementedError(
            "Phase D5: vendor.mortal.engine.MortalEngine.react_batch wiring "
            "is not yet implemented; pass JANTAMA_STUB=1 for stub responses"
        )
