"""Mortal 推論プロセスのエントリーポイント。

stdin から `{"type": "infer", "id": <int>, "tenhou_json": {...}}` を受け、
stdout に `{"type": "result", "id": <int>, "recommended": {...}, "candidates": [...]}` を返す。

- `--model <path>`: Mortal の `.pth` を `torch.load` で読み込む
- `--backend rocm|cpu`: 推論デバイス (デフォルト cpu)。`rocm` 指定時は
  PyTorch から見て `cuda` デバイスを使う (ROCm は内部的に CUDA API 互換)
- 環境変数 `JANTAMA_STUB=1`: モデル読込をスキップしスタブ応答を返す
  (後方互換: 旧 `--stub` フラグの代替)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

# `python mortal/main.py` 直叩きでも vendor.mortal.mortal.* を import できるよう
# python/ を sys.path に追加してから common / 上流 Mortal をロードする。
# `uv run jantama-mortal` 経由なら mortal/__init__.py が同様の調整を行う。
_PYTHON_ROOT = Path(__file__).resolve().parent.parent
if str(_PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(_PYTHON_ROOT))

from common import read_request, setup_stderr_logging, write_response  # noqa: E402
from mortal.mortal_engine import ModelLoadError, MortalEngine  # noqa: E402
from mortal.snapshot_to_mjai import SnapshotToMjaiConverter  # noqa: E402

logger = setup_stderr_logging("mortal")

# mortal プロセス寿命 = recognition セッション寿命なので、converter は 1 インスタンスを
# プロセススコープで保持する。snapshot_to_mjai.py のドキュメンテーション通り、
# 連続するスナップショットの差分を取るために状態 (前フレーム手牌・河長さ等) を保持する。
_converter = SnapshotToMjaiConverter()


def handle_infer(
    engine: MortalEngine,
    req: dict[str, Any],
    converter: SnapshotToMjaiConverter | None = None,
) -> dict[str, Any]:
    """`infer` リクエストを処理する。

    1. `tenhou_json` (recognition プロセスが emit した盤面スナップショット)
       を `SnapshotToMjaiConverter` に渡し、前フレからの差分 mjai event 列を得る。
    2. `MortalEngine.infer(mjai_events)` で整形済みレスポンスを得る。
    3. `{"type": "result", "id": ...}` を付加して返す。

    推論中の例外 (Phase D5 未実装の `NotImplementedError` や、変換失敗時の
    予期せぬエラー等) はプロセスごと落とさず、プロトコル準拠の `error` 応答に
    変換して返す (CodeRabbit on PR #52)。Rust monitor は `ProcessDied` 経由で
    無限再起動ループに陥らないよう、`type=="error"` を観測したらエラー表示に
    倒すだけで本プロセスは生かしておく契約。

    `converter` 引数はテスト時に独立インスタンスを差し込めるよう公開する。
    本番呼び出しではモジュールスコープの `_converter` がデフォルト。
    """
    req_id = req.get("id", -1)
    tenhou = req.get("tenhou_json", {})
    conv = converter if converter is not None else _converter
    try:
        mjai_events = conv.convert(tenhou)
        result = engine.infer(mjai_events)
    except NotImplementedError as exc:
        # Phase D5 (vendor.react_batch 配線) 未実装で発生するケース。
        # 利用者には「モデル未対応」と分かるメッセージを返す。
        logger.warning("infer not implemented: %s", exc)
        return {"type": "error", "id": req_id, "message": str(exc)}
    except Exception as exc:  # noqa: BLE001 — プロセス生存を最優先
        # 変換 / 推論で予期せぬ例外が起きた場合も、プロセスは生かして次の
        # リクエストに備える。詳細はログで追えるよう exc_info を出す。
        logger.exception("infer failed: %s", exc)
        return {"type": "error", "id": req_id, "message": f"infer failed: {exc}"}
    return {"type": "result", "id": req_id, **result}


def _build_engine(args: argparse.Namespace) -> MortalEngine:
    """argparse 引数 / 環境変数から `MortalEngine` を構築する。"""
    if os.environ.get("JANTAMA_STUB") == "1":
        logger.info("JANTAMA_STUB=1 detected; running in stub mode")
        return MortalEngine.stub()

    if not args.model:
        logger.error("--model is required (or set JANTAMA_STUB=1 to run without a model)")
        raise SystemExit(2)

    try:
        return MortalEngine.from_pretrained(args.model, backend=args.backend)
    except ModelLoadError as exc:
        logger.error("failed to load Mortal model: %s", exc)
        raise SystemExit(1) from exc


def main() -> int:
    parser = argparse.ArgumentParser(description="雀魂AIアシスタント Mortal 推論プロセス")
    parser.add_argument("--model", type=str, default=None, help="Mortal モデルファイル (.pth)")
    parser.add_argument(
        "--backend",
        choices=["rocm", "cpu"],
        default="cpu",
        help="推論バックエンド (rocm=ROCm/CUDA, cpu)",
    )
    args = parser.parse_args()

    logger.info("mortal process started (model=%s, backend=%s)", args.model, args.backend)

    engine = _build_engine(args)
    logger.info("MortalEngine ready=%s name=%s", engine.is_ready(), engine.name)

    try:
        for req in read_request():
            req_type = req.get("type")
            if req_type == "infer":
                write_response(handle_infer(engine, req))
            elif req_type == "ping":
                write_response({"type": "pong", "id": req.get("id")})
            else:
                write_response(
                    {
                        "type": "error",
                        "id": req.get("id"),
                        "message": f"unknown type: {req_type}",
                    }
                )
    except KeyboardInterrupt:
        logger.info("mortal process interrupted")

    logger.info("mortal process exiting")
    return 0


if __name__ == "__main__":
    sys.exit(main())
