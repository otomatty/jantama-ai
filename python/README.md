# jantama-ai Python サイドプロセス

PRD §8.1 に基づき、認識プロセス (`recognition/`) と Mortal 推論プロセス (`mortal/`) を提供する。

## セットアップ

[uv](https://github.com/astral-sh/uv) を使用する。

```bash
cd python
uv sync
# 開発用 (ruff/black/pytest を含む)
uv sync --extra dev
# Mortal 推論用 (PyTorch を別途インストール)
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
# ROCm 7.2.1 の場合:
# uv pip install torch --index-url https://download.pytorch.org/whl/rocm7.2
```

## 通信仕様

Tauri (Rust) 側との通信は **stdin / stdout で JSON-lines** を使用する。
1 行 = 1 JSON オブジェクトで、終端に `\n` を必ず付ける。

### recognition プロセス

入力 (Rust → Python):
```json
{"type": "frame", "id": 123, "image_b64": "<base64-PNG>"}
```

出力 (Python → Rust):
```json
{"type": "result", "id": 123, "tenhou_json": { ... }, "confidence": 0.92}
```

### mortal プロセス

入力 (Rust → Python):
```json
{"type": "infer", "id": 123, "tenhou_json": { ... }}
```

出力 (Python → Rust):
```json
{
  "type": "result",
  "id": 123,
  "recommended": {"tile": "6m", "action_type": "discard", "expected_value": 0.32},
  "candidates": [
    {"tile": "6m", "action_type": "discard", "expected_value": 0.32},
    {"tile": "9p", "action_type": "discard", "expected_value": 0.18}
  ]
}
```

## 個別実行 (デバッグ用)

```bash
# 認識プロセスをスタンドアロン起動 (echo モード)
uv run jantama-recognition --echo

# Mortal プロセスをスタンドアロン起動 (スタブ応答)
uv run jantama-mortal --stub
```

## ビルド (PyInstaller でバンドル)

```bash
# Tauri バンドルに同梱する exe を生成
uv run pyinstaller --onefile --name jantama-recognition recognition/main.py
uv run pyinstaller --onefile --name jantama-mortal mortal/main.py
```
