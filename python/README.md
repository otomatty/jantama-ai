# jantama-ai Python サイドプロセス

PRD §8.1 に基づき、認識プロセス (`recognition/`) と Mortal 推論プロセス (`mortal/`) を提供する。

## セットアップ

[uv](https://github.com/astral-sh/uv) を使用する。

> 先にリポジトリルートで `git submodule update --init --recursive` を実行し、
> `python/vendor/mortal/` (Equim-chan/Mortal) を取得しておくこと。

```bash
cd python
uv sync
# 開発用 (ruff/black/pytest を含む)
uv sync --extra dev
# Mortal 推論用 (PyTorch / toml / tqdm / tensorboard)
uv sync --extra mortal
# PyTorch の wheel を環境に合わせ差し替える場合は別途インデックスを指定する。
# CPU 版:
# uv pip install torch --index-url https://download.pytorch.org/whl/cpu
# ROCm 7.2.1 の場合:
# uv pip install torch --index-url https://download.pytorch.org/whl/rocm7.2
```

## 上流 Mortal の取り込み

`python/vendor/mortal/` に [Equim-chan/Mortal](https://github.com/Equim-chan/Mortal)
を git submodule として配置している (issue #17 / PRD §10 Phase D)。

- ライセンスは **AGPL-3.0** (`python/vendor/mortal/LICENSE`)。
  AGPL-3.0 自体は (a) 著作権表示・ライセンス本文の同梱、(b) 改変時の変更箇所
  明示、(c) ネットワーク経由でユーザに提供する場合の対応ソース公開、といった
  条件下での再配布を許可しています。
  本リポジトリでは PRD §10 の方針に従い、Mortal のソース・学習済みモデル
  のいずれも成果物に同梱・再配布せず、利用者が submodule 経由で直接取得
  する運用とします (= プロジェクト方針としての非配布であり、ライセンス制限
  ではありません)。
- `vendor/mortal/mortal/` 配下を Python の namespace package として参照する。
  `mortal/__init__.py` および `mortal/main.py` で `python/` ディレクトリを
  `sys.path` に追加しているため、以下のように import できる:

  ```python
  from vendor.mortal.mortal.engine import MortalEngine
  ```

- `libriichi` (Rust 製の Python 拡張) は pip からは取得できないため、
  Phase D2 で Mortal リポジトリ側の Cargo.toml から build する。`extras = mortal`
  には現状含めていない。

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
