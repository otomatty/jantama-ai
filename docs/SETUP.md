# 開発環境セットアップ手順

このプロジェクトは Cowork 経由で雛形を作成済みです。
Windows 上で開発を始めるための手順を以下にまとめます。

## 0. 前提ツールのインストール

| ツール | 確認コマンド | 備考 |
|---|---|---|
| Node.js 20+ | `node -v` | https://nodejs.org/ |
| Rust 1.77+ | `rustc -V` | `rustup default stable` で stable をデフォルトに |
| Python 3.12+ | `py -V` | https://www.python.org/ |
| uv | `uv --version` | `pip install uv` または winget |
| git | `git -v` | Git for Windows |
| Microsoft C++ Build Tools | — | Tauri ビルドに必須 (https://v2.tauri.app/start/prerequisites/) |
| WebView2 ランタイム | — | Win11 は標準同梱 |

## 1. git リポジトリ初期化

PowerShell をプロジェクトルートで開いて:

```powershell
cd C:\Users\saedg\apps\jantama-ai
powershell -ExecutionPolicy Bypass -File scripts\init-git.ps1
```

または手動で:

```powershell
git init -b main
git add .
git commit -m "chore: initial project scaffold (Phase A)"
```

## 2. フロントエンド依存のインストール

```powershell
npm install
```

## 3. Python サブプロセス依存のインストール

```powershell
cd python
uv sync --extra dev
cd ..
```

## 4. 動作確認 (3 段階)

### 4-1. ブラウザだけで UI 確認

```powershell
npm run dev
```

→ <http://localhost:1420> を開く。Tauri RPC はスタブ値で動くため、UI レイアウトの確認に使える。

### 4-2. Tauri デスクトップアプリで起動

```powershell
npm run tauri:dev
```

→ 初回は Rust 依存のビルドに 5〜10 分かかる。完了すると Windows ネイティブのウィンドウで起動する。

### 4-3. Python 単体テスト

```powershell
cd python
uv run pytest
```

### 4-4. Python サブプロセスを手動で起動 (デバッグ)

```powershell
cd python

# 認識プロセス: stdin に JSON-lines を流すと結果が返る
uv run jantama-recognition
# 例: {"type": "ping", "id": 1} を入力 → {"type": "pong", "id": 1} が返る

# Mortal プロセス
uv run jantama-mortal --stub
# 例: {"type": "infer", "id": 1, "tenhou_json": {}} を入力
```

## 5. アイコンファイルの配置 (任意)

```powershell
# 1024x1024 のソース画像を用意して
npm run tauri icon path\to\source.png
```

→ `src-tauri/icons/` 配下に必要サイズが自動生成される。

## 6. 本番ビルド

```powershell
npm run tauri:build
```

→ `src-tauri/target/release/bundle/msi/` または `nsis/` に Windows インストーラが生成される。

> Python 側を同梱する場合は別途 PyInstaller でビルド後、Tauri の `bundle.resources` に追加する (Phase F)。

## 次のステップ

- Phase B: Tauri ↔ Python の JSON-lines 通信を `monitor.rs` に実装する。
- Phase C: 雀魂のスクショから手牌・河を OpenCV で認識する。
- Phase D: Mortal モデルを実際にロードして推論を実行する。

詳細は [`docs/PRD.md`](./PRD.md) §10 を参照。
