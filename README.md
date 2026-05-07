# 雀魂AIアシスタント (jantama-ai)

雀魂(Mahjong Soul)の対局画面をリアルタイムにキャプチャし、麻雀AI Mortal による推奨アクションと期待値を即座に表示するデスクトップアプリです。

> 詳細仕様は [`docs/PRD.md`](./docs/PRD.md) を参照してください。

## アーキテクチャ

```
Tauri Desktop App (Windows .exe)
├── Frontend: React + TypeScript + Vite + Tailwind v4 + shadcn/ui
├── Backend: Rust (xcap, tauri-plugin-sql, tauri-plugin-store)
└── Sub-processes (stdin/stdout JSON-lines)
    ├── recognition: Python + OpenCV (画像 → 天鳳JSON)
    └── mortal: Python + PyTorch (天鳳JSON → 推奨候補)
```

## ディレクトリ構成

```
jantama-ai/
├── src/                  # フロントエンド (React + TS)
│   ├── components/ui/    # shadcn/ui コンポーネント
│   ├── screens/          # MainScreen / SettingsScreen
│   ├── state/            # useReducer ベースの状態管理
│   ├── lib/              # Tauri RPC ラッパー
│   ├── types/            # 共通型定義
│   └── hooks/
├── src-tauri/            # Tauri (Rust)
│   ├── src/              # main.rs, lib.rs, capture.rs, monitor.rs, ...
│   ├── migrations/       # SQLite マイグレーション
│   ├── capabilities/     # Tauri 2.x 権限定義
│   └── icons/
├── python/               # Python サブプロセス
│   ├── recognition/      # 画像 → 天鳳JSON
│   ├── mortal/           # 天鳳JSON → 推奨候補
│   ├── common/           # 共通: JSON-lines I/O, ロギング
│   └── tests/
├── docs/
│   └── PRD.md
└── scripts/
```

## 必要環境

- Windows 11
- Node.js 20+
- Rust 1.77+ (`rustup default stable`)
- Python 3.12+
- [uv](https://github.com/astral-sh/uv) (Python パッケージ管理)
- Tauri 2.x prerequisites: <https://v2.tauri.app/start/prerequisites/>

## 初回セットアップ

```bash
# 1. フロント依存
npm install

# 2. Python 依存 (uv)
cd python
uv sync --extra dev
cd ..

# 3. Mortal モデルを別途取得 (PRD §10 Phase B 以降で必要)
#    https://github.com/Equim-chan/Mortal から学習済みモデルを入手して
#    任意の場所に配置し、設定画面でパスを指定する。
```

## 開発時の起動

```bash
# Tauri 開発モード (Vite + Rust が同時に起動する)
npm run tauri:dev

# フロントだけブラウザで動作確認したい場合
npm run dev   # → http://localhost:1420
```

ブラウザ単体起動時は Tauri RPC がスタブ値を返すため、UI レイアウトの確認に使えます。

## ビルド

```bash
# 開発ビルド (.exe 生成、未署名)
npm run tauri:build

# Python 側を PyInstaller で同梱する手順は python/README.md 参照
```

## Phase 別実装状況 (PRD §10)

| Phase | 内容 | 状況 |
|---|---|---|
| A | Tauri + React + TS 雛形、画面骨組み | ✅ 完了 |
| B | Python 連携 (JSON-lines, モデルパス指定) | スケルトンのみ |
| C | 雀魂スクショから手牌/河を認識 → 天鳳JSON | 未着手 |
| D | Mortal 推論結果の UI 表示・状態管理 | 未着手 |
| E | SQLite 永続化、自動削除ジョブ | テーブル定義のみ |
| F | PyInstaller 同梱、インストーラ生成 | 未着手 |
| G | LLM 連携 (推奨理由) | Should |

## ライセンス・注意事項

- 個人利用に限ります。Mortal モデルは配布しません。
- 雀魂のメモリ・通信は一切読まず、画面キャプチャのみで動作します。
- 自動操作 (ボット化) は行いません。
