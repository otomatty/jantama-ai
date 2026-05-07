# アイコンファイル配置場所

`tauri.conf.json` の `bundle.icon` で参照されているアイコンを以下のファイル名で配置してください。
初回ビルド時には、Tauri CLI の `npm run tauri icon <path/to/source.png>` コマンドで一括生成可能です。

必要ファイル:
- `32x32.png`
- `128x128.png`
- `128x128@2x.png`
- `icon.icns` (macOS用、Windows ビルド時は不要)
- `icon.ico` (Windows用)

例:
```bash
# 1024x1024 のソース画像から各サイズを自動生成する
npm run tauri icon ../docs/source-icon.png
```
