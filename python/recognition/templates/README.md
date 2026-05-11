# 牌・風・アクション テンプレート画像 (issue #16)

雀魂卓上の OpenCV テンプレートマッチング系認識器 (`tile_recognizer` /
`river_recognizer` / `melds_recognizer` / `wind_recognizer` /
`turn_recognizer`) がこのディレクトリから 48 種のテンプレを読み込む。

すべて未配置の状態でも各 recognizer は警告ログを 1 回出すだけで落ちず、
fail-closed (`[]` / `None` を返し、上位 `BoardRecognizer` でスタブ値に
フォールバック) で動作する。本リポジトリは issue #16 完了までその状態。

## 進捗チェックリスト

### 牌テンプレ (`*.png`) — 37 枚

`tile_recognizer.TILE_CODES` の順序と一致。`0m/0p/0s` は赤 5 (Mortal/天鳳
慣例)。

萬子:
- [ ] `1m.png`
- [ ] `2m.png`
- [ ] `3m.png`
- [ ] `4m.png`
- [ ] `5m.png`
- [ ] `6m.png`
- [ ] `7m.png`
- [ ] `8m.png`
- [ ] `9m.png`
- [ ] `0m.png` (赤 5m)

筒子:
- [ ] `1p.png`
- [ ] `2p.png`
- [ ] `3p.png`
- [ ] `4p.png`
- [ ] `5p.png`
- [ ] `6p.png`
- [ ] `7p.png`
- [ ] `8p.png`
- [ ] `9p.png`
- [ ] `0p.png` (赤 5p)

索子:
- [ ] `1s.png`
- [ ] `2s.png`
- [ ] `3s.png`
- [ ] `4s.png`
- [ ] `5s.png`
- [ ] `6s.png`
- [ ] `7s.png`
- [ ] `8s.png`
- [ ] `9s.png`
- [ ] `0s.png` (赤 5s)

字牌 (1z=東, 2z=南, 3z=西, 4z=北, 5z=白, 6z=發, 7z=中):
- [ ] `1z.png`
- [ ] `2z.png`
- [ ] `3z.png`
- [ ] `4z.png`
- [ ] `5z.png`
- [ ] `6z.png`
- [ ] `7z.png`

### 風ラベルテンプレ (`winds/*.png`) — 4 枚

卓上の自風 / 場風ラベル。手牌 1z..4z の牌絵柄とは別系統 (フォント / 装飾が
異なる) のため `wind_recognizer` 専用。

- [ ] `winds/east.png` (東)
- [ ] `winds/south.png` (南)
- [ ] `winds/west.png` (西)
- [ ] `winds/north.png` (北)

### アクションボタンテンプレ (`actions/*.png`) — 7 枚

卓上右下に表示される鳴き / リーチ / 和了ボタン。`turn_recognizer` がボタン
本体のアイコン部分 (テキスト + 縁取りの短冊形) に対し `cv2.matchTemplate`
を画面右下領域でスライド走査する。

- [ ] `actions/chi.png` (チー)
- [ ] `actions/pon.png` (ポン)
- [ ] `actions/kan.png` (カン — 大明槓 / 加槓 / 暗槓 共通)
- [ ] `actions/riichi.png` (リーチ)
- [ ] `actions/tsumo.png` (ツモ)
- [ ] `actions/ron.png` (ロン)
- [ ] `actions/pass.png` (スキップ / パス)

## 形式・サイズ

- すべて 8bit グレースケール PNG を推奨 (カラー保存でも `cv2.IMREAD_GRAYSCALE`
  で読まれるが、ファイルサイズ削減のため)。
- 牌テンプレは全 37 枚で同一サイズが望ましい (`load_tile_templates` が
  最初に読めたテンプレのサイズに `cv2.resize` で揃える)。
- 推奨サイズ: 手牌 1 牌の標準スケール基準で **64×96 程度**。
- 風 / アクションは各カテゴリ内でサイズ統一すれば OK (`wind_recognizer`,
  `turn_recognizer` がカテゴリ単位で同一サイズ前提)。

## 横向き / 複数解像度の PNG は不要 (設計判断)

issue #16 の本文では「横向き版を 90 度回転で生成」「縮小サイズも複数解像度
生成」を要求していたが、本リポジトリでは以下の理由で **別ファイルを置かない**
方針を採用した。

| 観点 | 実装 | 該当箇所 |
| --- | --- | --- |
| 河のリーチ宣言牌 / 副露の横向き牌 | `cv2.rotate` で起動時に CW/CCW 両方向を内部生成 | `river_recognizer._load` / `melds_recognizer._load` |
| 手牌大 / 河小 などの解像度差 | `cv2.resize` でセル毎にテンプレサイズへフィット | `tile_recognizer.fit_to_template_size` ほか |

このため、ユーザが用意するのは **縦向き 1 解像度の 48 枚のみ** で全認識器が
動作する。

## 収集ワークフロー

1. 雀魂を起動し、目的の牌が手牌 / 河 / 副露に出ているフレームのスクショを
   集める。鳴き / リーチ / ロンなど一部アクションは特定局面でしか出ないので
   複数局のスクショを跨ぐ。
2. `recognition.tools.extract_template` CLI で各スクショから牌領域をピクセル
   矩形 (`x y w h`) で指定して切り出す:

   ```bash
   uv run python -m recognition.tools.extract_template \
       /path/to/screenshot.png 1m 540 940 60 90 \
       --out python/recognition/templates --size 64x96
   ```

   - 第 2 引数は `1m`〜`7z` / `winds/east` / `actions/chi` のいずれかの牌コード。
   - サブカテゴリは `winds/east` のように `/` 区切りで指定すれば
     `templates/winds/east.png` に保存される。
   - `--size 64x96` を指定するとグレースケール化後にリサイズして保存する。
     省略時はクロップしたサイズのまま保存。
3. このチェックリストの該当行を `- [x]` に書き換えてコミット。
4. 配置確認 (37 枚 + 4 枚 + 7 枚揃ったら fail-closed が解除される):

   ```bash
   uv run pytest tests/test_smoke.py -q
   uv run python -m recognition.main --help    # ロード警告が消えるはず
   ```

## fail-closed 挙動 (現状)

| recognizer | 揃わない枚数 | 動作 |
| --- | --- | --- |
| `tile` / `river` / `melds` | 37 種未満 | `recognize_*` が `[]` / `0.0` を返す + 起動時 warning |
| `wind` | 4 種未満 | `recognize` が `(None, 0.0)` を返す + 起動時 warning |
| `turn` (アクション部) | 7 種未満 | ボタン検出は無効化、手牌枚数のみで「自分の手番」を判定 |

部分セットだと「該当しない 1 種が必ず誤分類される」NCC の性質上、半端な
配置をするとむしろ精度が下がるため fail-closed をデフォルトにしている。

## Tesseract OCR (issue #12) — 参考

局名 (`round_label`) / 点棒 (`scores`) / 巡目 (`turn`) は `pytesseract` 経由で
Tesseract OCR を呼ぶため、ここのテンプレ画像とは別系統。Tesseract バイナリ
本体は別途インストールが必要。

- Windows: <https://github.com/UB-Mannheim/tesseract/wiki> から installer を
  入れ、`tesseract.exe` を PATH に追加。`jpn` 言語パックも一緒に入れる。
- Tesseract 不在時は `ocr_recognizer` が警告ログを 1 度出し、該当フィールドは
  既定値 (`stub_tenhou_json()` 由来) を使う。
