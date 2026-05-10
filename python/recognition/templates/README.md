# 牌テンプレート画像 (issue #16)

`tile_recognizer.TileRecognizer` がここから 37 種の牌画像をロードする。

## 命名規則

ファイル名は牌コード (Mortal/天鳳慣例) + `.png`。

```
1m.png 2m.png 3m.png 4m.png 5m.png 6m.png 7m.png 8m.png 9m.png 0m.png   萬子 (0m = 赤5m)
1p.png 2p.png 3p.png 4p.png 5p.png 6p.png 7p.png 8p.png 9p.png 0p.png   筒子 (0p = 赤5p)
1s.png 2s.png 3s.png 4s.png 5s.png 6s.png 7s.png 8s.png 9s.png 0s.png   索子 (0s = 赤5s)
1z.png 2z.png 3z.png 4z.png 5z.png 6z.png 7z.png                         字牌 (東南西北白發中)
```

## 形式

- グレースケールでロードされる (`cv2.IMREAD_GRAYSCALE`)。カラー保存でも問題ないがファイルサイズ削減のため 8bit グレースケールを推奨。
- 全牌で同一サイズが望ましい (異なる場合は最初に読めた牌のサイズに `cv2.resize` で揃える)。
- 推奨サイズ: 雀魂手牌 1 牌の標準スケールを参考に 64×96 程度。

## 状態

issue #16 (牌画像テンプレートデータの収集と整備) で配置される予定。
それまでは空 (この `README.md` と `.gitkeep` のみ) で、`TileRecognizer` は
警告ログを 1 回出した上で `recognize_hand` が常に `([], 0.0)` を返す。
