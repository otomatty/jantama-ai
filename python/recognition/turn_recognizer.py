"""自分の手番検出 (issue #15)。

PRD §3.2 の「監視中・盤面なし ↔ 監視中・推奨表示中」の遷移トリガを
recognition 側で算出するための専用モジュール。

検出シグナル:

1. アクションボタン (`templates/actions/<key>.png`) のテンプレマッチ:
   雀魂卓上の右下に表示される「チー/ポン/カン/リーチ/ツモ/ロン/パス」ボタンの
   有無を 1 種類ずつ独立して検出する。位置が固定でない (使えるボタンだけが
   右詰めで現れる) ので `cv2.matchTemplate` の結果マップ全体の最大 NCC を
   見る (`tile_recognizer` のグリッド分割とは違う)。
2. 思考タイマー: 自家ネームプレート周りに出るタイマー色 (黄〜橙系) を
   HSV inRange の占有率で検出する。

どちらも、雀魂 UI のテンプレ実画像が揃わない (issue #16 系) 状態でも
graceful degrade するように、未配備時は「検出できなかった」扱いの戻り値
を返す。BoardRecognizer 側で `hand_count >= 14` の補助シグナルとマージする。
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from recognition.tile_recognizer import RoiRect

logger = logging.getLogger("recognition")

# 雀魂の「自分が選べる」アクション 7 種。`available_actions` の値域もここに揃え、
# `src/types/index.ts` の `ActionType` ユニオン (`discard` を加えた 8 種) のうち
# テンプレ検出で同定できるものだけをリストアップする。`discard` は「手牌が 14 牌」
# 派生で BoardRecognizer 側が付与するため、ここには含めない。
ACTION_KEYS: list[str] = ["chi", "pon", "kan", "riichi", "ron", "tsumo", "pass"]

# NCC (cv2.TM_CCOEFF_NORMED) の絶対しきい値。`WindRecognizer` の 0.5 は
# 「4 種のうち相対的にベスト + 絶対しきい値」の二段構えだったが、アクション
# ボタンは独立な binary 検出器 (それぞれ「出ているか / いないか」のみ) なので
# 「相対的にベスト」が効かない。誤検出率 <5% (受け入れ基準) を満たすには
# 絶対しきい値を高く設定する必要がある。雀魂のボタン画像は高コントラスト
# で固定形状なので、実マッチは 0.85+ になる想定。0.75 で fail-closed する。
ACTION_MATCH_THRESHOLD = 0.75

# 思考タイマーの色域 (HSV)。雀魂デフォルトスキンのタイマーリングは黄〜橙系。
# H は OpenCV 仕様で 0..179 の半周。スキンによってはタイマー色が変わるので、
# 将来 issue で設定可能化する余地を残してモジュールトップに括り出す。
_TIMER_HSV_LOWER = np.array([10, 100, 150], dtype=np.uint8)
_TIMER_HSV_UPPER = np.array([35, 255, 255], dtype=np.uint8)

# タイマー ROI のマスク占有率がこれ以上なら「タイマー出現中」と判定する。
# ROI を ネームプレート + 円形タイマーぴったりに切るとタイマー部分が 20%+ になる
# 想定。広めに切ると比率が下がるので保守的に 5% で判定する。
TIMER_OCCUPANCY_THRESHOLD = 0.05


def _crop_roi(bgr_frame: np.ndarray, roi: RoiRect) -> np.ndarray | None:
    """ROI 比率を実ピクセルに直して切り出し。サイズ 0 になったら `None`。

    `ocr_recognizer._crop_roi` と同形だが、モジュール循環を避けるためにここに
    複製を置く。
    """
    if bgr_frame is None or bgr_frame.size == 0:
        return None
    h, w = bgr_frame.shape[:2]
    x0 = max(0, min(w, int(roi.x * w)))
    y0 = max(0, min(h, int(roi.y * h)))
    x1 = max(0, min(w, int((roi.x + roi.w) * w)))
    y1 = max(0, min(h, int((roi.y + roi.h) * h)))
    if x1 <= x0 or y1 <= y0:
        return None
    return bgr_frame[y0:y1, x0:x1]


class TurnRecognizer:
    """アクションボタンと思考タイマーから自分の手番状態を推定する。

    `WindRecognizer` 同様、テンプレ群を 1 度だけロードして使い回す。テンプレが
    `ACTION_KEYS` の全種揃わない場合は fail-closed (= 検出を無効化)。
    """

    def __init__(self, template_dir: Path) -> None:
        self.template_dir = template_dir
        self._templates: dict[str, np.ndarray] = {}
        self._loaded = False
        self._warned_no_action_roi = False
        self._warned_no_timer_roi = False
        self._load()

    def _load(self) -> None:
        if not self.template_dir.is_dir():
            logger.warning(
                "action templates dir not found: %s (issue #16 で配置予定); "
                "ボタン検出を無効化して 14 牌目フォールバックのみで動作します",
                self.template_dir,
            )
            return

        loaded: dict[str, np.ndarray] = {}
        for key in ACTION_KEYS:
            path = self.template_dir / f"{key}.png"
            if not path.is_file():
                continue
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                logger.warning("failed to read action template: %s", path)
                continue
            loaded[key] = img

        # `tile_recognizer` / `wind_recognizer` と同様、partial set は fail-closed。
        # 「うちの 4 種しか揃わない」状態で動かすと、揃っているボタンしか検出
        # できず available_actions が常に欠落するため、UI 側の判断が中途半端に
        # 歪む。揃っていないことを警告ログで明示してから無効化する。
        if len(loaded) < len(ACTION_KEYS):
            missing = [k for k in ACTION_KEYS if k not in loaded]
            if missing:
                logger.warning(
                    "partial action template set in %s: missing %s; "
                    "ボタン検出を無効化して 14 牌目フォールバックのみで動作します",
                    self.template_dir,
                    ", ".join(missing),
                )
            return

        self._templates = loaded
        self._loaded = True
        logger.info("loaded %d action templates from %s", len(loaded), self.template_dir)

    def detect_buttons(
        self,
        bgr_frame: np.ndarray,
        action_roi: RoiRect | None,
    ) -> list[str]:
        """アクションボタン ROI から検出されたアクション名のリストを返す。

        位置はフレームごとに変わる (右詰めで使えるボタンだけ出る) ので、
        ROI 全体に対して `cv2.matchTemplate` をスライドさせて最大 NCC を見る。
        テンプレ未ロード or ROI 未指定なら空リスト。
        """
        if not self._loaded:
            return []
        if action_roi is None:
            if not self._warned_no_action_roi:
                logger.warning(
                    "action_buttons ROI not calibrated; button detection disabled. "
                    "Run ROI calibration to enable call/riichi/tsumo/ron detection."
                )
                self._warned_no_action_roi = True
            return []
        crop = _crop_roi(bgr_frame, action_roi)
        if crop is None:
            return []
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        detected: list[str] = []
        for key in ACTION_KEYS:
            tmpl = self._templates.get(key)
            if tmpl is None:
                continue
            th, tw = tmpl.shape[:2]
            # crop の方が小さいと matchTemplate がエラーを投げるので fail-safe。
            if gray.shape[0] < th or gray.shape[1] < tw:
                continue
            res = cv2.matchTemplate(gray, tmpl, cv2.TM_CCOEFF_NORMED)
            score = float(res.max())
            if score >= ACTION_MATCH_THRESHOLD:
                detected.append(key)
        return detected

    def detect_timer_active(
        self,
        bgr_frame: np.ndarray,
        timer_roi: RoiRect | None,
    ) -> bool:
        """思考タイマー ROI に黄〜橙色のリングが出ているかを判定する。

        ROI 未指定なら `False`。テンプレロード状態とは独立 (色判定なので)。
        """
        if timer_roi is None:
            if not self._warned_no_timer_roi:
                logger.warning(
                    "turn_timer ROI not calibrated; timer detection disabled. "
                    "Falls back to 14-tile hand count only for own-turn detection."
                )
                self._warned_no_timer_roi = True
            return False
        crop = _crop_roi(bgr_frame, timer_roi)
        if crop is None:
            return False
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, _TIMER_HSV_LOWER, _TIMER_HSV_UPPER)
        if mask.size == 0:
            return False
        occupancy = float(np.count_nonzero(mask)) / float(mask.size)
        return occupancy >= TIMER_OCCUPANCY_THRESHOLD
