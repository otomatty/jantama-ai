"""盤面情報 (手牌 / ドラ / 自風 / 場風 / 局 / 巡目 / 点棒) を 1 フレームから
まとめて認識する (issue #12)。

設計方針:
- 各サブ認識器の例外は個別に握り、1 項目の失敗が他項目を巻き込まない
- 認識できなかったフィールドは「Rust 側 build_board_summary がスキーマを通る
  安全な既定値」を使う。フィールド単位の `None` を tenhou_json に出すと
  build_board_summary が GameBoardSummary 全体を None に倒す (= UI が
  「盤面なし」表示) ため、必須フィールドはダミーでも埋める
- ROI 未指定 / Tesseract 不在 / テンプレ未配置の各 graceful degrade は
  サブ認識器側で実装済み。BoardRecognizer は結果を集約するだけ。
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any

import numpy as np

from recognition import ocr_recognizer
from recognition.tile_recognizer import RoiRect, TileRecognizer
from recognition.turn_recognizer import TurnRecognizer
from recognition.wind_recognizer import WindRecognizer

logger = logging.getLogger("recognition")


# tenhou_json の「フィールドが認識できなかったとき」の既定値。
# Rust 側 `build_board_summary` の必須フィールドを満たすために残す。
# issue #15: `my_turn` / `available_actions` は手番検出の出力。検出シグナルが
# 全て無効な場合は「手番でない」(= 推論スキップ) 側に倒すため、既定値は
# `False` / `[]`。
DEFAULT_TENHOU_JSON: dict[str, Any] = {
    "hand": [],
    "river": [],
    "dora_indicators": ["5p"],
    "self_wind": "東",
    "round_wind": "東",
    "turn": 1,
    "scores": [25000, 25000, 25000, 25000],
    "melds": [],
    "my_turn": False,
    "available_actions": [],
}


# 「打牌アクション」と「鳴き系アクション」の集合。手牌が 14 枚のフレーム (=
# 自分のツモ番) では rinshan/kakan/riichi/tsumo を併せて出せる可能性があるが、
# `chi/pon/ron/pass` は他家からの打牌に対するレスポンスなので 13 枚側で出る。
# `_derive_turn_state` で組み合わせをフィルタするための定数。
# 「自分のツモ番に上乗せ可能なアクション」(打牌に加えてリーチ宣言 / 自摸和 /
# 暗槓・加槓を選べる)。`_derive_turn_state` の出力順序もこの並びに合わせる。
_DISCARD_TURN_EXTRA_ACTIONS_ORDERED: tuple[str, ...] = ("riichi", "tsumo", "kan")
# 「他家打牌へのレスポンス」として hand_count<14 のフレームで現れるボタン群。
# kan は大明槓があり得るのでここに含める (= call 系として扱う)。出力順序も同じ。
_CALL_TURN_ACTIONS_ORDERED: tuple[str, ...] = ("chi", "pon", "kan", "ron", "pass")
_CALL_ONLY_BUTTONS: frozenset[str] = frozenset({"chi", "pon", "ron", "pass"})
# 「自分のツモ番でのみ出るボタン」(14 牌フォールバックが取りこぼした場合の
# エッジケース対策)。kan は call 経路と重複させるため含めない。
_DISCARD_ONLY_BUTTONS: frozenset[str] = frozenset({"riichi", "tsumo"})

# `hand_count >= 14` または「タイマー active のみ」を根拠に `my_turn=True` に
# 切り替える前に必要な連続フレーム数 (= 2 で「2 フレーム連続で真なら flip」)。
# ボタン検出は即時反映するため、これは hand/timer 経路だけが対象。雀魂のツモ
# アニメーション中に hand_recognizer が 14 牌目を一瞬だけ拾うケースを抑制する
# (受け入れ基準: 誤検出率 <5%)。
_HAND_TIMER_DEBOUNCE_FRAMES = 2


def _roi(calib: dict[str, Any], key: str) -> RoiRect | None:
    """`roi_calibration` 辞書から RoiRect を取り出す。型不一致は `None`。"""
    if not isinstance(calib, dict):
        return None
    return RoiRect.from_dict(calib.get(key))


class BoardRecognizer:
    """各サブ認識器を保持して 1 フレームから tenhou_json を組み立てる。"""

    def __init__(self, template_dir: Path) -> None:
        self.template_dir = template_dir
        self.tile_recognizer = TileRecognizer(template_dir)
        # winds テンプレは `templates/winds/` 配下に置く規約 (templates/README.md)。
        self.wind_recognizer = WindRecognizer(template_dir / "winds")
        # actions テンプレは `templates/actions/` 配下 (issue #15)。
        self.turn_recognizer = TurnRecognizer(template_dir / "actions")
        # 「scores OCR は通ったが self_wind が未認識」状態の警告は対局ごとに 1 度。
        # 毎フレーム出すと監視ループのログを埋め尽くす。
        self._warned_scores_without_wind = False
        # hand_count>=14 / timer のみが根拠の `my_turn=True` を debounce するため、
        # 連続成立フレーム数を保持する。ボタン検出時は即時 True で、このカウンタは
        # リセット不要 (= 次フレームで hand/timer 経路に落ちたら 0 から数え直し)。
        self._my_turn_streak = 0

    def recognize(
        self,
        bgr_frame: np.ndarray,
        roi_calibration: dict[str, Any] | None,
    ) -> tuple[dict[str, Any], float]:
        """1 フレームを認識して `(tenhou_json, confidence)` を返す。

        `confidence` は手牌 + ドラの最小値 (テンプレマッチの NCC スコア最小)。
        OCR / wind は単一マッチなので confidence 計算には混ぜず、内部ログで
        出すに留める。0.0 は「何も認識できなかった」を意味する。
        """
        # deepcopy: 浅いコピーだと dora_indicators 等のリストが DEFAULT 共有のまま残り、
        # 将来呼び出し側が `tenhou["scores"].append(...)` 等で in-place 変更すると
        # 既定値が汚染される (CodeRabbit nit on PR #44)。フレームあたり 1 dict のコピーは
        # 計測誤差レベル。
        tenhou: dict[str, Any] = copy.deepcopy(DEFAULT_TENHOU_JSON)
        confidence = 0.0

        if bgr_frame is None or bgr_frame.size == 0:
            return tenhou, confidence

        calib = roi_calibration if isinstance(roi_calibration, dict) else {}

        # ----- 手牌 (issue #11) -----
        try:
            hand_tiles, hand_conf = self.tile_recognizer.recognize_hand(
                bgr_frame, _roi(calib, "hand")
            )
            if hand_tiles:
                tenhou["hand"] = hand_tiles
                confidence = hand_conf
        except Exception:  # noqa: BLE001
            logger.warning("hand recognition failed", exc_info=True)

        # ----- ドラ表示牌 (issue #12) -----
        try:
            dora_tiles, dora_conf = self.tile_recognizer.recognize_dora(
                bgr_frame, _roi(calib, "doras")
            )
            if dora_tiles:
                tenhou["dora_indicators"] = dora_tiles
                # confidence は手牌・ドラのうち低い方を採用 (= フレーム全体の最低品質)。
                confidence = min(confidence, dora_conf) if confidence > 0 else dora_conf
        except Exception:  # noqa: BLE001
            logger.warning("dora recognition failed", exc_info=True)

        # ----- 自風 (issue #12) -----
        self_wind_recognized = False
        try:
            wind_label, _wind_conf = self.wind_recognizer.recognize(
                bgr_frame, _roi(calib, "self_wind")
            )
            if wind_label is not None:
                tenhou["self_wind"] = wind_label
                self_wind_recognized = True
        except Exception:  # noqa: BLE001
            logger.warning("self_wind recognition failed", exc_info=True)

        # ----- 局名 + 場風 (issue #12) -----
        round_label: str | None = None
        try:
            round_label = ocr_recognizer.recognize_round_label(bgr_frame, _roi(calib, "round_info"))
        except Exception:  # noqa: BLE001
            logger.warning("round_label recognition failed", exc_info=True)
        if round_label is not None:
            tenhou["round_label"] = round_label
            round_wind = ocr_recognizer.round_label_to_wind(round_label)
            if round_wind is not None:
                tenhou["round_wind"] = round_wind

        # ----- 巡目 (issue #12) -----
        try:
            turn = ocr_recognizer.recognize_turn(bgr_frame, _roi(calib, "turn_counter"))
            if turn is not None:
                tenhou["turn"] = turn
        except Exception:  # noqa: BLE001
            logger.warning("turn recognition failed", exc_info=True)

        # ----- 点棒 (issue #12) -----
        # Codex P1 on PR #44: scores 配列は座順 (東→南→西→北) なので、self_wind が
        # 未認識のまま既定 "東" だと、非起家局面で他家の点数が「自分の持ち点」
        # として UI に流れる (build_board_summary が scores[self_wind_index] を引く)。
        # self_wind の実認識が成功したフレームのみ scores を採用し、それ以外は
        # 既定値 (全 25000) のままにしてフロントの「持ち点表示は信頼できない」
        # フォールバックに倒す。
        try:
            scores = ocr_recognizer.recognize_scores(bgr_frame, _roi(calib, "scores"))
            if scores is not None:
                if self_wind_recognized:
                    tenhou["scores"] = scores
                elif not self._warned_scores_without_wind:
                    logger.warning(
                        "scores OCR succeeded but self_wind is not recognized; "
                        "dropping scores to avoid wrong-seat score lookup. "
                        "Install wind templates (issue #16) to enable score reporting."
                    )
                    self._warned_scores_without_wind = True
        except Exception:  # noqa: BLE001
            logger.warning("scores recognition failed", exc_info=True)

        # ----- 自分の手番検出 (issue #15) -----
        # ボタンテンプレと思考タイマー色判定、加えて手牌が 14 牌である事実を
        # 組み合わせて my_turn / available_actions を決める。検出失敗時は
        # 「手番でない」側に倒し、Rust 側で mortal をスキップする。
        try:
            buttons = self.turn_recognizer.detect_buttons(bgr_frame, _roi(calib, "action_buttons"))
            timer_active = self.turn_recognizer.detect_timer_active(
                bgr_frame, _roi(calib, "turn_timer")
            )
            hand_count = len(tenhou["hand"])
            my_turn, actions = self._derive_turn_state(hand_count, buttons, timer_active)
            tenhou["my_turn"] = my_turn
            tenhou["available_actions"] = actions
        except Exception:  # noqa: BLE001
            logger.warning("turn recognition failed", exc_info=True)

        return tenhou, confidence

    def _derive_turn_state(
        self,
        hand_count: int,
        buttons: list[str],
        timer_active: bool,
    ) -> tuple[bool, list[str]]:
        """検出シグナルから (my_turn, available_actions) を算出する。

        判定優先度:
        1. call 系ボタン (chi/pon/ron/pass 等、他家打牌レスポンス) は hand_count
           より優先。hand 認識が一時的に 14 牌を誤検出していてもボタンを
           取りこぼさない (CodeRabbit major on PR #47)。
        2. hand_count >= 14 (debounce 適用): 自分のツモ番。打牌可能 +
           ボタンで上乗せ可能なアクション (`riichi/tsumo/kan`) をマージする。
        3. discard 系ボタンのみ (riichi/tsumo): 14 牌目を取りこぼしている
           エッジケース (ROI ずれ等)。即時 my_turn=True に倒す。
        4. timer_active のみ (debounce 適用): 上記いずれも検知できないが
           タイマーが出ているという保守的なフォールバック。`["discard"]` に倒す。
        5. 何も無し → `(False, [])`。

        ボタン検出経路はいずれも debounce を経ない (= 即時反映)。
        hand_count / timer のみが根拠の場合だけ `_my_turn_streak` をカウントする。
        """
        button_set = set(buttons)
        # call 系・discard 系の判定は kan も含めた全体集合で行う。
        has_call_button = bool(button_set & _CALL_ONLY_BUTTONS)

        # 1. call 系ボタンは hand_count より優先 (即時反映)
        if has_call_button:
            self._my_turn_streak = 0
            # 出力順序は ACTION_KEYS 順 (chi → pon → kan → ron → pass)。
            # kan が混在していたら call 系として一緒に出す (大明槓ケース)。
            actions = [k for k in _CALL_TURN_ACTIONS_ORDERED if k in button_set]
            return True, actions

        # 2. ツモ番 (14 牌): hand_count をベースに ["discard"] + 上乗せ可能ボタン
        if hand_count >= 14:
            self._my_turn_streak += 1
            if self._my_turn_streak < _HAND_TIMER_DEBOUNCE_FRAMES and not buttons:
                # 1 フレーム目はまだ flip しない。ただしボタンが出ている場合は
                # ボタン経由で即時反映する (debounce 不要)。
                return False, []
            # 順序を安定させるため定数の並びに合わせる。
            extras = [k for k in _DISCARD_TURN_EXTRA_ACTIONS_ORDERED if k in button_set]
            return True, ["discard", *extras]

        # 3. riichi/tsumo ボタンのみ (hand_count を 14 牌として検出し損ねたケース)。
        # 仕様上「ボタン検出は即時反映」なので my_turn=True に倒し、ボタン情報を
        # そのまま返す。`tsumo` 単独であっても和了可能タイミングなので推奨を出す。
        if button_set & _DISCARD_ONLY_BUTTONS:
            self._my_turn_streak = 0
            actions = [k for k in _DISCARD_TURN_EXTRA_ACTIONS_ORDERED if k in button_set]
            return True, actions

        # 4. タイマーだけが根拠 (debounce 適用)
        if timer_active:
            self._my_turn_streak += 1
            if self._my_turn_streak < _HAND_TIMER_DEBOUNCE_FRAMES:
                return False, []
            return True, ["discard"]

        # 5. 全シグナル空 → 手番でない
        self._my_turn_streak = 0
        return False, []
