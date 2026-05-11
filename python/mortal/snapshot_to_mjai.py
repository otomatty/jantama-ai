"""tenhou_json スナップショット → mjai event 列 変換層 (issue #19, Phase D3)。

Mortal の入力は **mjai 形式の event 列** (過去から現在までの局の流れ) だが、
recognition プロセスはスクショ 1 枚から「現在の盤面のスナップショット」しか
取れない。本モジュールはこのギャップを埋め、`SnapshotToMjaiConverter` が
連続するスナップショットの差分を見て mjai event を生成する。

設計方針:
- 監視ループ側で本クラスのインスタンスを 1 つ保持し、毎フレームの
  snapshot を `convert()` に渡す。差分から増えた dahai/tsumo/meld/reach
  を event 列として返す
- 局またぎは `round_label` (例: "東1局") 変化、または手牌総入れ替えで検出し
  `start_kyoku` を発行する
- 不整合 (前フレ→現フレで矛盾、ハンド枚数が想定外、等) は警告ログを出して
  「現在のスナップショットだけで擬似 event を生成」するフォールバックに倒す
- Phase D4 で本変換結果を libriichi `mjai.Bot` へ食わせて Mortal を動かす

pai 表記:
- 入力 (tenhou_json): "0m"/"0p"/"0s" を赤 5、それ以外は "1m" 等
- 出力 (mjai): "5mr"/"5pr"/"5sr" を赤 5 (libriichi/Mortal 互換)
- 字牌は 1z..7z (1=東, 2=南, 3=西, 4=北, 5=白, 6=發, 7=中)

座順:
- スナップショット内の `player` フィールドは天鳳座順 (= mjai 座順):
  0=自家, 1=下家(shimocha), 2=対面(toimen), 3=上家(kamicha)
- 副露 (meld) の `from` は鳴いた本人から見た相対座順:
  0=自家(暗槓のみ), 1=下家, 2=対面, 3=上家
  → mjai の絶対 target = (meld.player + meld.from) % 4
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any

logger = logging.getLogger("mortal.snapshot_to_mjai")


# tenhou JSON の自風表記 → mjai bakaze。
_WIND_JP_TO_MJAI: dict[str, str] = {
    "東": "E",
    "南": "S",
    "西": "W",
    "北": "N",
    # 念のため英字直入力も受ける (BoardRecognizer 既定値はカナ漢字)。
    "E": "E",
    "S": "S",
    "W": "W",
    "N": "N",
}

# 自風 → 0-3 の座席 index (E=0, S=1, W=2, N=3)。oya 算出に使う。
_WIND_TO_INDEX: dict[str, int] = {
    "東": 0,
    "南": 1,
    "西": 2,
    "北": 3,
    "E": 0,
    "S": 1,
    "W": 2,
    "N": 3,
}


def tenhou_pai_to_mjai(code: str) -> str:
    """tenhou_json の牌コード ("0m" 等) を mjai pai 表記に変換する。

    - "0m" / "0p" / "0s" は赤 5 として "5mr" / "5pr" / "5sr"
    - その他はそのまま (mjai と同じ 1m..9m / 1p..9p / 1s..9s / 1z..7z)
    """
    if not isinstance(code, str) or len(code) < 2:
        return code
    if code in ("0m", "0p", "0s"):
        return f"5{code[1]}r"
    return code


def wind_jp_to_mjai(wind: str) -> str:
    """自風/場風 (kanji or 英字) を mjai の "E"/"S"/"W"/"N" に変換する。

    認識失敗等で未知の値が来た場合は "E" にフォールバックする
    (BoardRecognizer の DEFAULT_TENHOU_JSON も "東" = E なので妥当な既定値)。
    """
    return _WIND_JP_TO_MJAI.get(wind, "E")


def parse_round_label(label: str | None) -> tuple[str, int] | None:
    """OCR 由来の局名ラベル (例: "東1局") を (bakaze, kyoku) に分解する。

    雀魂は東風戦 (東1〜東4) と東南戦 (東1〜南4) があり、ラベルは
    "{場風}{数字}局" の形式で来る。パース不能なら None。
    """
    if not isinstance(label, str) or len(label) < 2:
        return None
    bakaze_jp = label[0]
    bakaze = _WIND_JP_TO_MJAI.get(bakaze_jp)
    if bakaze is None:
        return None
    # 数字を 1 文字以上取り出す ("東1局", "南10局" 等は実際にはないが安全側で複数桁許容)。
    num_str = ""
    for ch in label[1:]:
        if ch.isdigit():
            num_str += ch
        else:
            break
    if not num_str:
        return None
    try:
        kyoku = int(num_str)
    except ValueError:
        return None
    if not 1 <= kyoku <= 4:
        return None
    return bakaze, kyoku


def _compute_oya(self_wind: str) -> int:
    """自風から oya (東家の絶対 player index) を算出する。

    mjai の player 順は 0=自家, 1=下家, 2=対面, 3=上家。
    自風が E のとき oya=0, S のとき oya=3, W のとき oya=2, N のとき oya=1。
    """
    idx = _WIND_TO_INDEX.get(self_wind, 0)
    return (4 - idx) % 4


def _hand_counter(hand: Any) -> Counter[str]:
    """`hand` フィールドを Counter に変換する。型不正は空 Counter。"""
    if not isinstance(hand, list):
        return Counter()
    return Counter(p for p in hand if isinstance(p, str))


def _river_signature(river: Any, player: int) -> list[tuple[str, bool]]:
    """指定 player の河を (tile, riichi) のリストに正規化する (出現順保持)。"""
    out: list[tuple[str, bool]] = []
    if not isinstance(river, list):
        return out
    for entry in river:
        if not isinstance(entry, dict):
            continue
        if entry.get("player") != player:
            continue
        tile = entry.get("tile")
        if not isinstance(tile, str):
            continue
        riichi = bool(entry.get("riichi", False))
        out.append((tile, riichi))
    return out


def _melds_for_player(melds: Any, player: int) -> list[dict[str, Any]]:
    """指定 player の副露リストを出現順で抜き出す。"""
    out: list[dict[str, Any]] = []
    if not isinstance(melds, list):
        return out
    for meld in melds:
        if not isinstance(meld, dict):
            continue
        if meld.get("player") != player:
            continue
        out.append(meld)
    return out


def _meld_key(meld: dict[str, Any]) -> tuple[str, tuple[str, ...], int]:
    """副露の同一性判定用のキー (type, tiles tuple, from)。"""
    tiles = meld.get("tiles", [])
    tiles_tuple = tuple(tiles) if isinstance(tiles, list) else ()
    return (
        str(meld.get("type", "")),
        tiles_tuple,
        int(meld.get("from", 0) or 0),
    )


def _build_meld_event(meld: dict[str, Any]) -> dict[str, Any] | None:
    """副露 dict を mjai イベントに変換する。`type` が不正なら None。

    chi / pon / minkan / ankan / kakan を mjai の
    chi / pon / daiminkan / ankan / kakan にマッピングする。
    """
    actor = meld.get("player")
    if not isinstance(actor, int) or not 0 <= actor <= 3:
        return None
    raw_tiles = meld.get("tiles", [])
    if not isinstance(raw_tiles, list):
        return None
    tiles = [tenhou_pai_to_mjai(t) for t in raw_tiles if isinstance(t, str)]
    if not tiles:
        return None
    mtype = meld.get("type")
    from_offset = int(meld.get("from", 0) or 0)
    target = (actor + from_offset) % 4

    if mtype == "chi":
        if len(tiles) < 3:
            return None
        # 雀魂 UI は鳴いた牌を一番左に置く規約だが、recognition 側で
        # 整列保証されているとは限らない。consumed = tiles の残り 2 牌。
        called = tiles[0]
        consumed = tiles[1:]
        return {
            "type": "chi",
            "actor": actor,
            "target": target,
            "pai": called,
            "consumed": consumed,
        }
    if mtype == "pon":
        if len(tiles) < 3:
            return None
        called = tiles[0]
        consumed = tiles[1:]
        return {
            "type": "pon",
            "actor": actor,
            "target": target,
            "pai": called,
            "consumed": consumed,
        }
    if mtype == "minkan":
        if len(tiles) < 4:
            return None
        called = tiles[0]
        consumed = tiles[1:]
        return {
            "type": "daiminkan",
            "actor": actor,
            "target": target,
            "pai": called,
            "consumed": consumed,
        }
    if mtype == "ankan":
        if len(tiles) < 4:
            return None
        return {
            "type": "ankan",
            "actor": actor,
            "consumed": tiles,
        }
    if mtype == "kakan":
        if len(tiles) < 4:
            return None
        # kakan は 既存 pon に 4 枚目を加えた形。pai = 加えた牌, consumed = 既存 3 枚。
        added = tiles[-1]
        consumed = tiles[:-1]
        return {
            "type": "kakan",
            "actor": actor,
            "pai": added,
            "consumed": consumed,
        }
    return None


class SnapshotToMjaiConverter:
    """連続スナップショット → mjai event 列の差分変換器。

    監視ループでインスタンスを 1 つ保持し、フレームごとに `convert()` を呼ぶ。
    返値は「前フレームから今フレームまでに新しく起きた」mjai event の列。

    例:

    >>> conv = SnapshotToMjaiConverter()
    >>> conv.convert(snapshot_at_kyoku_start)  # → [start_kyoku, ...]
    >>> conv.convert(snapshot_after_self_tsumo)  # → [tsumo]
    >>> conv.convert(snapshot_after_self_dahai)  # → [dahai]
    """

    def __init__(self) -> None:
        # 局シグネチャ: (bakaze, kyoku, oya, hand_signature)。手牌が完全に
        # 入れ替わったケースを補助検出するため hand_signature も含める。
        self._kyoku_signature: tuple[str, int, int] | None = None
        # 局開始時の自家手牌 (= mjai start_kyoku.tehais[0])。前局との比較で
        # 完全入れ替えを検知するためにも使う。
        self._initial_self_hand: list[str] = []
        # 直前フレームの自家手牌 (Counter)。tsumo 検知に使う。
        self._prev_self_hand: Counter[str] = Counter()
        # プレイヤー別: 既に dahai event として発行済みの river 長さ。
        # 次フレームでこの長さを超えた分が新しい捨牌。
        self._river_emitted_len: list[int] = [0, 0, 0, 0]
        # プレイヤー別: 既に reach event を発行済みか。雀魂の河は riichi 宣言牌
        # を横向きで表現し、リーチ後も継続して残るため、フラグで一度きりに絞る。
        self._reach_emitted: list[bool] = [False, False, False, False]
        # プレイヤー別: 既に発行済みの副露キーのリスト (出現順)。
        self._melds_emitted: list[list[tuple[str, tuple[str, ...], int]]] = [
            [],
            [],
            [],
            [],
        ]
        # start_kyoku を 1 度も発行していない初期状態。
        self._kyoku_started: bool = False

    # ----------------------------- public API -----------------------------

    def reset(self) -> None:
        """全状態をリセットする (新セッション開始時等)。"""
        self._kyoku_signature = None
        self._initial_self_hand = []
        self._prev_self_hand = Counter()
        self._river_emitted_len = [0, 0, 0, 0]
        self._reach_emitted = [False, False, False, False]
        self._melds_emitted = [[], [], [], []]
        self._kyoku_started = False

    def convert(self, snapshot: dict[str, Any]) -> list[dict[str, Any]]:
        """スナップショット 1 枚を受け取り、新たに発生した mjai event 列を返す。

        前フレームとの差分が取れない / 矛盾している場合はフォールバックとして
        現フレームから擬似 event 列 (start_kyoku + 観測可能な discards) を返す。
        """
        if not isinstance(snapshot, dict):
            logger.warning("snapshot is not a dict; emitting empty event list")
            return []
        try:
            return self._convert_inner(snapshot)
        except Exception:  # noqa: BLE001 — 監視ループを止めないため握る
            logger.warning(
                "snapshot_to_mjai conversion failed; falling back to pseudo events",
                exc_info=True,
            )
            return self._fallback(snapshot)

    # ----------------------------- internals -----------------------------

    def _convert_inner(self, snapshot: dict[str, Any]) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []

        # 1. 局またぎ検出 (round_label or self_wind/round_wind 変化、または初回)。
        new_sig = self._kyoku_signature_of(snapshot)
        if not self._kyoku_started or new_sig != self._kyoku_signature:
            events.append(self._build_start_kyoku(snapshot))
            self._reset_kyoku_state(snapshot, new_sig)
            return events

        # 2. 自家 tsumo 検出 (手牌が 13 → 14 で 1 牌増えた = ツモ)。
        cur_hand = _hand_counter(snapshot.get("hand"))
        gained = cur_hand - self._prev_self_hand
        lost = self._prev_self_hand - cur_hand
        if sum(cur_hand.values()) == 14 and sum(self._prev_self_hand.values()) == 13:
            if sum(gained.values()) == 1 and sum(lost.values()) == 0:
                tsumo_tile = next(iter(gained.elements()))
                events.append(
                    {
                        "type": "tsumo",
                        "actor": 0,
                        "pai": tenhou_pai_to_mjai(tsumo_tile),
                    }
                )
            else:
                logger.warning(
                    "self hand grew by %d but multiset diff is ambiguous "
                    "(gained=%s lost=%s); skipping tsumo emission",
                    1,
                    dict(gained),
                    dict(lost),
                )

        # 3. 副露差分 (player ごとに発行済みリストとの差を見る)。
        meld_events = self._diff_melds(snapshot)
        events.extend(meld_events)

        # 4. 河差分 (新規捨牌 → dahai、riichi フラグ → reach + reach_accepted)。
        river_events = self._diff_rivers(snapshot)
        events.extend(river_events)

        # 5. 次フレーム比較用に state を更新。
        self._prev_self_hand = cur_hand
        return events

    def _kyoku_signature_of(self, snapshot: dict[str, Any]) -> tuple[str, int, int]:
        """局を一意に同定する signature を作る。

        round_label があれば (bakaze, kyoku) を使う。無ければ
        (round_wind, self_wind_index) を使う (= 局またぎ精度は落ちるが
        場風変化なら拾える)。oya は self_wind から算出。
        """
        round_wind = snapshot.get("round_wind", "東")
        self_wind = snapshot.get("self_wind", "東")
        bakaze = wind_jp_to_mjai(round_wind if isinstance(round_wind, str) else "東")
        oya = _compute_oya(self_wind if isinstance(self_wind, str) else "東")
        kyoku_num = 1

        round_label = snapshot.get("round_label")
        parsed = parse_round_label(round_label) if isinstance(round_label, str) else None
        if parsed is not None:
            bakaze, kyoku_num = parsed
        else:
            # round_label が無い場合は turn=1 かつ手牌 13 枚なら "新局" と推定したい
            # が、確定情報が無いので bakaze + self_wind index を擬似 kyoku key
            # とする (= 同じ場風で席が変わったら別局扱い)。
            self_idx = _WIND_TO_INDEX.get(self_wind if isinstance(self_wind, str) else "東", 0)
            kyoku_num = max(1, min(4, self_idx + 1))

        return (bakaze, kyoku_num, oya)

    def _build_start_kyoku(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        """現スナップショットから start_kyoku event を構築する。

        他家手牌は観測不能なので "?" 13 枚で埋める (Phase D4 で libriichi
        側がどう扱うかに合わせる。フォーマット上はこのプレースホルダで通る
        想定)。
        """
        bakaze, kyoku, oya = self._kyoku_signature_of(snapshot)
        scores_raw = snapshot.get("scores")
        if isinstance(scores_raw, list) and len(scores_raw) == 4:
            try:
                scores = [int(s) for s in scores_raw]
            except (TypeError, ValueError):
                scores = [25000, 25000, 25000, 25000]
        else:
            scores = [25000, 25000, 25000, 25000]

        dora_indicators = snapshot.get("dora_indicators") or []
        if (
            isinstance(dora_indicators, list)
            and dora_indicators
            and isinstance(dora_indicators[0], str)
        ):
            dora_marker = tenhou_pai_to_mjai(dora_indicators[0])
        else:
            dora_marker = "1m"

        self_hand_raw = snapshot.get("hand") or []
        if isinstance(self_hand_raw, list):
            self_hand = [tenhou_pai_to_mjai(t) for t in self_hand_raw if isinstance(t, str)]
        else:
            self_hand = []
        # start_kyoku は手牌 13 枚が前提。配牌時点を取れていれば 13 枚で来るが、
        # 既にツモ後 (14 枚) のフレームを最初に観測した場合は末尾 1 枚を捨てて 13 枚にする。
        # 逆に 13 枚未満ならフォールバックの "?" でパディング。
        if len(self_hand) >= 14:
            self_hand = self_hand[:13]
        while len(self_hand) < 13:
            self_hand.append("?")

        tehais: list[list[str]] = [self_hand, ["?"] * 13, ["?"] * 13, ["?"] * 13]

        return {
            "type": "start_kyoku",
            "bakaze": bakaze,
            "dora_marker": dora_marker,
            "kyoku": kyoku,
            "honba": 0,
            "kyotaku": 0,
            "oya": oya,
            "scores": scores,
            "tehais": tehais,
        }

    def _reset_kyoku_state(
        self,
        snapshot: dict[str, Any],
        new_sig: tuple[str, int, int],
    ) -> None:
        """局またぎ時に内部 state を新局向けへ初期化する。"""
        self._kyoku_signature = new_sig
        self._kyoku_started = True
        hand = snapshot.get("hand")
        if isinstance(hand, list):
            self._initial_self_hand = [t for t in hand if isinstance(t, str)]
        else:
            self._initial_self_hand = []
        self._prev_self_hand = _hand_counter(snapshot.get("hand"))
        # 局開始時点で既に河に牌があってもよい (途中フレームから捕捉した場合)。
        # その場合は「既に発行済み」扱いにして dahai を二重発行しない。
        self._river_emitted_len = [
            len(_river_signature(snapshot.get("river"), p)) for p in range(4)
        ]
        # reach も同様: 局またぎでリセット (前局のフラグを引きずらない)。
        # ただし新スナップショット時点で riichi が立っていれば、それは前局からの
        # 持ち越しではなく単に「途中フレームから観測した」ケース。reach event は
        # 過去のものなので発行せず、emitted=True に倒す。
        new_reach = [False, False, False, False]
        for p in range(4):
            tiles = _river_signature(snapshot.get("river"), p)
            if any(r for _, r in tiles):
                new_reach[p] = True
        self._reach_emitted = new_reach
        # 副露も同様に「既に発行済み」扱い。
        self._melds_emitted = [
            [_meld_key(m) for m in _melds_for_player(snapshot.get("melds"), p)] for p in range(4)
        ]

    def _diff_rivers(self, snapshot: dict[str, Any]) -> list[dict[str, Any]]:
        """4 家の河を前回観測長さと比較し、新しい dahai/reach event を発行する。"""
        events: list[dict[str, Any]] = []
        for player in range(4):
            tiles = _river_signature(snapshot.get("river"), player)
            prev_len = self._river_emitted_len[player]
            if len(tiles) < prev_len:
                # 鳴かれて河が縮んだ等、整合性が崩れた可能性。
                # 縮んだぶんは「既知のものは消えた」と解釈し、emitted を新長さまで戻す。
                logger.warning(
                    "river of player %d shrunk %d -> %d; resyncing emitted length",
                    player,
                    prev_len,
                    len(tiles),
                )
                self._river_emitted_len[player] = len(tiles)
                continue
            new_entries = tiles[prev_len:]
            for tile, riichi in new_entries:
                # 横向き = リーチ宣言牌。reach 未発行ならまず reach + reach_accepted、
                # 続いて dahai (= リーチ宣言の打牌) を発行する。
                if riichi and not self._reach_emitted[player]:
                    events.append({"type": "reach", "actor": player})
                    events.append(
                        {
                            "type": "dahai",
                            "actor": player,
                            "pai": tenhou_pai_to_mjai(tile),
                            "tsumogiri": False,
                        }
                    )
                    events.append({"type": "reach_accepted", "actor": player})
                    self._reach_emitted[player] = True
                else:
                    events.append(
                        {
                            "type": "dahai",
                            "actor": player,
                            "pai": tenhou_pai_to_mjai(tile),
                            "tsumogiri": False,
                        }
                    )
            self._river_emitted_len[player] = len(tiles)
        return events

    def _diff_melds(self, snapshot: dict[str, Any]) -> list[dict[str, Any]]:
        """4 家の副露を前回観測リストと比較し、新規副露を mjai event に変換する。"""
        events: list[dict[str, Any]] = []
        for player in range(4):
            cur = _melds_for_player(snapshot.get("melds"), player)
            cur_keys = [_meld_key(m) for m in cur]
            prev_keys = self._melds_emitted[player]
            # 前回キーが prefix になっていない場合 (= 順序が変わった / 失われた)
            # は警告のみ出して emitted を上書きし、新規分のみ発行する。
            if cur_keys[: len(prev_keys)] != prev_keys:
                logger.warning(
                    "melds of player %d became inconsistent (prev=%s cur=%s); "
                    "resyncing without re-emitting",
                    player,
                    prev_keys,
                    cur_keys,
                )
                # 重複発行を避けるため、prev に無い key だけを新規扱いにする。
                prev_set = set(prev_keys)
                new_melds = [m for m, k in zip(cur, cur_keys, strict=False) if k not in prev_set]
            else:
                new_melds = cur[len(prev_keys) :]
            for meld in new_melds:
                event = _build_meld_event(meld)
                if event is not None:
                    events.append(event)
                else:
                    logger.warning("failed to build meld event from %s; skipping", meld)
            self._melds_emitted[player] = cur_keys
        return events

    def _fallback(self, snapshot: dict[str, Any]) -> list[dict[str, Any]]:
        """例外時に、現スナップショットだけから擬似 mjai event 列を生成する。

        精度は落ちるが Mortal を動かすには十分: start_kyoku + 観測されている
        全 dahai (出現順) + 全副露。tsumo / reach はスナップショット単独では
        信頼度が低いので省略する。
        """
        if not isinstance(snapshot, dict):
            return []
        events: list[dict[str, Any]] = []
        try:
            events.append(self._build_start_kyoku(snapshot))
        except Exception:  # noqa: BLE001
            logger.warning("fallback start_kyoku build failed", exc_info=True)
            return []

        # 河を player 単位で出現順に並べて全部 dahai に。
        for player in range(4):
            for tile, _riichi in _river_signature(snapshot.get("river"), player):
                events.append(
                    {
                        "type": "dahai",
                        "actor": player,
                        "pai": tenhou_pai_to_mjai(tile),
                        "tsumogiri": False,
                    }
                )

        # 副露を player 単位で出現順に並べて mjai event 化。
        for player in range(4):
            for meld in _melds_for_player(snapshot.get("melds"), player):
                event = _build_meld_event(meld)
                if event is not None:
                    events.append(event)

        # 内部 state はフォールバック後も「現スナップショットを既知」とみなして
        # 二重発行を避ける。
        new_sig = self._kyoku_signature_of(snapshot)
        self._reset_kyoku_state(snapshot, new_sig)
        return events
