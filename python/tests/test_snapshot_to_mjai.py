"""issue #19: tenhou_json スナップショット → mjai event 列変換のテスト。

受け入れ基準:
- 連続スナップショット (start_kyoku → tsumo → dahai × 4 周) が正しく event 列
  に変換される
- 不整合時もクラッシュせず警告ログを出して継続
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from mortal.snapshot_to_mjai import (
    SnapshotToMjaiConverter,
    parse_round_label,
    tenhou_pai_to_mjai,
    wind_jp_to_mjai,
)

# ----------------------- helpers -----------------------


def _base_snapshot() -> dict[str, Any]:
    """13 牌配牌直後の自家を想定した最小スナップショット。"""
    return {
        "hand": [
            "1m",
            "2m",
            "3m",
            "4m",
            "5m",
            "6m",
            "7m",
            "8m",
            "9m",
            "1p",
            "2p",
            "3p",
            "4p",
        ],
        "river": [],
        "melds": [],
        "dora_indicators": ["5p"],
        "self_wind": "東",
        "round_wind": "東",
        "round_label": "東1局",
        "turn": 1,
        "scores": [25000, 25000, 25000, 25000],
    }


def _add_self_tsumo(snapshot: dict[str, Any], tile: str) -> dict[str, Any]:
    s = copy.deepcopy(snapshot)
    s["hand"].append(tile)
    return s


def _self_dahai(snapshot: dict[str, Any], tile: str) -> dict[str, Any]:
    s = copy.deepcopy(snapshot)
    s["hand"].remove(tile)
    s["river"].append({"player": 0, "tile": tile, "tedashi": True})
    return s


def _add_river(
    snapshot: dict[str, Any],
    player: int,
    tile: str,
    *,
    riichi: bool = False,
) -> dict[str, Any]:
    s = copy.deepcopy(snapshot)
    entry: dict[str, Any] = {"player": player, "tile": tile, "tedashi": True}
    if riichi:
        entry["riichi"] = True
    s["river"].append(entry)
    return s


# ----------------------- module helper tests -----------------------


def test_tenhou_pai_to_mjai_red_5() -> None:
    assert tenhou_pai_to_mjai("0m") == "5mr"
    assert tenhou_pai_to_mjai("0p") == "5pr"
    assert tenhou_pai_to_mjai("0s") == "5sr"


def test_tenhou_pai_to_mjai_passthrough() -> None:
    assert tenhou_pai_to_mjai("1m") == "1m"
    assert tenhou_pai_to_mjai("9s") == "9s"
    assert tenhou_pai_to_mjai("7z") == "7z"


def test_wind_jp_to_mjai_basic() -> None:
    assert wind_jp_to_mjai("東") == "E"
    assert wind_jp_to_mjai("南") == "S"
    assert wind_jp_to_mjai("西") == "W"
    assert wind_jp_to_mjai("北") == "N"


def test_wind_jp_to_mjai_unknown_falls_back_to_east() -> None:
    assert wind_jp_to_mjai("") == "E"
    assert wind_jp_to_mjai("???") == "E"


def test_parse_round_label_valid() -> None:
    assert parse_round_label("東1局") == ("E", 1)
    assert parse_round_label("東4局") == ("E", 4)
    assert parse_round_label("南2局") == ("S", 2)


def test_parse_round_label_invalid() -> None:
    assert parse_round_label(None) is None
    assert parse_round_label("") is None
    assert parse_round_label("中5局") is None  # 不正な bakaze 文字
    assert parse_round_label("東x局") is None
    assert parse_round_label("東5局") is None  # 5 局以降は不正


# ----------------------- start_kyoku -----------------------


def test_first_snapshot_emits_start_kyoku() -> None:
    conv = SnapshotToMjaiConverter()
    events = conv.convert(_base_snapshot())
    assert len(events) == 1
    sk = events[0]
    assert sk["type"] == "start_kyoku"
    assert sk["bakaze"] == "E"
    assert sk["kyoku"] == 1
    assert sk["oya"] == 0  # 自風 東 → oya = 0
    assert sk["scores"] == [25000, 25000, 25000, 25000]
    assert sk["dora_marker"] == "5p"
    assert len(sk["tehais"]) == 4
    assert len(sk["tehais"][0]) == 13
    # 他家は観測不能なので "?" で埋まる。
    assert sk["tehais"][1] == ["?"] * 13


def test_start_kyoku_oya_when_self_is_south() -> None:
    conv = SnapshotToMjaiConverter()
    s = _base_snapshot()
    s["self_wind"] = "南"
    s["round_label"] = "東2局"
    events = conv.convert(s)
    assert events[0]["type"] == "start_kyoku"
    # 南家 = 自家 0 から見て上家 (3) が東家。
    assert events[0]["oya"] == 3
    assert events[0]["bakaze"] == "E"
    assert events[0]["kyoku"] == 2


def test_start_kyoku_red_dora_converted() -> None:
    conv = SnapshotToMjaiConverter()
    s = _base_snapshot()
    s["dora_indicators"] = ["0m"]
    events = conv.convert(s)
    assert events[0]["dora_marker"] == "5mr"


# ----------------------- tsumo / dahai cycle -----------------------


def test_self_tsumo_then_dahai_cycle() -> None:
    """受け入れ基準: 連続スナップショット (start_kyoku → tsumo → dahai × 4 周)。

    自家視点でツモ → 打牌を 4 周繰り返した場合に正しく event 列が生成されること。
    ツモ牌と同じ牌を打牌するので tsumogiri=True が立つ。
    """
    conv = SnapshotToMjaiConverter()
    s0 = _base_snapshot()
    events = conv.convert(s0)
    assert events[0]["type"] == "start_kyoku"

    # 4 周分のツモ/打牌を作る。ツモ牌と打牌は同じにして手牌を不変に保つ
    # (= 各サイクルが 13 → 14 → 13 を厳密に踏む)。
    cycle_tiles = ["1s", "2s", "3s", "4s"]
    current = s0
    for tile in cycle_tiles:
        s_tsumo = _add_self_tsumo(current, tile)
        ev_tsumo = conv.convert(s_tsumo)
        assert len(ev_tsumo) == 1
        assert ev_tsumo[0]["type"] == "tsumo"
        assert ev_tsumo[0]["actor"] == 0
        assert ev_tsumo[0]["pai"] == tile

        s_dahai = _self_dahai(s_tsumo, tile)
        ev_dahai = conv.convert(s_dahai)
        assert len(ev_dahai) == 1
        assert ev_dahai[0]["type"] == "dahai"
        assert ev_dahai[0]["actor"] == 0
        assert ev_dahai[0]["pai"] == tile
        # ツモった牌をそのまま切ったので tsumogiri=True (gemini Medium on PR #51)。
        assert ev_dahai[0]["tsumogiri"] is True

        current = s_dahai


def test_self_dahai_tedashi_when_different_from_tsumo() -> None:
    """ツモ牌と異なる牌を切った場合は tsumogiri=False (= 手出し)。"""
    conv = SnapshotToMjaiConverter()
    s0 = _base_snapshot()
    conv.convert(s0)
    s_tsumo = _add_self_tsumo(s0, "1s")  # 1s をツモ
    conv.convert(s_tsumo)
    # 手から 9m を切る (1s は手に残る)。
    s_dahai = copy.deepcopy(s_tsumo)
    s_dahai["hand"].remove("9m")
    s_dahai["river"].append({"player": 0, "tile": "9m", "tedashi": True})
    events = conv.convert(s_dahai)
    assert len(events) == 1
    assert events[0]["type"] == "dahai"
    assert events[0]["pai"] == "9m"
    assert events[0]["tsumogiri"] is False


def test_red_5_tsumo_is_converted_to_5mr() -> None:
    conv = SnapshotToMjaiConverter()
    s0 = _base_snapshot()
    conv.convert(s0)
    s1 = _add_self_tsumo(s0, "0m")
    events = conv.convert(s1)
    assert events[0]["type"] == "tsumo"
    assert events[0]["pai"] == "5mr"


def test_other_player_dahai_is_emitted() -> None:
    """他家 dahai の前にはプレースホルダ tsumo が挿入される (mjai 仕様)。"""
    conv = SnapshotToMjaiConverter()
    s0 = _base_snapshot()
    conv.convert(s0)
    # 下家 (player=1) が捨てる。
    s1 = _add_river(s0, player=1, tile="9m")
    events = conv.convert(s1)
    types = [e["type"] for e in events]
    assert types == ["tsumo", "dahai"]
    assert events[0]["actor"] == 1
    assert events[0]["pai"] == "?"  # 他家ツモは観測不能
    assert events[1]["actor"] == 1
    assert events[1]["pai"] == "9m"


def test_river_events_preserve_chronological_order_across_players() -> None:
    """複数家が同フレームに打牌した場合、player 番号順ではなく出現順で発行される。"""
    conv = SnapshotToMjaiConverter()
    s0 = _base_snapshot()
    conv.convert(s0)
    # 出現順 (時系列): 下家(1) → 対面(2) → 上家(3) → 自家(0)
    s1 = copy.deepcopy(s0)
    s1["river"] = [
        {"player": 1, "tile": "9m", "tedashi": True},
        {"player": 2, "tile": "8m", "tedashi": True},
        {"player": 3, "tile": "7m", "tedashi": True},
        {"player": 0, "tile": "6m", "tedashi": True},
    ]
    # 自家の打牌前にツモが必要なので hand の不整合を避けるため手牌を 14 → 13 に。
    s1["hand"] = list(s1["hand"])
    s1["hand"].remove("6m" if "6m" in s1["hand"] else s1["hand"][0])
    events = conv.convert(s1)
    # 各 dahai の actor 順序が river 配列順そのまま (player 番号順 0,1,2,3 ではない)。
    dahai_actors = [e["actor"] for e in events if e["type"] == "dahai"]
    assert dahai_actors == [1, 2, 3, 0]


# ----------------------- reach -----------------------


def test_riichi_declaration_emits_reach_dahai_accepted() -> None:
    conv = SnapshotToMjaiConverter()
    s0 = _base_snapshot()
    conv.convert(s0)
    # 対面 (player=2) がリーチ宣言牌として "5z" を捨てた。
    s1 = _add_river(s0, player=2, tile="5z", riichi=True)
    events = conv.convert(s1)
    types = [e["type"] for e in events]
    # 他家のため先頭にプレースホルダ tsumo が入る。
    assert types == ["tsumo", "reach", "dahai", "reach_accepted"]
    assert events[0]["actor"] == 2
    assert events[1]["actor"] == 2
    assert events[2]["actor"] == 2
    assert events[2]["pai"] == "5z"
    assert events[3]["actor"] == 2


def test_reach_is_not_re_emitted_for_subsequent_riichi_tiles() -> None:
    conv = SnapshotToMjaiConverter()
    s0 = _base_snapshot()
    conv.convert(s0)
    s1 = _add_river(s0, player=3, tile="1z", riichi=True)
    conv.convert(s1)
    # 既にリーチ済みのプレイヤーが追加で捨てた場合、reach は再発行されない。
    # (本来 riichi=True は宣言牌のみだが、recognition のノイズで横向き判定が
    # 別牌でも立つ可能性を想定したガード)。
    s2 = _add_river(s1, player=3, tile="2z", riichi=True)
    events = conv.convert(s2)
    types = [e["type"] for e in events]
    assert "reach" not in types
    assert "reach_accepted" not in types
    # 他家のため tsumo プレースホルダが入った後 dahai。
    assert types == ["tsumo", "dahai"]


# ----------------------- melds (pon/chi/kan) -----------------------


def test_pon_meld_emits_pon_event() -> None:
    conv = SnapshotToMjaiConverter()
    s0 = _base_snapshot()
    conv.convert(s0)
    # 下家 (player=1) が上家 (from=3) から 5m を pon した。
    # 1 番下家から見て上家 (from=3) は、絶対座標で (1+3)%4 = 0 = 自家。
    s1 = copy.deepcopy(s0)
    s1["melds"] = [
        {
            "player": 1,
            "type": "pon",
            "tiles": ["5m", "5m", "5m"],
            "from": 3,
        }
    ]
    events = conv.convert(s1)
    assert len(events) == 1
    e = events[0]
    assert e["type"] == "pon"
    assert e["actor"] == 1
    assert e["target"] == 0  # (1 + 3) % 4 = 0
    assert e["pai"] == "5m"
    assert e["consumed"] == ["5m", "5m"]


def test_chi_meld_emits_chi_event_with_called_index_leftmost() -> None:
    """called_index=0: 左端の牌 (3p) を鳴いた → pai=3p, consumed=[4p,5p]。"""
    conv = SnapshotToMjaiConverter()
    s0 = _base_snapshot()
    conv.convert(s0)
    s1 = copy.deepcopy(s0)
    s1["melds"] = [
        {
            "player": 0,
            "type": "chi",
            "tiles": ["3p", "4p", "5p"],
            "from": 3,
            "called_index": 0,
        }
    ]
    events = conv.convert(s1)
    assert len(events) == 1
    e = events[0]
    assert e["type"] == "chi"
    assert e["actor"] == 0
    assert e["target"] == 3  # (0 + 3) % 4 = 3
    assert e["pai"] == "3p"
    assert e["consumed"] == ["4p", "5p"]


def test_chi_meld_called_index_middle() -> None:
    """called_index=1: 中央の牌 (4m) を鳴いた → pai=4m, consumed=[3m,5m] (CodeRabbit Critical)。"""
    conv = SnapshotToMjaiConverter()
    s0 = _base_snapshot()
    conv.convert(s0)
    s1 = copy.deepcopy(s0)
    s1["melds"] = [
        {
            "player": 0,
            "type": "chi",
            "tiles": ["3m", "4m", "5m"],
            "from": 3,
            "called_index": 1,
        }
    ]
    events = conv.convert(s1)
    assert events[0]["pai"] == "4m"
    assert events[0]["consumed"] == ["3m", "5m"]


def test_chi_meld_called_index_rightmost() -> None:
    """called_index=2: 右端の牌 (9p) を鳴いた → pai=9p, consumed=[7p,8p]。"""
    conv = SnapshotToMjaiConverter()
    s0 = _base_snapshot()
    conv.convert(s0)
    s1 = copy.deepcopy(s0)
    s1["melds"] = [
        {
            "player": 0,
            "type": "chi",
            "tiles": ["7p", "8p", "9p"],
            "from": 3,
            "called_index": 2,
        }
    ]
    events = conv.convert(s1)
    assert events[0]["pai"] == "9p"
    assert events[0]["consumed"] == ["7p", "8p"]


def test_chi_meld_missing_called_index_falls_back_to_zero() -> None:
    """called_index 欠落 (旧 schema) では tiles[0] にフォールバック。"""
    conv = SnapshotToMjaiConverter()
    s0 = _base_snapshot()
    conv.convert(s0)
    s1 = copy.deepcopy(s0)
    s1["melds"] = [
        {
            "player": 0,
            "type": "chi",
            "tiles": ["3p", "4p", "5p"],
            "from": 3,
            # called_index 未指定
        }
    ]
    events = conv.convert(s1)
    assert events[0]["pai"] == "3p"
    assert events[0]["consumed"] == ["4p", "5p"]


def test_minkan_emits_daiminkan_event() -> None:
    conv = SnapshotToMjaiConverter()
    s0 = _base_snapshot()
    conv.convert(s0)
    s1 = copy.deepcopy(s0)
    s1["melds"] = [
        {
            "player": 2,
            "type": "minkan",
            "tiles": ["7z", "7z", "7z", "7z"],
            "from": 1,
        }
    ]
    events = conv.convert(s1)
    assert len(events) == 1
    assert events[0]["type"] == "daiminkan"
    assert events[0]["actor"] == 2
    assert events[0]["target"] == 3  # (2 + 1) % 4 = 3


def test_ankan_emits_ankan_event_without_target() -> None:
    conv = SnapshotToMjaiConverter()
    s0 = _base_snapshot()
    conv.convert(s0)
    s1 = copy.deepcopy(s0)
    s1["melds"] = [
        {
            "player": 0,
            "type": "ankan",
            "tiles": ["1z", "1z", "1z", "1z"],
            "from": 0,
        }
    ]
    events = conv.convert(s1)
    assert len(events) == 1
    e = events[0]
    assert e["type"] == "ankan"
    assert e["actor"] == 0
    assert e["consumed"] == ["1z", "1z", "1z", "1z"]
    assert "target" not in e


def test_kakan_event_shape() -> None:
    conv = SnapshotToMjaiConverter()
    s0 = _base_snapshot()
    conv.convert(s0)
    s1 = copy.deepcopy(s0)
    s1["melds"] = [
        {
            "player": 1,
            "type": "kakan",
            # 既存 pon ["2m","2m","2m"] に加槓 1 枚 → 4 枚目を末尾に。
            "tiles": ["2m", "2m", "2m", "2m"],
            "from": 2,
        }
    ]
    events = conv.convert(s1)
    assert len(events) == 1
    e = events[0]
    assert e["type"] == "kakan"
    assert e["actor"] == 1
    assert e["pai"] == "2m"
    assert e["consumed"] == ["2m", "2m", "2m"]


def test_meld_persists_across_frames_without_re_emit() -> None:
    """副露は次フレーム以降も meld list に残るが、event は 1 回だけ発行される。"""
    conv = SnapshotToMjaiConverter()
    s0 = _base_snapshot()
    conv.convert(s0)
    meld = {"player": 1, "type": "pon", "tiles": ["3z", "3z", "3z"], "from": 3}
    s1 = copy.deepcopy(s0)
    s1["melds"] = [meld]
    events = conv.convert(s1)
    assert len(events) == 1
    assert events[0]["type"] == "pon"
    # 次フレームでも同じ meld が残っているが再発行しない。
    events2 = conv.convert(copy.deepcopy(s1))
    assert events2 == []


# ----------------------- kyoku boundary -----------------------


def test_kyoku_change_emits_new_start_kyoku() -> None:
    conv = SnapshotToMjaiConverter()
    s0 = _base_snapshot()
    conv.convert(s0)
    # 東1局 → 東2局。河や手牌の遷移も新局に入れ替わる想定。
    s1 = _base_snapshot()
    s1["round_label"] = "東2局"
    s1["self_wind"] = "北"  # 親が変わって自分は北家
    events = conv.convert(s1)
    assert events[0]["type"] == "start_kyoku"
    assert events[0]["kyoku"] == 2
    # 北家視点で oya は (4 - 3) % 4 = 1。
    assert events[0]["oya"] == 1


def test_kyoku_fallback_uses_fixed_1_when_no_round_label() -> None:
    """round_label が無い場合、kyoku は 1 で固定 (gemini Medium on PR #51)。

    自風 index から kyoku を推定するのは不正確 (南家でも東2局とは限らない)
    なので、確定情報無しでは 1 にフォールバックする。
    """
    conv = SnapshotToMjaiConverter()
    s = _base_snapshot()
    s.pop("round_label", None)
    s["self_wind"] = "南"
    events = conv.convert(s)
    assert events[0]["type"] == "start_kyoku"
    assert events[0]["kyoku"] == 1
    # oya は self_wind から算出されるのでこちらは変わる (南家 → oya=3)。
    assert events[0]["oya"] == 3


def test_start_kyoku_with_14_tile_hand_emits_followup_tsumo() -> None:
    """14 牌で局を観測した場合、start_kyoku (13 牌) + 続く tsumo (14 枚目) を発行。

    14 牌目を切り捨てるだけだと mjai ストリーム上で消失する (gemini Medium on PR #51)。
    """
    conv = SnapshotToMjaiConverter()
    s = _base_snapshot()
    s["hand"] = list(s["hand"]) + ["1s"]  # 14 牌目を追加
    events = conv.convert(s)
    assert len(events) == 2
    assert events[0]["type"] == "start_kyoku"
    assert len(events[0]["tehais"][0]) == 13
    assert events[1] == {"type": "tsumo", "actor": 0, "pai": "1s"}


def test_no_redundant_start_kyoku_when_signature_unchanged() -> None:
    conv = SnapshotToMjaiConverter()
    s0 = _base_snapshot()
    events = conv.convert(s0)
    assert events[0]["type"] == "start_kyoku"
    # 同一スナップショットを再投入 → 何も発行されない。
    events2 = conv.convert(copy.deepcopy(s0))
    assert events2 == []


# ----------------------- fallbacks -----------------------


def test_non_dict_snapshot_returns_empty() -> None:
    conv = SnapshotToMjaiConverter()
    # 型不正は warning 出して空リスト。例外は伝播しない。
    assert conv.convert(None) == []  # type: ignore[arg-type]
    assert conv.convert("not a dict") == []  # type: ignore[arg-type]


def test_ambiguous_hand_growth_does_not_crash() -> None:
    """手牌が 13 → 15 等の異常遷移でも、警告のみで処理継続する。"""
    conv = SnapshotToMjaiConverter()
    s0 = _base_snapshot()
    conv.convert(s0)
    s1 = copy.deepcopy(s0)
    s1["hand"] = list(s1["hand"]) + ["1s", "2s"]  # 13 → 15 (異常)
    # クラッシュせず、event は発行されないこと (tsumo として確定できない)。
    events = conv.convert(s1)
    assert isinstance(events, list)
    assert all(e["type"] != "tsumo" for e in events)


def test_reset_clears_state() -> None:
    conv = SnapshotToMjaiConverter()
    conv.convert(_base_snapshot())
    conv.reset()
    # reset 後は次回 convert で start_kyoku が再発行される。
    events = conv.convert(_base_snapshot())
    assert events[0]["type"] == "start_kyoku"


def test_river_shrink_resyncs_without_crash(caplog: pytest.LogCaptureFixture) -> None:
    """雀魂で他家が鳴いて河から牌が消えた場合に、emitted 長を再同期する。"""
    conv = SnapshotToMjaiConverter()
    s0 = _base_snapshot()
    conv.convert(s0)
    s1 = _add_river(s0, player=1, tile="9m")
    s1 = _add_river(s1, player=1, tile="9s")
    conv.convert(s1)
    # 鳴かれて player 1 の最後の捨牌が消えた状態。
    s2 = copy.deepcopy(s1)
    s2["river"] = [s2["river"][0]]  # 9m のみ残る (9s は鳴かれて消失)
    # クラッシュせず警告だけ出る。
    events = conv.convert(s2)
    # 新規 dahai は生まれない (= 既知の 9m はもう発行済み)。
    assert all(e["type"] != "dahai" for e in events)


# ----------------------- mid-stream attachment -----------------------


def test_attaching_mid_kyoku_does_not_re_emit_existing_events() -> None:
    """既に途中の局からスナップショット観測を始めた場合、過去の river/meld は
    重複発行しない (初回 start_kyoku のみ発行する)。"""
    conv = SnapshotToMjaiConverter()
    s = _base_snapshot()
    s["river"] = [
        {"player": 0, "tile": "9m", "tedashi": True},
        {"player": 1, "tile": "1s", "tedashi": True},
    ]
    s["melds"] = [{"player": 2, "type": "pon", "tiles": ["1z", "1z", "1z"], "from": 1}]
    events = conv.convert(s)
    # 初回は start_kyoku 1 件のみ。
    assert len(events) == 1
    assert events[0]["type"] == "start_kyoku"
    # 同じ snapshot をもう一度投げても event 発生しない。
    assert conv.convert(copy.deepcopy(s)) == []
