"""issue #20: `mortal.action_formatter` の単体テスト。

policy/value/q_values → InferenceResult 互換 dict 整形ロジックを検証する。
torch / vendor mortal に一切依存しないため、`[mortal]` extras 抜きでも実行可能。
"""

from __future__ import annotations

import pytest

import mortal  # noqa: F401 — sys.path 調整副作用
from mortal.action_formatter import (
    format_inference_result,
    make_primary_label,
    parse_mjai_action,
)

# --------------------------------- parse_mjai_action ---------------------------------


def test_parse_discard_basic() -> None:
    parsed = parse_mjai_action("discard:5m")
    assert parsed["action_type"] == "discard"
    assert parsed["tile"] == "5m"
    assert parsed["raw_action"] == "discard:5m"
    # discard には action_label を付けない (issue #20 の例の "discard" 行に倣う)。
    assert "action_label" not in parsed


def test_parse_discard_red_five() -> None:
    """赤 5 (`5mr`) は丸ごと 1 牌として保持される (分割しない)。"""
    parsed = parse_mjai_action("discard:5mr")
    assert parsed["action_type"] == "discard"
    assert parsed["tile"] == "5mr"


def test_parse_reach() -> None:
    parsed = parse_mjai_action("reach")
    assert parsed["action_type"] == "riichi"
    assert parsed["action_label"] == "リーチ"
    assert "tile" not in parsed


def test_parse_pon() -> None:
    """issue #20 仕様: `pon:1m1m` → `tile="1m"` (先頭 1 牌のみ採用)。"""
    parsed = parse_mjai_action("pon:5m5m")
    assert parsed["action_type"] == "pon"
    assert parsed["action_label"] == "ポン"
    # libriichi が consumed を連結して emit するが、UI 表示用には先頭 1 牌のみ。
    # raw 連結 "5m5m" を tile に入れるとフロントで "5m5m" が描画される事故になる。
    assert parsed["tile"] == "5m"


def test_parse_pon_with_red_five() -> None:
    """赤 5 を含む鳴き (`5mr` 3 文字) も先頭 1 牌として正しく切り出す。"""
    parsed = parse_mjai_action("pon:5mr5m")
    assert parsed["tile"] == "5mr"
    parsed_normal_first = parse_mjai_action("pon:5m5mr")
    assert parsed_normal_first["tile"] == "5m"


def test_parse_ankan_takes_first_tile() -> None:
    """ankan は同種 4 牌だが tile には先頭 1 牌のみ入れる。"""
    parsed = parse_mjai_action("ankan:5m5m5m5m")
    assert parsed["action_type"] == "kan"
    assert parsed["tile"] == "5m"


def test_parse_chi_variants_all_map_to_chi() -> None:
    """`chi_low` / `chi_mid` / `chi_high` の 3 変種はすべて ActionType=chi に集約。"""
    for prefix in ("chi_low", "chi_mid", "chi_high"):
        parsed = parse_mjai_action(f"{prefix}:5m")
        assert parsed["action_type"] == "chi", f"failed for {prefix}"
        assert parsed["action_label"] == "チー"
        assert parsed["tile"] == "5m"


def test_parse_kan_variants_all_map_to_kan() -> None:
    for prefix in ("daiminkan", "ankan", "kakan"):
        parsed = parse_mjai_action(f"{prefix}:1z")
        assert parsed["action_type"] == "kan", f"failed for {prefix}"
        assert parsed["action_label"] == "カン"


def test_parse_hora_defaults_to_tsumo() -> None:
    """`hora` は context-free なので tsumo にデフォルト (TODO: 後続 issue で文脈判定)。"""
    parsed = parse_mjai_action("hora")
    assert parsed["action_type"] == "tsumo"
    assert parsed["action_label"] == "ツモ"


def test_parse_ryukyoku() -> None:
    parsed = parse_mjai_action("ryukyoku")
    assert parsed["action_type"] == "pass"
    assert parsed["action_label"] == "九種九牌"


def test_parse_none_is_pass() -> None:
    parsed = parse_mjai_action("none")
    assert parsed["action_type"] == "pass"
    assert parsed["action_label"] == "スルー"


def test_parse_unknown_prefix_falls_back_to_pass() -> None:
    """未知 prefix は安全側で pass。raw_action は保持される。"""
    parsed = parse_mjai_action("mystery_action:xyz")
    assert parsed["action_type"] == "pass"
    assert parsed["raw_action"] == "mystery_action:xyz"


# --------------------------------- make_primary_label ---------------------------------


def test_primary_label_discard_includes_tile() -> None:
    parsed = parse_mjai_action("discard:6m")
    assert make_primary_label(parsed) == "6m を切る"


def test_primary_label_riichi() -> None:
    assert make_primary_label(parse_mjai_action("reach")) == "リーチ"


def test_primary_label_each_action_type() -> None:
    # Rust src-tauri/src/monitor.rs:format_primary_label と同一マッピング。
    cases = {
        "pon:5m5m": "ポン",
        "chi_low:5m": "チー",
        "daiminkan:1z": "カン",
        "hora": "ツモ",
        "none": "スルー",
    }
    for action, expected in cases.items():
        assert make_primary_label(parse_mjai_action(action)) == expected, action


def test_primary_label_discard_without_tile_falls_back() -> None:
    """`discard:` 後が空でも壊れない (汎用「打牌」表記)。"""
    assert make_primary_label({"action_type": "discard"}) == "打牌"


# --------------------------------- format_inference_result ---------------------------------


def _sample_policy() -> dict[str, float]:
    # 7 件: top-5 抽出と sort 検証 + 1 件は閾値以下、もう 1 件は補欠。
    return {
        "discard:1m": 0.05,
        "discard:6m": 0.61,
        "discard:9p": 0.18,
        "discard:1z": 0.04,
        "reach": 0.10,
        "pon:5m5m": 0.02,
        "discard:2m": 1e-9,  # illegal (閾値以下) — 除外されるべき
    }


def _sample_q_values() -> dict[str, float]:
    return {
        "discard:1m": -0.10,
        "discard:6m": 0.32,
        "discard:9p": 0.18,
        "discard:1z": -0.05,
        "reach": 0.40,
        "pon:5m5m": -0.20,
        "discard:2m": -1.0,
    }


def test_format_returns_top_5_candidates() -> None:
    result = format_inference_result(_sample_policy(), 0.32, _sample_q_values())
    assert len(result["candidates"]) == 5


def test_format_recommended_is_policy_argmax() -> None:
    result = format_inference_result(_sample_policy(), 0.32, _sample_q_values())
    rec = result["recommended"]
    assert rec["tile"] == "6m"
    assert rec["action_type"] == "discard"
    assert rec["probability"] == pytest.approx(0.61)
    # expected_value は Q value (= 局期待ポイントの delta、符号付き可)。
    assert rec["expected_value"] == pytest.approx(0.32)


def test_format_candidates_sorted_descending_by_probability() -> None:
    result = format_inference_result(_sample_policy(), 0.32, _sample_q_values())
    probs = [c["probability"] for c in result["candidates"]]
    assert probs == sorted(probs, reverse=True)


def test_format_filters_actions_below_threshold() -> None:
    """確率が `_LEGAL_POLICY_THRESHOLD` 未満の action (illegal) は除外される。"""
    result = format_inference_result(_sample_policy(), 0.32, _sample_q_values())
    raw_actions = [c["detail"] for c in result["candidates"]]
    # discard:2m (1e-9) は閾値以下なので candidates に現れないことを確認。
    # detail = "P=X.XX, Q=±X.XX" 形式なので、policy=1e-9 の Q=-1.00 が現れないことを検証。
    assert all("Q=-1.00" not in d for d in raw_actions)


def test_format_primary_label_matches_recommended() -> None:
    result = format_inference_result(_sample_policy(), 0.32, _sample_q_values())
    assert result["primary_label"] == "6m を切る"


def test_format_preserves_raw_fields_for_debug() -> None:
    """raw policy/value/q_values は SQLite log や S-03 デバッグビュー用に温存する。"""
    policy = _sample_policy()
    q = _sample_q_values()
    result = format_inference_result(policy, 0.32, q)
    assert result["policy"] == policy
    assert result["q_values"] == q
    assert result["value"] == pytest.approx(0.32)
    assert isinstance(result["timestamp"], str) and "T" in result["timestamp"]


def test_format_candidate_detail_contains_p_and_q() -> None:
    result = format_inference_result(_sample_policy(), 0.32, _sample_q_values())
    rec = result["recommended"]
    assert "P=" in rec["detail"]
    assert "Q=" in rec["detail"]


def test_format_empty_policy_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        format_inference_result({}, 0.0, {})


def test_format_all_below_threshold_raises() -> None:
    """全 action が illegal な policy は呼び出し側のバグなので明示的に失敗させる。"""
    with pytest.raises(ValueError, match="no legal actions"):
        format_inference_result(
            {"discard:1m": 1e-9, "reach": 1e-12},
            0.0,
            {"discard:1m": 0.0, "reach": 0.0},
        )


def test_format_invalid_top_k_raises() -> None:
    with pytest.raises(ValueError, match="top_k"):
        format_inference_result(_sample_policy(), 0.0, _sample_q_values(), top_k=0)


def test_format_top_k_smaller_than_actions() -> None:
    """top_k=2 なら candidates も 2 件に絞られる。"""
    result = format_inference_result(_sample_policy(), 0.32, _sample_q_values(), top_k=2)
    assert len(result["candidates"]) == 2
    assert result["candidates"][0]["tile"] == "6m"


def test_format_handles_missing_q_value_gracefully() -> None:
    """q_values に対応 key が無くても 0.0 fallback で動く。"""
    policy = {"discard:6m": 0.61, "reach": 0.39}
    q_values: dict[str, float] = {"discard:6m": 0.32}
    result = format_inference_result(policy, 0.32, q_values)
    reach_candidate = next(c for c in result["candidates"] if c["action_type"] == "riichi")
    assert reach_candidate["expected_value"] == 0.0
