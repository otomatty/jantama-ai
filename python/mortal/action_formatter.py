"""Mortal の policy/value/q_values → `InferenceResult` 整形レイヤ (issue #20)。

Mortal は `policy` (各 mjai action の確率分布) と `value` (局の期待ポイント)
を返すが、フロントが期待するのは `RecommendationCandidate[]` 整形済みの shape。
本モジュールは純粋関数のみで構成し、`torch` / vendor `mortal` に一切依存
しない。これは `test_mortal_engine_module_has_no_top_level_torch_import`
が課している規約と同じ理由で、CI が `[mortal]` extras 未インストール環境でも
通るようにするため。

mjai action 文字列の規約 (libriichi / vendor Mortal が emit する形式):

- `discard:<pai>` — 打牌。例 "discard:5m", "discard:5mr" (赤 5)
- `reach` — リーチ宣言
- `pon:<pai>` — ポン (called tile のみ表記)
- `chi_low:<pai>` / `chi_mid:<pai>` / `chi_high:<pai>` — チー 3 変種
- `daiminkan:<pai>` / `ankan:<pai>` / `kakan:<pai>` — 各カン
- `hora` — 和了 (mjai 仕様では tsumo/ron の区別はここでは付かない;
  直前の event 文脈で判定する必要があるが、本モジュールは context-free
  なので tsumo にデフォルト)
- `ryukyoku` — 九種九牌
- `none` — 何もしない (パス)
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

# mjai action prefix → frontend `ActionType` (src/types/index.ts:158-167)。
# `chi_low`/`chi_mid`/`chi_high` は frontend では区別しないので全て "chi"。
_ACTION_TYPE_MAP: dict[str, str] = {
    "discard": "discard",
    "reach": "riichi",
    "pon": "pon",
    "chi": "chi",
    "chi_low": "chi",
    "chi_mid": "chi",
    "chi_high": "chi",
    "daiminkan": "kan",
    "ankan": "kan",
    "kakan": "kan",
    "hora": "tsumo",  # TODO(#follow-up): mjai context (直前 event) で tsumo/ron 判定
    "ryukyoku": "pass",
    "none": "pass",
}

# 各 action の日本語ラベル (Rust `src-tauri/src/monitor.rs:format_primary_label`
# に揃える)。`discard` はタイルが入るので個別に組み立てる。
_ACTION_LABEL_MAP: dict[str, str] = {
    "reach": "リーチ",
    "pon": "ポン",
    "chi": "チー",
    "chi_low": "チー",
    "chi_mid": "チー",
    "chi_high": "チー",
    "daiminkan": "カン",
    "ankan": "カン",
    "kakan": "カン",
    "hora": "ツモ",
    "ryukyoku": "九種九牌",
    "none": "スルー",
}

# 受け入れ済み (= legal) と判定する確率の下限。softmax 後に illegal action は
# 通常 0 に倒れるが、浮動小数誤差で 1e-30 程度残ることがあるため、これらを
# candidates から除外する閾値として使う。
_LEGAL_POLICY_THRESHOLD = 1e-6


def parse_mjai_action(action_str: str) -> dict[str, Any]:
    """mjai action 文字列を frontend `RecommendationCandidate` 互換 dict にパースする。

    返値の必須キー: `action_type` (ActionType enum 値), `raw_action`。
    optional: `tile` (打牌 / 鳴き対象牌), `action_label` (日本語表示文)。

    未知の prefix は安全側で `pass` 扱い + raw_action を保持する。
    """
    raw = action_str if isinstance(action_str, str) else str(action_str)
    if ":" in raw:
        prefix, _, tile_str = raw.partition(":")
    else:
        prefix, tile_str = raw, ""
    prefix = prefix.strip()
    tile_str = tile_str.strip()

    action_type = _ACTION_TYPE_MAP.get(prefix, "pass")
    parsed: dict[str, Any] = {
        "action_type": action_type,
        "raw_action": raw,
    }
    if tile_str:
        # policy は called pai のみを emit するため、tile_str はそのまま 1 牌として
        # 採用する。赤 5 ("5mr") も丸ごと尊重する。
        parsed["tile"] = tile_str
    label = _ACTION_LABEL_MAP.get(prefix)
    if label:
        parsed["action_label"] = label
    return parsed


def make_primary_label(parsed: dict[str, Any]) -> str:
    """`parse_mjai_action` の結果から日本語の `primary_label` を組み立てる。

    Rust `src-tauri/src/monitor.rs:748` の `format_primary_label` と
    完全に同じマッピングを Python 側で再現する。
    """
    action_type = parsed.get("action_type", "pass")
    if action_type == "discard":
        tile = parsed.get("tile")
        if tile:
            return f"{tile} を切る"
        return "打牌"
    label = parsed.get("action_label")
    if isinstance(label, str) and label:
        return label
    # action_label が無い (未知 prefix の pass フォールバック等) ときは
    # ActionType 別の既定表記に倒す。
    return {
        "riichi": "リーチ",
        "tsumo": "ツモ",
        "ron": "ロン",
        "pon": "ポン",
        "chi": "チー",
        "kan": "カン",
        "pass": "スルー",
    }.get(action_type, "スルー")


def _build_candidate(action: str, probability: float, q_value: float) -> dict[str, Any]:
    parsed = parse_mjai_action(action)
    candidate: dict[str, Any] = {
        "action_type": parsed["action_type"],
        "expected_value": float(q_value),
        "probability": float(probability),
        # S-03 デバッグビュー用の補足文字列。P/Q を 1 行で確認できるようにする。
        "detail": f"P={probability:.2f}, Q={q_value:+.2f}",
    }
    if "tile" in parsed:
        candidate["tile"] = parsed["tile"]
    if "action_label" in parsed:
        candidate["action_label"] = parsed["action_label"]
    return candidate


def format_inference_result(
    policy: dict[str, float],
    value: float,
    q_values: dict[str, float],
    *,
    top_k: int = 5,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Mortal の raw 出力を `InferenceResult` 互換 dict に整形する。

    - policy を確率降順でソートし、`_LEGAL_POLICY_THRESHOLD` 未満を除外。
    - 先頭 `top_k` 件を candidates として返す。
    - `recommended` = candidates[0] (= policy argmax)。
    - `primary_label` は recommended から生成。
    - 返値には raw `policy` / `value` / `q_values` も含める (デバッグ用)。
      フロント `InferenceResult` 型は未知キーを silently drop するため安全。
    """
    if not isinstance(policy, dict) or not policy:
        raise ValueError("policy must be a non-empty dict[str, float]")
    if not isinstance(q_values, dict):
        raise ValueError("q_values must be dict[str, float]")
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    # 浮動小数の比較で同確率タイが起きたとき、再現性のため action 名で 2 次ソート。
    sorted_actions = sorted(
        (
            (action, float(prob))
            for action, prob in policy.items()
            if isinstance(action, str) and float(prob) >= _LEGAL_POLICY_THRESHOLD
        ),
        key=lambda kv: (-kv[1], kv[0]),
    )
    top = sorted_actions[:top_k]
    candidates = [
        _build_candidate(action, prob, float(q_values.get(action, 0.0))) for action, prob in top
    ]
    if not candidates:
        raise ValueError("policy contains no legal actions above threshold")

    recommended = candidates[0]
    primary_label = make_primary_label(parse_mjai_action(top[0][0]))

    return {
        "recommended": recommended,
        "candidates": candidates,
        "primary_label": primary_label,
        # raw fields — Rust 側は ignore するが、SQLite inference_log (D6/E3) や
        # 単体テストのデバッグで使うため敢えて出力する。
        "policy": dict(policy),
        "value": float(value),
        "q_values": dict(q_values),
        "timestamp": timestamp or datetime.now(UTC).isoformat(),
    }
