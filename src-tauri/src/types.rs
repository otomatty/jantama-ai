// PRD §6 (データモデル) と src/types/index.ts に対応する Rust 型定義。

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaptureWindow {
    pub id: String,
    pub title: String,
    pub app_name: Option<String>,
    pub is_minimized: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DataRetentionDays {
    pub inference_log: u32,
    pub tile_image: u32,
    pub error_log: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WindowPosition {
    pub x: i32,
    pub y: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WindowSize {
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InferenceBackend {
    #[default]
    Rocm,
    Cpu,
}

/// ROI 矩形 (PRD §9 / issue #10)。
///
/// 比率指定 (0.0〜1.0) で保存する。キャプチャ解像度が変わっても追従できる。
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct RoiRect {
    pub x: f64,
    pub y: f64,
    pub w: f64,
    pub h: f64,
}

/// 河 4 領域 (自家・下家・対面・上家)。
///
/// 未指定領域は `null` を明示的に書き出す (= `skip_serializing_if` を付けない)。
/// フロント側 (`src/types/index.ts`) の `RoiCalibration` は各フィールドを
/// `RoiRect | null` で受け取るため、欠落フィールドだと `undefined` になり
/// `=== null` 判定が崩れる (Phase C で recognition プロセスが領域有無を
/// 判別する際にも同じ問題になる)。
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RiverRois {
    #[serde(default, rename = "self")]
    pub self_seat: Option<RoiRect>,
    #[serde(default)]
    pub right: Option<RoiRect>,
    #[serde(default)]
    pub across: Option<RoiRect>,
    #[serde(default)]
    pub left: Option<RoiRect>,
}

/// 副露 (鳴き) 4 領域 (自家・下家・対面・上家)。issue #14。
///
/// 構造は `RiverRois` と同じ (4 家分の `Option<RoiRect>` をネスト)。
/// `self` は Rust の予約語のため `self_seat` で保持し、JSON 上は `self` キーで
/// 永続化する。`#[serde(default)]` により旧 settings.json (本フィールド欠落)
/// も全 None で読み戻せる (互換性)。
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MeldRois {
    #[serde(default, rename = "self")]
    pub self_seat: Option<RoiRect>,
    #[serde(default)]
    pub right: Option<RoiRect>,
    #[serde(default)]
    pub across: Option<RoiRect>,
    #[serde(default)]
    pub left: Option<RoiRect>,
}

/// 認識用 ROI キャリブレーション (issue #10 / #12 / #14)。
///
/// issue #12 で `scores` (4 家分の点棒を 1 つの領域に並べた帯) と
/// `turn_counter` (巡目カウンタの数字) を追加。issue #14 で `melds`
/// (4 家分の副露領域) を追加。いずれも `#[serde(default)]` なので旧
/// settings.json (これらフィールド欠落) でも `None` で読み戻せる。
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RoiCalibration {
    #[serde(default)]
    pub hand: Option<RoiRect>,
    #[serde(default)]
    pub doras: Option<RoiRect>,
    #[serde(default)]
    pub rivers: RiverRois,
    #[serde(default)]
    pub melds: MeldRois,
    #[serde(default)]
    pub round_info: Option<RoiRect>,
    #[serde(default)]
    pub self_wind: Option<RoiRect>,
    #[serde(default)]
    pub scores: Option<RoiRect>,
    #[serde(default)]
    pub turn_counter: Option<RoiRect>,
    /// 自分の手番でアクションボタン (チー/ポン/カン/リーチ/ツモ/ロン/パス) が
    /// 表示される領域 (issue #15)。雀魂では使えるボタンだけが右詰めで現れるので、
    /// ROI は「最も多くのボタンが並んだ時の最大幅」をカバーするよう広めに切る。
    #[serde(default)]
    pub action_buttons: Option<RoiRect>,
    /// 自家ネームプレート周りに出る思考タイマー (円形リング) の領域 (issue #15)。
    /// `TurnRecognizer.detect_timer_active` が HSV inRange で占有率を見る。
    #[serde(default)]
    pub turn_timer: Option<RoiRect>,
}

fn default_show_llm_reason() -> bool {
    true
}

fn default_show_danger_safe() -> bool {
    true
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppSettings {
    pub capture_target_window_id: Option<String>,
    pub capture_target_window_title: Option<String>,
    pub mortal_model_path: Option<String>,
    #[serde(default)]
    pub inference_backend: InferenceBackend,
    #[serde(default = "default_show_llm_reason")]
    pub show_llm_reason: bool,
    #[serde(default = "default_show_danger_safe")]
    pub show_danger_safe: bool,
    pub window_position: Option<WindowPosition>,
    pub window_size: Option<WindowSize>,
    pub data_retention_days: DataRetentionDays,
    #[serde(default)]
    pub hotkey_settings: Option<serde_json::Value>,
    /// ROI キャリブレーション (issue #10)。古い設定ファイルとの互換性のため
    /// `default` でフォールバックする (= 全領域 None)。
    #[serde(default)]
    pub roi_calibration: RoiCalibration,
}

impl Default for AppSettings {
    fn default() -> Self {
        Self {
            capture_target_window_id: None,
            capture_target_window_title: None,
            mortal_model_path: None,
            inference_backend: InferenceBackend::default(),
            show_llm_reason: true,
            show_danger_safe: true,
            window_position: None,
            window_size: None,
            data_retention_days: DataRetentionDays {
                inference_log: 30,
                tile_image: 7,
                error_log: 90,
            },
            hotkey_settings: None,
            roi_calibration: RoiCalibration::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActionType {
    Discard,
    Riichi,
    Chi,
    Pon,
    Kan,
    Ron,
    Tsumo,
    Pass,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecommendationCandidate {
    pub tile: Option<String>,
    pub action_type: ActionType,
    pub expected_value: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub action_label: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub probability: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DangerTile {
    pub tile: String,
    pub level: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceResult {
    pub recommended: RecommendationCandidate,
    pub candidates: Vec<RecommendationCandidate>,
    pub timestamp: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub primary_label: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub danger: Option<Vec<DangerTile>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub safe: Option<Vec<String>>,
}

/// PRD §6 / src/types/index.ts の `GameBoardSummary` に対応。
/// recognition プロセスが返す `tenhou_json` から抽出してフロントへ届ける。
///
/// issue #15: `my_turn` と `available_actions` を保持する。フロントの
/// `MainBody` ゲートと、Rust の `should_skip_inference` 判定の両方で使う。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameBoardSummary {
    pub hand: Vec<String>,
    pub self_wind: String,
    pub round_wind: String,
    pub turn: u32,
    pub dora_indicators: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub score: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub round_label: Option<String>,
    /// 「今自分が選択肢を持っているか」(打牌 or 鳴き or 和了 など)。
    /// `Some(false)` のとき Rust 監視ループは mortal 推論をスキップする。
    /// `None` (recognition 側がフィールド未対応 = 旧スキーマ) のときはフェイル
    /// セーフで「mortal を呼ぶ側」に倒す (Codex P1 / CodeRabbit major on PR #47)。
    #[serde(default)]
    pub my_turn: Option<bool>,
    /// 取り得るアクション (`discard` / `riichi` / `chi` / `pon` / `kan` /
    /// `ron` / `tsumo` / `pass`)。`Some(空配列)` なら recommend 表示を行わない。
    /// `None` は旧スキーマ互換 (= フィールド欠落)。
    #[serde(default)]
    pub available_actions: Option<Vec<String>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// issue #10: `roi_calibration` 未指定の旧 JSON を読み戻しても
    /// `Default` で全領域 None になることを確認する (互換性レグレッションテスト)。
    #[test]
    fn app_settings_deserializes_legacy_json_without_roi_calibration() {
        let legacy = serde_json::json!({
            "capture_target_window_id": "1",
            "capture_target_window_title": "Mahjong Soul",
            "mortal_model_path": "/path/to/model",
            "window_position": null,
            "window_size": null,
            "data_retention_days": {
                "inference_log": 30,
                "tile_image": 7,
                "error_log": 90
            }
        });
        let parsed: AppSettings = serde_json::from_value(legacy).expect("legacy parse");
        assert!(parsed.roi_calibration.hand.is_none());
        assert!(parsed.roi_calibration.rivers.self_seat.is_none());
        assert!(parsed.roi_calibration.rivers.right.is_none());
    }

    /// issue #12: scores / turn_counter フィールドが旧 settings.json (これらが
    /// 存在しない) からも `None` で読み戻せる (#[serde(default)] のレグレッション)。
    #[test]
    fn roi_calibration_deserializes_without_scores_and_turn_counter() {
        let legacy = serde_json::json!({
            "hand": null,
            "doras": null,
            "rivers": {},
            "round_info": null,
            "self_wind": null,
        });
        let parsed: RoiCalibration = serde_json::from_value(legacy).expect("legacy parse");
        assert!(parsed.scores.is_none());
        assert!(parsed.turn_counter.is_none());
    }

    /// issue #15: action_buttons / turn_timer フィールドが旧 settings.json
    /// (これらが存在しない) からも `None` で読み戻せる。
    #[test]
    fn roi_calibration_deserializes_without_action_buttons_and_turn_timer() {
        let legacy = serde_json::json!({
            "hand": null,
            "doras": null,
            "rivers": {},
            "round_info": null,
            "self_wind": null,
            "scores": null,
            "turn_counter": null,
        });
        let parsed: RoiCalibration = serde_json::from_value(legacy).expect("legacy parse");
        assert!(parsed.action_buttons.is_none());
        assert!(parsed.turn_timer.is_none());
    }

    /// issue #10: river 領域のキーは `self` で永続化される
    /// (Rust の予約語回避で `self_seat` にしているが JSON 上は `self`)。
    #[test]
    fn river_rois_serialize_with_self_key() {
        let mut roi = RoiCalibration::default();
        roi.rivers.self_seat = Some(RoiRect {
            x: 0.1,
            y: 0.2,
            w: 0.3,
            h: 0.4,
        });
        let json = serde_json::to_value(&roi).unwrap();
        let rivers = json.get("rivers").expect("rivers field");
        assert!(rivers.get("self").is_some());
        assert!(rivers.get("self_seat").is_none());

        let round_trip: RoiCalibration = serde_json::from_value(json).unwrap();
        assert_eq!(round_trip.rivers.self_seat.unwrap().x, 0.1);
    }

    /// issue #14: melds 領域も同様に JSON 上は `self` キーで永続化される。
    /// melds フィールド欠落時の Default フォールバックも合わせて確認。
    #[test]
    fn meld_rois_serialize_with_self_key() {
        let mut roi = RoiCalibration::default();
        roi.melds.self_seat = Some(RoiRect {
            x: 0.5,
            y: 0.6,
            w: 0.2,
            h: 0.1,
        });
        let json = serde_json::to_value(&roi).unwrap();
        let melds = json.get("melds").expect("melds field");
        assert!(melds.get("self").is_some());
        assert!(melds.get("self_seat").is_none());

        let round_trip: RoiCalibration = serde_json::from_value(json).unwrap();
        assert_eq!(round_trip.melds.self_seat.unwrap().x, 0.5);
    }

    /// issue #14: melds フィールド欠落の旧 settings.json も全 None で読み戻せる。
    #[test]
    fn roi_calibration_deserializes_without_melds() {
        let legacy = serde_json::json!({
            "hand": null,
            "doras": null,
            "rivers": {},
            "round_info": null,
            "self_wind": null,
        });
        let parsed: RoiCalibration = serde_json::from_value(legacy).expect("legacy parse");
        assert!(parsed.melds.self_seat.is_none());
        assert!(parsed.melds.right.is_none());
        assert!(parsed.melds.across.is_none());
        assert!(parsed.melds.left.is_none());
    }
}
