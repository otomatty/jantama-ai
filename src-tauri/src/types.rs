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
}
