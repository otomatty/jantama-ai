// Tauri invoke コマンド (フロントエンドからの RPC エントリポイント)
//
// 命名規則: フロント側 src/lib/tauriCommands.ts と同名で揃える。

use crate::types::{
    ActionType, AppSettings, CaptureWindow, InferenceResult, RecommendationCandidate,
};
use crate::{capture, monitor, AppState};
use chrono::Utc;
use tauri::{AppHandle, Manager, State};
use tauri_plugin_store::StoreExt;

const SETTINGS_STORE_FILE: &str = "settings.json";
const SETTINGS_KEY: &str = "app_settings";

#[tauri::command]
pub async fn list_capture_windows() -> Result<Vec<CaptureWindow>, String> {
    capture::list_windows().map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn load_settings(app: AppHandle) -> Result<Option<AppSettings>, String> {
    let store = app
        .store(SETTINGS_STORE_FILE)
        .map_err(|e| e.to_string())?;
    let value = store.get(SETTINGS_KEY);
    if let Some(v) = value {
        let parsed: AppSettings =
            serde_json::from_value(v).map_err(|e| format!("settings parse error: {}", e))?;
        Ok(Some(parsed))
    } else {
        Ok(None)
    }
}

#[tauri::command]
pub async fn save_settings(app: AppHandle, settings: AppSettings) -> Result<(), String> {
    let store = app
        .store(SETTINGS_STORE_FILE)
        .map_err(|e| e.to_string())?;
    store.set(
        SETTINGS_KEY,
        serde_json::to_value(&settings).map_err(|e| e.to_string())?,
    );
    store.save().map_err(|e| e.to_string())?;
    Ok(())
}

#[tauri::command]
pub async fn start_monitoring(
    app: AppHandle,
    state: State<'_, AppState>,
) -> Result<(), String> {
    // 既に動作中なら何もしない
    {
        let guard = state.monitor_handle.lock().unwrap();
        if guard.is_some() {
            return Ok(());
        }
    }

    // 設定からキャプチャ対象を取得
    let settings = load_settings(app.clone())
        .await?
        .unwrap_or_default();
    let target = settings
        .capture_target_window_id
        .clone()
        .unwrap_or_default();

    let handle = monitor::start(target);
    *state.monitor_handle.lock().unwrap() = Some(handle);
    Ok(())
}

#[tauri::command]
pub async fn stop_monitoring(state: State<'_, AppState>) -> Result<(), String> {
    let mut guard = state.monitor_handle.lock().unwrap();
    if let Some(mut h) = guard.take() {
        h.stop();
    }
    Ok(())
}

#[tauri::command]
pub async fn run_stub_inference() -> Result<InferenceResult, String> {
    // PRD §5.2 のサンプル数値をそのまま返すスタブ
    Ok(InferenceResult {
        recommended: RecommendationCandidate {
            tile: Some("6m".into()),
            action_type: ActionType::Discard,
            expected_value: 0.32,
            detail: None,
        },
        candidates: vec![
            RecommendationCandidate {
                tile: Some("6m".into()),
                action_type: ActionType::Discard,
                expected_value: 0.32,
                detail: None,
            },
            RecommendationCandidate {
                tile: Some("9p".into()),
                action_type: ActionType::Discard,
                expected_value: 0.18,
                detail: None,
            },
            RecommendationCandidate {
                tile: Some("1z".into()),
                action_type: ActionType::Discard,
                expected_value: -0.05,
                detail: None,
            },
        ],
        timestamp: Utc::now().to_rfc3339(),
    })
}
