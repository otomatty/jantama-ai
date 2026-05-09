// Tauri invoke コマンド (フロントエンドからの RPC エントリポイント)
//
// 命名規則: フロント側 src/lib/tauriCommands.ts と同名で揃える。

use crate::types::{
    ActionType, AppSettings, CaptureWindow, InferenceResult, RecommendationCandidate,
};
use crate::{capture, monitor, AppState};
use chrono::Utc;
use std::path::PathBuf;
use tauri::{AppHandle, State};
use tauri_plugin_store::StoreExt;

const SETTINGS_STORE_FILE: &str = "settings.json";
const SETTINGS_KEY: &str = "app_settings";

#[tauri::command]
pub async fn list_capture_windows() -> Result<Vec<CaptureWindow>, String> {
    capture::list_windows().map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn load_settings(app: AppHandle) -> Result<Option<AppSettings>, String> {
    let store = app.store(SETTINGS_STORE_FILE).map_err(|e| e.to_string())?;
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
    let store = app.store(SETTINGS_STORE_FILE).map_err(|e| e.to_string())?;
    store.set(
        SETTINGS_KEY,
        serde_json::to_value(&settings).map_err(|e| e.to_string())?,
    );
    store.save().map_err(|e| e.to_string())?;
    Ok(())
}

#[tauri::command]
pub async fn start_monitoring(app: AppHandle, state: State<'_, AppState>) -> Result<(), String> {
    // 既に動作中なら何もしない
    {
        let guard = state.monitor_handle.lock().unwrap();
        if guard.is_some() {
            return Ok(());
        }
    }

    // 設定からキャプチャ対象と Mortal モデルパスを取得
    let settings = load_settings(app.clone()).await?.unwrap_or_default();
    let target = settings
        .capture_target_window_id
        .clone()
        .unwrap_or_default();

    let python_cwd = monitor::resolve_python_project_dir();

    // 開発デフォルト: `uv run jantama-recognition` / `uv run jantama-mortal`。
    // PyInstaller バンドル後は別途 .exe パスを差し込めるよう Vec<String> で持つ。
    let recognition_program: PathBuf = "uv".into();
    let recognition_args: Vec<String> = vec!["run".into(), "jantama-recognition".into()];

    let mortal_program: PathBuf = "uv".into();
    let mut mortal_args: Vec<String> = vec!["run".into(), "jantama-mortal".into()];
    match settings.mortal_model_path.as_deref() {
        Some(path) if !path.trim().is_empty() => {
            mortal_args.push("--model".into());
            mortal_args.push(path.to_string());
        }
        _ => {
            // モデル未設定時は Python 側の --stub に倒して MVP UI を確認可能にする。
            mortal_args.push("--stub".into());
        }
    }

    let config = monitor::MonitorConfig {
        capture_target: target,
        recognition_program,
        recognition_args,
        mortal_program,
        mortal_args,
        python_cwd,
    };

    let handle = monitor::start(config).map_err(|e| e.to_string())?;
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
    let mk = |tile: &str, ev: f64, prob: f64| RecommendationCandidate {
        tile: Some(tile.into()),
        action_type: ActionType::Discard,
        expected_value: ev,
        action_label: Some("打牌".into()),
        probability: Some(prob),
        detail: None,
    };
    Ok(InferenceResult {
        recommended: mk("6m", 0.32, 0.61),
        candidates: vec![
            mk("6m", 0.32, 0.61),
            mk("9p", 0.18, 0.22),
            mk("1z", -0.05, 0.11),
        ],
        timestamp: Utc::now().to_rfc3339(),
        primary_label: Some("6m を切る".into()),
        reason: None,
        danger: None,
        safe: None,
    })
}
