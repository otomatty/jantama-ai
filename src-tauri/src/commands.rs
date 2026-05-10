// Tauri invoke コマンド (フロントエンドからの RPC エントリポイント)
//
// 命名規則: フロント側 src/lib/tauriCommands.ts と同名で揃える。

use crate::types::{
    ActionType, AppSettings, CaptureWindow, InferenceResult, RecommendationCandidate,
};
use crate::{capture, monitor, AppState};
use chrono::Utc;
use serde::Serialize;
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

    // dev/release それぞれの起動方式は `PythonProcess::spawn_recognition` /
    // `spawn_mortal` が `resolve_python_command` 経由で吸収する。
    // ここでは設定値だけ渡す。空文字列なら mortal は `--stub` で起動する。
    let mortal_model_path = settings.mortal_model_path.clone().unwrap_or_default();

    let config = monitor::MonitorConfig {
        capture_target: target,
        mortal_model_path,
        inference_backend: settings.inference_backend,
        roi_calibration: settings.roi_calibration.clone(),
    };

    let handle = monitor::start(app.clone(), config).map_err(|e| e.to_string())?;
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

/// 設定画面の ROI キャリブレーション UI で、対象ウィンドウのスクリーンショットを
/// 1 枚取得するためのコマンド (issue #10)。
///
/// 監視ループの 1Hz ポーリングと違い、これは「ボタン押下時に 1 度だけ」呼ばれる
/// 想定なので、画像サイズ + base64 PNG をそのままフロントへ返す。フロントは
/// `<canvas>` に描画してドラッグ操作で矩形を取らせる。
#[derive(Debug, Clone, Serialize)]
pub struct CalibrationCapture {
    pub width: u32,
    pub height: u32,
    pub image_b64: String,
}

#[tauri::command]
pub async fn capture_window_for_calibration(
    window_id: String,
) -> Result<CalibrationCapture, String> {
    let id = window_id.trim();
    if id.is_empty() {
        return Err("キャプチャ対象ウィンドウが選択されていません".into());
    }
    let img = capture::capture_window(id).map_err(|e| e.to_string())?;
    let (width, height) = img.dimensions();
    let image_b64 = capture::encode_png_base64(&img).map_err(|e| e.to_string())?;
    Ok(CalibrationCapture {
        width,
        height,
        image_b64,
    })
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
