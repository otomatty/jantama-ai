// 雀魂AIアシスタント Tauri バックエンド
//
// PRD §8.1 (全体構成図) に従い、以下を担う:
// - 画面キャプチャ (xcap)
// - Pythonサブプロセス管理 (recognition, mortal の 2 プロセス)
// - SQLite 経由のデータ永続化
// - 設定ファイル (tauri-plugin-store)

mod capture;
mod commands;
mod monitor;
mod python_proc;
mod types;

use std::sync::Mutex;
use tauri::Manager;

/// アプリ起動時に共有する状態。
pub struct AppState {
    /// 監視ループの停止フラグ。Some なら監視ループが走っている。
    pub monitor_handle: Mutex<Option<monitor::MonitorHandle>>,
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,jantama_ai_lib=debug".into()),
        )
        .init();

    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_store::Builder::new().build())
        .plugin(
            tauri_plugin_sql::Builder::default()
                .add_migrations("sqlite:jantama-ai.db", db_migrations())
                .build(),
        )
        .setup(|app| {
            app.manage(AppState {
                monitor_handle: Mutex::new(None),
            });
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            commands::list_capture_windows,
            commands::load_settings,
            commands::save_settings,
            commands::start_monitoring,
            commands::stop_monitoring,
            commands::run_stub_inference,
            commands::capture_window_for_calibration,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

/// PRD §6 のテーブル定義。tauri-plugin-sql のマイグレーション機構で適用する。
fn db_migrations() -> Vec<tauri_plugin_sql::Migration> {
    vec![tauri_plugin_sql::Migration {
        version: 1,
        description: "create initial tables",
        sql: include_str!("../migrations/001_initial.sql"),
        kind: tauri_plugin_sql::MigrationKind::Up,
    }]
}
