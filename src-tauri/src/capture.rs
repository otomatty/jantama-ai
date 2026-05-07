// 画面キャプチャモジュール
//
// PRD §8.2 Tauri (Rust側) - 画面キャプチャ: `xcap` クレート

use crate::types::CaptureWindow;
use std::error::Error;
use xcap::Window;

pub type CaptureError = Box<dyn Error + Send + Sync>;

/// 起動中のすべてのウィンドウを列挙する。
/// 空タイトルのウィンドウは除外する。
pub fn list_windows() -> Result<Vec<CaptureWindow>, CaptureError> {
    let windows = Window::all()?;
    let mut result = Vec::new();
    for w in windows {
        let title = w.title();
        if title.trim().is_empty() {
            continue;
        }
        result.push(CaptureWindow {
            id: w.id().to_string(),
            title: title.to_string(),
            app_name: Some(w.app_name().to_string()),
            is_minimized: w.is_minimized(),
        });
    }
    Ok(result)
}

/// 指定 ID のウィンドウをキャプチャして RGBA バイト列 (image::RgbaImage) を返す。
///
/// MVP では Python 認識プロセスへ stdin で渡す前提。
/// PNG 化したい場合は `image` クレート (xcap が再エクスポート) で encode する。
#[allow(dead_code)]
pub fn capture_window(window_id: &str) -> Result<xcap::image::RgbaImage, CaptureError> {
    let windows = Window::all()?;
    let w = windows
        .into_iter()
        .find(|w| w.id().to_string() == window_id)
        .ok_or_else(|| -> CaptureError {
            format!("ウィンドウが見つかりません: {}", window_id).into()
        })?;
    let image = w.capture_image()?;
    Ok(image)
}
