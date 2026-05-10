// 画面キャプチャモジュール
//
// PRD §8.2 Tauri (Rust側) - 画面キャプチャ: `xcap` クレート

use crate::types::CaptureWindow;
use base64::engine::general_purpose::STANDARD as B64;
use base64::Engine as _;
use std::error::Error;
use std::time::Instant;
use tracing::debug;
use xcap::image::codecs::png::{CompressionType, FilterType, PngEncoder};
use xcap::image::{ExtendedColorType, ImageEncoder, RgbaImage};
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
/// PNG 化したい場合は `encode_png_base64` を使う。
pub fn capture_window(window_id: &str) -> Result<RgbaImage, CaptureError> {
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

/// `RgbaImage` を PNG にエンコードし base64 (STANDARD) 文字列で返す。
///
/// Python 認識プロセスへ送る `{"type":"frame","image_b64":"..."}` の payload を
/// 組み立てるためのユーティリティ。PRD §7.1 の目安は 1920x1080 を 50ms 以内なので、
/// `CompressionType::Fast` + `FilterType::NoFilter` でエンコード時間を優先する
/// (1Hz の監視ループでサイズより速度が支配的)。
pub fn encode_png_base64(img: &RgbaImage) -> Result<String, CaptureError> {
    let started = Instant::now();
    let (width, height) = img.dimensions();
    let mut png_bytes = Vec::new();
    PngEncoder::new_with_quality(&mut png_bytes, CompressionType::Fast, FilterType::NoFilter)
        .write_image(img.as_raw(), width, height, ExtendedColorType::Rgba8)?;
    let encoded = B64.encode(&png_bytes);
    debug!(
        target: "capture",
        "encode_png_base64: {}x{} png={}B b64={}B in {}us",
        width,
        height,
        png_bytes.len(),
        encoded.len(),
        started.elapsed().as_micros()
    );
    Ok(encoded)
}

#[cfg(test)]
mod tests {
    use super::*;
    use xcap::image::{load_from_memory_with_format, ImageFormat, Rgba};

    #[test]
    fn encode_png_base64_roundtrip_red_10x10() {
        let img = RgbaImage::from_pixel(10, 10, Rgba([255, 0, 0, 255]));
        let encoded = encode_png_base64(&img).expect("encode_png_base64 failed");

        let decoded_bytes = B64.decode(&encoded).expect("base64 decode failed");
        let decoded = load_from_memory_with_format(&decoded_bytes, ImageFormat::Png)
            .expect("png decode failed")
            .to_rgba8();

        assert_eq!(decoded.dimensions(), (10, 10));
        assert_eq!(decoded.get_pixel(0, 0), &Rgba([255, 0, 0, 255]));
        assert_eq!(decoded.get_pixel(9, 9), &Rgba([255, 0, 0, 255]));
    }
}
