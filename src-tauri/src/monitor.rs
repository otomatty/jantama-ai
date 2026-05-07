// 監視ループ
//
// PRD §3 メインジャーニー: ユーザーが「監視ON」を押下すると、
// 別スレッドで以下を継続的に実行する。
//
//   1. 画面キャプチャ (Rust)
//   2. 認識プロセスへ送信 → 天鳳 JSON 受信 (Python A)
//   3. 推論プロセスへ送信 → 推奨候補 JSON 受信 (Python B)
//   4. Tauri Event でフロントエンドへ結果を emit
//
// 現状はスケルトンのみ。実装は Phase B 以降に詳細化する。

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use tracing::{info, warn};

pub struct MonitorHandle {
    pub stop_flag: Arc<AtomicBool>,
    pub join: Option<JoinHandle<()>>,
}

impl MonitorHandle {
    pub fn stop(&mut self) {
        info!(target: "monitor", "stopping monitor loop");
        self.stop_flag.store(true, Ordering::SeqCst);
        if let Some(j) = self.join.take() {
            let _ = j.join();
        }
    }
}

impl Drop for MonitorHandle {
    fn drop(&mut self) {
        self.stop();
    }
}

/// 監視ループを起動する。
///
/// MVP スケルトン: 1 秒ごとに「監視中」ログを吐くだけ。
/// Phase B でキャプチャ → 認識 → 推論を実装する。
pub fn start(_capture_target: String) -> MonitorHandle {
    let stop_flag = Arc::new(AtomicBool::new(false));
    let stop_for_thread = stop_flag.clone();

    let join = std::thread::spawn(move || {
        info!(target: "monitor", "monitor loop started");
        while !stop_for_thread.load(Ordering::SeqCst) {
            // TODO(Phase B): capture → recognize → infer → emit event
            std::thread::sleep(std::time::Duration::from_millis(500));
        }
        warn!(target: "monitor", "monitor loop terminated");
    });

    MonitorHandle {
        stop_flag,
        join: Some(join),
    }
}
