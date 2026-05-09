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
// Phase B (本 PR): recognition / mortal の 2 プロセスを起動し、
// 起動直後に ping/pong スモークテストを 1 サイクル流す所までを実装する。
// キャプチャ → 認識 → 推論の本ループは後続 Issue (B2/B3) で配線する。

use crate::python_proc::{PythonProcError, PythonProcess};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use thiserror::Error;
use tracing::{info, warn};

#[derive(Debug, Error)]
pub enum MonitorError {
    #[error("認識プロセス起動失敗: {0}")]
    RecognitionSpawn(PythonProcError),
    #[error("Mortal プロセス起動失敗: {0}")]
    MortalSpawn(PythonProcError),
}

/// `monitor::start` の引数。フロント設定や開発/本番ビルドの差異を
/// commands 側で吸収して渡す。
pub struct MonitorConfig {
    pub capture_target: String,
    pub recognition_program: PathBuf,
    pub recognition_args: Vec<String>,
    pub mortal_program: PathBuf,
    pub mortal_args: Vec<String>,
    pub python_cwd: Option<PathBuf>,
}

pub struct MonitorHandle {
    pub stop_flag: Arc<AtomicBool>,
    pub recognition: Arc<PythonProcess>,
    pub mortal: Arc<PythonProcess>,
    pub join: Option<JoinHandle<()>>,
}

impl MonitorHandle {
    pub fn stop(&mut self) {
        info!(target: "monitor", "stopping monitor loop");
        self.stop_flag.store(true, Ordering::SeqCst);
        // 監視スレッドが recv_line でブロックしている可能性があるため、
        // join より先に Python プロセスを kill して stdout を閉じる。
        self.recognition.kill();
        self.mortal.kill();
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
/// recognition / mortal の 2 プロセスを spawn し、起動直後に
/// ping/pong スモークテストを 1 サイクルだけ流す。失敗しても
/// プロセス自体は維持し warning ログのみ出して継続する
/// (uv の依存解決などで初回応答が遅れるケースを許容するため)。
pub fn start(config: MonitorConfig) -> Result<MonitorHandle, MonitorError> {
    let MonitorConfig {
        capture_target,
        recognition_program,
        recognition_args,
        mortal_program,
        mortal_args,
        python_cwd,
    } = config;

    info!(target: "monitor", "starting monitor for target='{}'", capture_target);

    let rec_args: Vec<&str> = recognition_args.iter().map(String::as_str).collect();
    let mortal_args_ref: Vec<&str> = mortal_args.iter().map(String::as_str).collect();

    let recognition = Arc::new(
        PythonProcess::spawn(
            "recognition",
            recognition_program.as_path(),
            &rec_args,
            python_cwd.as_deref(),
        )
        .map_err(MonitorError::RecognitionSpawn)?,
    );
    let mortal = Arc::new(
        PythonProcess::spawn(
            "mortal",
            mortal_program.as_path(),
            &mortal_args_ref,
            python_cwd.as_deref(),
        )
        .map_err(MonitorError::MortalSpawn)?,
    );

    if let Err(e) = ping_pong(&recognition, "recognition") {
        warn!(target: "monitor", "recognition smoke ping failed: {}", e);
    }
    if let Err(e) = ping_pong(&mortal, "mortal") {
        warn!(target: "monitor", "mortal smoke ping failed: {}", e);
    }

    let stop_flag = Arc::new(AtomicBool::new(false));
    let stop_for_thread = stop_flag.clone();

    let join = std::thread::spawn(move || {
        info!(target: "monitor", "monitor loop started");
        while !stop_for_thread.load(Ordering::SeqCst) {
            // TODO(Phase B2/B3): capture → recognize → infer → emit event
            std::thread::sleep(std::time::Duration::from_millis(500));
        }
        warn!(target: "monitor", "monitor loop terminated");
    });

    Ok(MonitorHandle {
        stop_flag,
        recognition,
        mortal,
        join: Some(join),
    })
}

fn ping_pong(proc: &PythonProcess, label: &str) -> Result<(), PythonProcError> {
    proc.send_line(r#"{"type":"ping","id":0}"#)?;
    let resp = proc.recv_line()?;
    info!(target: "monitor", "smoke ping/pong [{}]: {}", label, resp.trim());
    Ok(())
}

/// 開発時に `python/pyproject.toml` がある場所を解決する。
/// `cargo tauri dev` ではバイナリが `src-tauri/target/debug/` に生成されるため、
/// cwd か exe からの上位ディレクトリ走査で `python/` を探す。
pub fn resolve_python_project_dir() -> Option<PathBuf> {
    let mut candidates: Vec<PathBuf> = Vec::new();

    if let Ok(cwd) = std::env::current_dir() {
        candidates.push(cwd);
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            candidates.push(parent.to_path_buf());
        }
    }

    for start in candidates {
        let mut cur: Option<&Path> = Some(start.as_path());
        while let Some(dir) = cur {
            let candidate = dir.join("python");
            if candidate.join("pyproject.toml").is_file() {
                return Some(candidate);
            }
            cur = dir.parent();
        }
    }
    None
}
