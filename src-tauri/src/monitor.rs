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

const SMOKE_PING_ID: i64 = 0;

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

    let stop_flag = Arc::new(AtomicBool::new(false));
    let stop_for_thread = stop_flag.clone();
    // ping/pong は recv_line がブロッキングなため監視スレッド内で行い、
    // start_monitoring 自体は即座に MonitorHandle を返せるようにする。
    // こうしておけばスモークテストが応答しない場合でも、フロントから
    // stop_monitoring を呼ぶことで kill 経由で復帰できる。
    let recognition_for_thread = recognition.clone();
    let mortal_for_thread = mortal.clone();

    let join = std::thread::spawn(move || {
        info!(target: "monitor", "monitor loop started");
        run_smoke_ping(&recognition_for_thread, "recognition");
        run_smoke_ping(&mortal_for_thread, "mortal");
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

/// 監視スレッド内で 1 サイクルの ping/pong を流す。
/// 失敗時は warning ログのみ吐いて呼び出し元へは通知しない
/// (ループ自体は継続させ、復旧は monitor::stop に任せる)。
fn run_smoke_ping(proc: &PythonProcess, label: &str) {
    if let Err(e) = proc.send_line(&format!(r#"{{"type":"ping","id":{}}}"#, SMOKE_PING_ID)) {
        warn!(target: "monitor", "smoke ping send failed [{}]: {}", label, e);
        return;
    }
    let resp = match proc.recv_line() {
        Ok(s) => s,
        Err(e) => {
            warn!(target: "monitor", "smoke pong recv failed [{}]: {}", label, e);
            return;
        }
    };
    let trimmed = resp.trim();
    match validate_pong(trimmed) {
        Ok(()) => info!(target: "monitor", "smoke ping/pong [{}]: ok ({})", label, trimmed),
        Err(reason) => warn!(
            target: "monitor",
            "smoke ping/pong [{}]: {} (raw: {})",
            label, reason, trimmed
        ),
    }
}

/// レスポンスが期待通りの `{"type":"pong","id":SMOKE_PING_ID}` 形式かを検証する。
/// プロセスが起動直後に無関係なログを stdout に流した場合や、`error` 応答を
/// 返した場合に、誤って成功と判断しないようにする。
fn validate_pong(line: &str) -> Result<(), String> {
    let parsed: serde_json::Value =
        serde_json::from_str(line).map_err(|e| format!("invalid json: {}", e))?;
    match parsed.get("type").and_then(|v| v.as_str()) {
        Some("pong") => {}
        Some(other) => return Err(format!("unexpected type: {}", other)),
        None => return Err("missing 'type' field".into()),
    }
    match parsed.get("id").and_then(|v| v.as_i64()) {
        Some(id) if id == SMOKE_PING_ID => Ok(()),
        Some(other) => Err(format!(
            "id mismatch (expected {}, got {})",
            SMOKE_PING_ID, other
        )),
        None => Err("missing or non-integer 'id' field".into()),
    }
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
