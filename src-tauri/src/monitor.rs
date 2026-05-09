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
    // ping/pong は recv_line がブロッキングなため、監視ループとは別スレッドで
    // 並行に走らせる。こうしておけば
    //   - start_monitoring は即座に MonitorHandle を返せる
    //   - recognition と mortal のスモークが互いをブロックしない
    //   - 監視ループ本体の 500ms ポーリング (stop_flag 監視) も止まらない
    // スモークが応答しない場合でも、stop 経由の kill() で recv_line が
    // Err(Terminated) を返してスレッドは自然終了する。
    spawn_smoke_ping(recognition.clone(), "recognition");
    spawn_smoke_ping(mortal.clone(), "mortal");

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

/// 1 サイクルの ping/pong を fire-and-forget で流す専用ワーカーを起動する。
/// recv_line がブロッキングなので独立スレッドに切り出しておけば、
/// recognition / mortal どちらかが応答しなくても他方や監視ループ本体の
/// 500ms ポーリングは止まらない。スレッドへ渡した Arc が drop されることで
/// MonitorHandle::stop の kill() が伝播し、recv_line が Err(Terminated) で
/// 戻ってこのスレッドも自然終了する。
fn spawn_smoke_ping(proc: Arc<PythonProcess>, label: &'static str) {
    std::thread::spawn(move || {
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
    });
}

/// レスポンスが期待通りの `{"type":"pong","id":SMOKE_PING_ID}` であることを
/// 厳密に検証する。プロトコルは Rust ↔ Python 間の内部仕様なので、未知の
/// 追加フィールドや余分なキーは「想定外の応答」として扱い、smoke を
/// 成功にしない。これによりプロトコルドリフトを早期に検知できる。
fn validate_pong(line: &str) -> Result<(), String> {
    let parsed: serde_json::Value =
        serde_json::from_str(line).map_err(|e| format!("invalid json: {}", e))?;
    let obj = parsed
        .as_object()
        .ok_or_else(|| "response is not a JSON object".to_string())?;
    if obj.len() != 2 || !obj.contains_key("type") || !obj.contains_key("id") {
        return Err("response must contain exactly 'type' and 'id'".into());
    }
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
