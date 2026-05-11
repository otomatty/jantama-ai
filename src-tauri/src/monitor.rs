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
// Phase B2 (本 PR): キャプチャ → recognition → mortal → `inference-result`
// emit の一気通貫パイプラインを 1Hz で回す。

use crate::capture;
use crate::python_proc::{PythonProcError, PythonProcess};
use crate::types::{
    ActionType, GameBoardSummary, InferenceBackend, InferenceResult, MeldRois,
    RecommendationCandidate, RiverRois, RoiCalibration, RoiRect,
};
use chrono::Utc;
use serde::Serialize;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};
use tauri::{AppHandle, Emitter, Runtime};
use thiserror::Error;
use tracing::{info, warn};

const SMOKE_PING_ID: i64 = 0;
/// 監視ループの 1 サイクル間隔。後で短縮可能 (issue #4)。
const FRAME_INTERVAL: Duration = Duration::from_secs(1);
/// 停止フラグへの応答性を保つために sleep を細切れにする粒度。
const SLEEP_SLICE: Duration = Duration::from_millis(100);
/// PRD §7.4 (信頼性要件) に基づく Python 応答待ちタイムアウト。
/// 1Hz の監視ループに対し 3 秒待っても応答が無ければ「認識失敗 / 推論失敗」
/// として次フレームへスキップする。
const PYTHON_RECV_TIMEOUT: Duration = Duration::from_secs(3);
/// 起動直後の ping/pong は `uv run` のロード時間を踏まえて長めに取る。
/// (1 サイクル目の本ループ前に 1 度だけ流すため、ループ全体の 1Hz には影響しない)
const SMOKE_RECV_TIMEOUT: Duration = Duration::from_secs(10);

#[derive(Debug, Error)]
pub enum MonitorError {
    #[error("監視設定が不正です: {0}")]
    InvalidConfig(String),
    #[error("認識プロセス起動失敗: {0}")]
    RecognitionSpawn(PythonProcError),
    #[error("Mortal プロセス起動失敗: {0}")]
    MortalSpawn(PythonProcError),
}

/// `monitor::start` の引数。フロント設定からキャプチャ対象とモデル設定だけ
/// 受け取り、Python サブプロセスの起動方法 (dev `uv run` / release バンドル exe)
/// は `PythonProcess::spawn_recognition` / `spawn_mortal` 側で吸収する。
pub struct MonitorConfig {
    pub capture_target: String,
    /// 空文字列なら mortal を `--stub` モードで起動する。
    pub mortal_model_path: String,
    pub inference_backend: InferenceBackend,
    /// ROI キャリブレーション (issue #10)。recognition プロセスは
    /// frame request の `roi_calibration` を見て領域を切り出す。
    pub roi_calibration: RoiCalibration,
}

/// `Arc<PythonProcess>` を差し替え可能にしたスロット。
///
/// ProcessDied 検知時に新しいプロセスへ載せ替えても、外部から
/// 共有されている `Arc<ProcessSlot>` のハンドルはそのまま使える。
/// `MonitorHandle::stop` 経由で殺しに来る場合と、監視スレッドが
/// 再起動で差し替える場合の両方を扱う。
pub struct ProcessSlot {
    inner: Mutex<Arc<PythonProcess>>,
}

impl ProcessSlot {
    fn new(proc: PythonProcess) -> Self {
        Self {
            inner: Mutex::new(Arc::new(proc)),
        }
    }

    /// 現在格納している `PythonProcess` への共有ハンドルを取得する。
    /// 監視スレッドはサイクルごとにこれを呼んで最新のプロセスへ I/O する。
    pub fn current(&self) -> Arc<PythonProcess> {
        self.inner.lock().unwrap().clone()
    }

    /// `stop_flag` をスロットの mutex 内で再確認しつつ差し替える。
    ///
    /// `MonitorHandle::stop` は (1) `stop_flag` を立て (2) `kill()` で
    /// このスロットの mutex を取りに来る、という二段階で動く。再起動側で
    /// 「flag を見てから差し替える」を別ステップに分けると、間に stop の
    /// kill が割り込んで「stop 後に新プロセスがスロットへ載る」TOCTOU が
    /// 起きる。スロット mutex を握ったまま flag を読み、立っていれば
    /// 差し替えず `Some(new_proc)` で呼び出し側に返す。返した側が責任を
    /// 持って kill する。
    fn replace_unless_stopped(
        &self,
        new_proc: PythonProcess,
        stop_flag: &AtomicBool,
    ) -> Option<PythonProcess> {
        let mut guard = self.inner.lock().unwrap();
        if stop_flag.load(Ordering::SeqCst) {
            return Some(new_proc);
        }
        let old = std::mem::replace(&mut *guard, Arc::new(new_proc));
        // 古い方の kill はロックを離してから (kill は child wait + reader join で
        // 数十 ms ブロックし得るので、保持する必要のないロックは早めに開放する)。
        drop(guard);
        old.kill();
        None
    }

    /// 現在格納しているプロセスを kill する (差し替えはしない)。
    fn kill(&self) {
        self.inner.lock().unwrap().kill();
    }
}

pub struct MonitorHandle {
    pub stop_flag: Arc<AtomicBool>,
    pub recognition: Arc<ProcessSlot>,
    pub mortal: Arc<ProcessSlot>,
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

/// `inference-result` イベントの payload。
/// フロント側 (`src/types/index.ts`) の `InferenceResult` + `GameBoardSummary` に対応。
///
/// issue #15: `inference` を Optional に変更。`my_turn=false` のフレームでは
/// 監視ループが mortal をスキップして `inference=None` を emit するため。
/// `timestamp` は payload 自体のタイムスタンプ (mortal が走らなくても
/// last_recognized_at を heartbeat させるため必須)。
#[derive(Debug, Clone, Serialize)]
pub struct InferenceEventPayload {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub inference: Option<InferenceResult>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub board: Option<GameBoardSummary>,
    pub timestamp: String,
}

/// `recognition-error` イベントの payload。
/// フロント側 `AppError` (src/types/index.ts) と互換になるよう
/// `type` フィールドにエラー分類 (recognition / inference / capture) を入れる。
#[derive(Debug, Clone, Serialize)]
pub struct RecognitionErrorPayload {
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub message: String,
    pub occurred_at: String,
}

/// 監視ループを起動する。
///
/// recognition / mortal の 2 プロセスを spawn し、起動直後に
/// ping/pong スモークテストを 1 サイクル流したあと、本ループで
/// キャプチャ → recognition → mortal → emit を 1Hz で繰り返す。
pub fn start<R: Runtime>(
    app: AppHandle<R>,
    config: MonitorConfig,
) -> Result<MonitorHandle, MonitorError> {
    let MonitorConfig {
        capture_target,
        mortal_model_path,
        inference_backend,
        roi_calibration,
    } = config;

    // capture_target は監視ループに move されるので、ここで未設定なら
    // フォールバック手段なく run_cycle がずっと NoTarget を返し続け、
    // recognition / mortal の子プロセスを無駄に起動したまま 1Hz で
    // エラーを emit し続けることになる。spawn 前に弾く。
    if capture_target.trim().is_empty() {
        return Err(MonitorError::InvalidConfig(
            "capture target window id is empty".into(),
        ));
    }

    info!(target: "monitor", "starting monitor for target='{}'", capture_target);

    let recognition_proc =
        PythonProcess::spawn_recognition(&app).map_err(MonitorError::RecognitionSpawn)?;
    let mortal_proc = PythonProcess::spawn_mortal(&app, &mortal_model_path, inference_backend)
        .map_err(MonitorError::MortalSpawn)?;

    let recognition = Arc::new(ProcessSlot::new(recognition_proc));
    let mortal = Arc::new(ProcessSlot::new(mortal_proc));

    let stop_flag = Arc::new(AtomicBool::new(false));
    let stop_for_thread = stop_flag.clone();
    let rec_for_thread = recognition.clone();
    let mortal_for_thread = mortal.clone();
    let app_for_thread = app.clone();
    // mortal の再起動時に同じ引数で `spawn_mortal` を呼ぶため
    // 監視スレッドへ持ち込む。
    let mortal_model_path_for_thread = mortal_model_path.clone();
    // 同様に、recognition の frame request に乗せるため監視スレッドへ持ち込む。
    //
    // TODO(future): 監視中に ROI を更新したいユースケース (キャリブレーションを
    // 微調整しながら結果を見たい) では、`Arc<Mutex<RoiCalibration>>` または
    // `Arc<ArcSwap<RoiCalibration>>` に切り替えて、設定保存時にホットスワップ
    // できるようにする。MVP は「停止 → ROI 編集 → 再開」で運用する想定なので
    // ひとまず clone で固定する (gemini review on PR #42)。
    let roi_for_thread = roi_calibration.clone();

    let join = std::thread::spawn(move || {
        info!(target: "monitor", "monitor loop started");

        // スモーク ping/pong は本ループの前に済ませる。
        // 並行に走らせると `PythonProcess::request_line` の roundtrip ロックを
        // スモーク側が握ったまま recv で詰まり、本ループのフレーム要求が
        // 出せなくなる (#36 review by codex)。recognition と mortal は別プロセスなので
        // 双方のスモークだけはスレッド分割して並列化し、起動時間を短縮する。
        // 起動時のスモークは「初回ロードで多少詰まっても本ループは回す」方針。
        // 戻り値は捨てて、失敗していれば最初の数サイクルが timeout 経路で
        // recognition-error を吐くだけに留める (UI 側で error phase に倒れる)。
        let rec_smoke = {
            let p = rec_for_thread.current();
            std::thread::spawn(move || {
                let _ = smoke_ping(p.as_ref(), "recognition");
            })
        };
        let mortal_smoke = {
            let p = mortal_for_thread.current();
            std::thread::spawn(move || {
                let _ = smoke_ping(p.as_ref(), "mortal");
            })
        };
        let _ = rec_smoke.join();
        let _ = mortal_smoke.join();

        // PRD §7.4: ProcessDied 時の自動再起動はライフタイム合計で 1 回まで。
        // これを超えたらフロント側で `phase = error` に固定して停止操作を待つ。
        let mut restart_tracker = RestartTracker::new();

        // SMOKE_PING_ID (=0) と衝突しないよう 1 から開始。
        let mut frame_id: i64 = SMOKE_PING_ID + 1;
        while !stop_for_thread.load(Ordering::SeqCst) {
            let cycle_start = Instant::now();
            let recognition_proc = rec_for_thread.current();
            let mortal_proc = mortal_for_thread.current();
            match run_cycle(
                &capture_target,
                frame_id,
                recognition_proc.as_ref(),
                mortal_proc.as_ref(),
                &roi_for_thread,
            ) {
                Ok((inference, board, timestamp)) => {
                    let payload = InferenceEventPayload {
                        inference,
                        board,
                        timestamp,
                    };
                    if let Err(e) = app_for_thread.emit("inference-result", &payload) {
                        warn!(target: "monitor", "emit inference-result failed: {}", e);
                    }
                }
                Err(e) => {
                    handle_cycle_error(
                        &app_for_thread,
                        &e,
                        &rec_for_thread,
                        &mortal_for_thread,
                        &mortal_model_path_for_thread,
                        inference_backend,
                        &mut restart_tracker,
                        &stop_for_thread,
                    );
                }
            }
            frame_id = frame_id.wrapping_add(1);
            sleep_until_next_cycle(cycle_start, &stop_for_thread);
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

/// `ProcessDied` 時の再起動回数を追跡する。
/// PRD §7.4 で「Mortal 推論失敗 → 監視は継続」「認識失敗 → 次フレームへスキップ」
/// と定められており、本実装ではライフタイム 1 回までの再起動を許容して、
/// 失敗を超えた場合はフロント側を `phase = error` で停止させる。
struct RestartTracker {
    recognition_attempted: bool,
    mortal_attempted: bool,
}

impl RestartTracker {
    fn new() -> Self {
        Self {
            recognition_attempted: false,
            mortal_attempted: false,
        }
    }
}

/// 1 サイクル経過後、次サイクル開始までを `SLEEP_SLICE` 刻みで待機する。
/// stop_flag が立てば即座に抜けて停止に応答する。
fn sleep_until_next_cycle(cycle_start: Instant, stop_flag: &AtomicBool) {
    let deadline = cycle_start + FRAME_INTERVAL;
    while !stop_flag.load(Ordering::SeqCst) {
        let now = Instant::now();
        if now >= deadline {
            break;
        }
        let remaining = deadline - now;
        std::thread::sleep(remaining.min(SLEEP_SLICE));
    }
}

#[derive(Debug, Error)]
pub(crate) enum CycleError {
    #[error("capture target window id is empty")]
    NoTarget,
    #[error("capture failed: {0}")]
    Capture(String),
    #[error("png encode failed: {0}")]
    PngEncode(String),
    /// recognition プロセスからの応答が `PYTHON_RECV_TIMEOUT` 内に来なかった。
    /// PRD §7.4 「認識失敗 → スキップして次フレームへ」に該当 (warn ログ)。
    #[error("recognition timeout")]
    RecognitionTimeout,
    /// recognition プロセスへの I/O が失敗した (BrokenPipe など)。
    #[error("recognition io failed: {0}")]
    RecognitionIo(PythonProcError),
    /// recognition プロセスが死亡 (stdout EOF) した。再起動の対象。
    #[error("recognition process died")]
    RecognitionDied,
    /// recognition の 1 行が JSON としてパースできない。
    #[error("recognition parse failed: {0}")]
    RecognitionParseFail(String),
    /// パースは通ったがプロトコル上の必須フィールドが欠けている等。
    #[error("recognition response invalid: {0}")]
    RecognitionInvalid(String),
    /// mortal プロセスからの応答が `PYTHON_RECV_TIMEOUT` 内に来なかった。
    /// PRD §7.4 「Mortal 推論失敗 → エラー表示+ログ記録+監視は継続」に該当。
    #[error("mortal timeout")]
    MortalTimeout,
    #[error("mortal io failed: {0}")]
    MortalIo(PythonProcError),
    /// mortal プロセスが死亡。再起動の対象。
    #[error("mortal process died")]
    MortalDied,
    #[error("mortal parse failed: {0}")]
    MortalParseFail(String),
    #[error("mortal response invalid: {0}")]
    MortalInvalid(String),
}

impl CycleError {
    /// フロント側 `AppError.type` (src/types/index.ts) と整合する分類を返す。
    /// timeout / parse fail / process died も全て元のステージ
    /// (recognition or inference) に集約して、UI での表示を一貫させる。
    pub(crate) fn kind(&self) -> &'static str {
        match self {
            Self::NoTarget | Self::Capture(_) | Self::PngEncode(_) => "capture",
            Self::RecognitionTimeout
            | Self::RecognitionIo(_)
            | Self::RecognitionDied
            | Self::RecognitionParseFail(_)
            | Self::RecognitionInvalid(_) => "recognition",
            Self::MortalTimeout
            | Self::MortalIo(_)
            | Self::MortalDied
            | Self::MortalParseFail(_)
            | Self::MortalInvalid(_) => "inference",
        }
    }

    /// このエラーが「プロセス死亡」起因なら、どのプロセスかを返す。
    /// 監視ループはこれを使って 1 度だけ再起動を試みる (PRD §7.4)。
    pub(crate) fn dead_process(&self) -> Option<DeadProcess> {
        match self {
            Self::RecognitionDied => Some(DeadProcess::Recognition),
            Self::MortalDied => Some(DeadProcess::Mortal),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DeadProcess {
    Recognition,
    Mortal,
}

/// PythonProcError を recognition 側の CycleError 派生にマッピングする。
fn map_recognition_error(e: PythonProcError) -> CycleError {
    match e {
        PythonProcError::Timeout(_) => CycleError::RecognitionTimeout,
        PythonProcError::Terminated => CycleError::RecognitionDied,
        // BrokenPipe など stdin 書き込みエラーは「もう死んでいる」事象として扱う。
        // 子が死んだ瞬間は `Terminated` か `SpawnFailed(BrokenPipe)` のどちらが
        // 起きるか OS / タイミング依存なので、両方を ProcessDied 扱いに寄せる。
        PythonProcError::SpawnFailed(io_err) if io_err.kind() == std::io::ErrorKind::BrokenPipe => {
            CycleError::RecognitionDied
        }
        other => CycleError::RecognitionIo(other),
    }
}

fn map_mortal_error(e: PythonProcError) -> CycleError {
    match e {
        PythonProcError::Timeout(_) => CycleError::MortalTimeout,
        PythonProcError::Terminated => CycleError::MortalDied,
        PythonProcError::SpawnFailed(io_err) if io_err.kind() == std::io::ErrorKind::BrokenPipe => {
            CycleError::MortalDied
        }
        other => CycleError::MortalIo(other),
    }
}

/// 1 フレーム分の capture → recognition → (条件付き) mortal を実行する。
///
/// issue #15: `my_turn=false` のフレームでは mortal を呼ばずに
/// `(None, board, timestamp)` を返す。フロントは `inference: null` を受けて
/// IdleBody 表示を維持する。
fn run_cycle(
    capture_target: &str,
    frame_id: i64,
    recognition: &PythonProcess,
    mortal: &PythonProcess,
    roi_calibration: &RoiCalibration,
) -> Result<(Option<InferenceResult>, Option<GameBoardSummary>, String), CycleError> {
    if capture_target.trim().is_empty() {
        return Err(CycleError::NoTarget);
    }

    let img =
        capture::capture_window(capture_target).map_err(|e| CycleError::Capture(e.to_string()))?;
    let image_b64 =
        capture::encode_png_base64(&img).map_err(|e| CycleError::PngEncode(e.to_string()))?;

    // recognition: フレーム送信 → tenhou_json 受信
    // issue #10: roi_calibration は比率指定で、Python 側はキャプチャサイズに
    // 掛け合わせて領域を切り出す。未キャリブレーション時もフィールドは送るが
    // 各領域は null になる (Python 側はフォールバックで全画面を見る想定)。
    // 送信前に必ず `sanitize_roi_calibration` を通し、破損値 (NaN / 負値 /
    // 1 超過 / 端越え) を None に落としてから渡す。Python 側のクロップは
    // `[0..1]` 前提なので、ここで弾かないと無効領域でランタイムエラーを起こす
    // (CodeRabbit Major on PR #42)。
    let safe_roi = sanitize_roi_calibration(roi_calibration);
    let frame_req = serde_json::json!({
        "type": "frame",
        "id": frame_id,
        "image_b64": image_b64,
        "roi_calibration": safe_roi,
    });
    // 直前のサイクルがタイムアウトで終わった場合、その応答が今このサイクルの
    // 受信窓に滑り込んでくる可能性がある。drain だけでは送信〜recv の隙間で
    // 届いた stale を取りこぼすので、`accept` フィルタで「期待した id でない
    // 応答は捨てて受信を続ける」ことで正しい応答に追い付く。
    let rec_line = recognition
        .request_line_with_filter(&frame_req.to_string(), PYTHON_RECV_TIMEOUT, |line| {
            response_id_matches(line, frame_id)
        })
        .map_err(map_recognition_error)?;
    let rec_parsed: serde_json::Value = serde_json::from_str(rec_line.trim())
        .map_err(|e| CycleError::RecognitionParseFail(format!("json: {}", e)))?;
    if rec_parsed.get("type").and_then(|v| v.as_str()) != Some("result") {
        return Err(CycleError::RecognitionInvalid(format!(
            "unexpected type: {}",
            rec_parsed
                .get("type")
                .and_then(|v| v.as_str())
                .unwrap_or("")
        )));
    }
    // フィルタを通過した時点で id は一致しているが、欠損ガードのため再確認する。
    let rec_id = rec_parsed
        .get("id")
        .and_then(|v| v.as_i64())
        .ok_or_else(|| CycleError::RecognitionInvalid("missing id".into()))?;
    if rec_id != frame_id {
        return Err(CycleError::RecognitionInvalid(format!(
            "id mismatch: expected {}, got {}",
            frame_id, rec_id
        )));
    }
    let tenhou_json = rec_parsed
        .get("tenhou_json")
        .cloned()
        .ok_or_else(|| CycleError::RecognitionInvalid("missing tenhou_json".into()))?;
    let board = build_board_summary(&tenhou_json);

    // issue #15: 手番でないフレームでは mortal を呼ばない。recognition だけ
    // 走らせて board のみを emit することで、opponent turn のコストを大幅に
    // 削減する。`should_skip_inference` が true なら早期 return。
    if should_skip_inference(board.as_ref()) {
        return Ok((None, board, Utc::now().to_rfc3339()));
    }

    // mortal: tenhou_json 送信 → 推奨候補受信
    let infer_req = serde_json::json!({
        "type": "infer",
        "id": frame_id,
        "tenhou_json": tenhou_json,
    });
    let infer_line = mortal
        .request_line_with_filter(&infer_req.to_string(), PYTHON_RECV_TIMEOUT, |line| {
            response_id_matches(line, frame_id)
        })
        .map_err(map_mortal_error)?;
    let infer_parsed: serde_json::Value = serde_json::from_str(infer_line.trim())
        .map_err(|e| CycleError::MortalParseFail(format!("json: {}", e)))?;
    if infer_parsed.get("type").and_then(|v| v.as_str()) != Some("result") {
        return Err(CycleError::MortalInvalid(format!(
            "unexpected type: {}",
            infer_parsed
                .get("type")
                .and_then(|v| v.as_str())
                .unwrap_or("")
        )));
    }
    let infer_id = infer_parsed
        .get("id")
        .and_then(|v| v.as_i64())
        .ok_or_else(|| CycleError::MortalInvalid("missing id".into()))?;
    if infer_id != frame_id {
        return Err(CycleError::MortalInvalid(format!(
            "id mismatch: expected {}, got {}",
            frame_id, infer_id
        )));
    }

    let recommended_value = infer_parsed
        .get("recommended")
        .cloned()
        .ok_or_else(|| CycleError::MortalInvalid("missing recommended".into()))?;
    let recommended: RecommendationCandidate = serde_json::from_value(recommended_value)
        .map_err(|e| CycleError::MortalInvalid(format!("recommended: {}", e)))?;
    let candidates_value = infer_parsed
        .get("candidates")
        .cloned()
        .unwrap_or_else(|| serde_json::json!([]));
    let candidates: Vec<RecommendationCandidate> = serde_json::from_value(candidates_value)
        .map_err(|e| CycleError::MortalInvalid(format!("candidates: {}", e)))?;
    let timestamp = infer_parsed
        .get("timestamp")
        .and_then(|v| v.as_str())
        .map(str::to_string)
        .unwrap_or_else(|| Utc::now().to_rfc3339());

    let primary_label = format_primary_label(&recommended);

    let inference = InferenceResult {
        recommended,
        candidates,
        timestamp: timestamp.clone(),
        primary_label,
        reason: None,
        danger: None,
        safe: None,
    };

    Ok((Some(inference), board, timestamp))
}

/// run_cycle で発生したエラーを処理する:
/// 1. ログ + recognition-error イベント emit (UI を error phase に遷移させる)
/// 2. SQLite `error_log` に INSERT (TODO: E3 で実装)
/// 3. ProcessDied なら 1 度だけ Python プロセスの再起動を試みる
#[allow(clippy::too_many_arguments)]
fn handle_cycle_error<R: Runtime>(
    app: &AppHandle<R>,
    err: &CycleError,
    rec_slot: &Arc<ProcessSlot>,
    mortal_slot: &Arc<ProcessSlot>,
    mortal_model_path: &str,
    inference_backend: InferenceBackend,
    restart_tracker: &mut RestartTracker,
    stop_flag: &AtomicBool,
) {
    warn!(target: "monitor", "cycle failed: {}", err);

    let payload = RecognitionErrorPayload {
        kind: err.kind(),
        message: err.to_string(),
        occurred_at: Utc::now().to_rfc3339(),
    };
    if let Err(emit_err) = app.emit("recognition-error", &payload) {
        warn!(target: "monitor", "emit recognition-error failed: {}", emit_err);
    }

    // TODO(E3): error_log テーブルへ INSERT する。
    // 現状 tauri-plugin-sql が Rust から直接 INSERT する API を持たないため、
    // E3 で sqlx などの直接接続を導入してから差し込む。
    // INSERT INTO error_log (timestamp, error_type, message, stack_trace, related_game_state)
    //   VALUES (now, payload.kind, payload.message, NULL, NULL)
    let _ = (&payload.kind, &payload.message);

    // ProcessDied なら 1 度だけ自動再起動を試みる。
    let Some(dead) = err.dead_process() else {
        return;
    };
    // 監視停止中 (MonitorHandle::stop が kill 後に呼んだケース) で再起動すると、
    // 直後に Drop で再 kill するだけの無駄なプロセスを生むので何もしない。
    if stop_flag.load(Ordering::SeqCst) {
        return;
    }
    let already_attempted = match dead {
        DeadProcess::Recognition => restart_tracker.recognition_attempted,
        DeadProcess::Mortal => restart_tracker.mortal_attempted,
    };
    if already_attempted {
        warn!(
            target: "monitor",
            "{:?} died again after restart; giving up (UI stays in error phase)",
            dead
        );
        return;
    }
    match dead {
        DeadProcess::Recognition => restart_tracker.recognition_attempted = true,
        DeadProcess::Mortal => restart_tracker.mortal_attempted = true,
    }

    info!(target: "monitor", "attempting one-shot restart of {:?}", dead);
    let spawn_result = match dead {
        DeadProcess::Recognition => PythonProcess::spawn_recognition(app),
        DeadProcess::Mortal => {
            PythonProcess::spawn_mortal(app, mortal_model_path, inference_backend)
        }
    };
    match spawn_result {
        Ok(new_proc) => {
            // spawn_* 中に MonitorHandle::stop() が走った可能性がある。
            // `replace_unless_stopped` がスロット mutex 内で stop_flag を読むので、
            // stop() の (flag 立て→slot.kill) と差し替えが交差しても、新プロセスが
            // 「stop 後にスロットへ載って取り残される」状態にはならない。
            let bailed = match dead {
                DeadProcess::Recognition => rec_slot.replace_unless_stopped(new_proc, stop_flag),
                DeadProcess::Mortal => mortal_slot.replace_unless_stopped(new_proc, stop_flag),
            };
            if let Some(unused) = bailed {
                info!(
                    target: "monitor",
                    "stop requested during {:?} respawn; killing new process",
                    dead
                );
                unused.kill();
                return;
            }
            info!(target: "monitor", "{:?} restarted", dead);
            // 再起動直後の Python は uv のモジュールロード等で初期応答が遅く、
            // そのまま次の run_cycle に入ると 3 秒タイムアウトを使い切って
            // 再起動枠を浪費するだけになりがち。長めの SMOKE_RECV_TIMEOUT で
            // 一度 ping を流してウォームアップし、応答可能な状態を確認してから
            // 本ループへ戻す。
            //
            // smoke が失敗した場合は再起動を「成功」と見なせない: 不健全な
            // プロセスをスロットに残しても次サイクル以降は restart 枠を
            // 既に使い切っているのでエラーを emit し続けるだけになる。スロットの
            // プロセスを kill し、追加の `recognition-error` を流して UI を
            // error phase に固定する。
            let warm = match dead {
                DeadProcess::Recognition => rec_slot.current(),
                DeadProcess::Mortal => mortal_slot.current(),
            };
            let smoke_label = match dead {
                DeadProcess::Recognition => "recognition",
                DeadProcess::Mortal => "mortal",
            };
            if let Err(reason) = smoke_ping(warm.as_ref(), smoke_label) {
                warn!(
                    target: "monitor",
                    "{:?} smoke after restart failed: {}; killing freshly-spawned process",
                    dead, reason
                );
                drop(warm);
                match dead {
                    DeadProcess::Recognition => rec_slot.kill(),
                    DeadProcess::Mortal => mortal_slot.kill(),
                }
                let payload = RecognitionErrorPayload {
                    kind: match dead {
                        DeadProcess::Recognition => "recognition",
                        DeadProcess::Mortal => "inference",
                    },
                    message: format!("restart smoke failed: {}", reason),
                    occurred_at: Utc::now().to_rfc3339(),
                };
                if let Err(emit_err) = app.emit("recognition-error", &payload) {
                    warn!(target: "monitor", "emit recognition-error failed: {}", emit_err);
                }
            }
        }
        Err(spawn_err) => {
            warn!(
                target: "monitor",
                "{:?} restart failed: {}; UI will remain in error phase",
                dead, spawn_err
            );
            // 再起動自体に失敗した場合も UI へ通知して error phase を維持する。
            let payload = RecognitionErrorPayload {
                kind: match dead {
                    DeadProcess::Recognition => "recognition",
                    DeadProcess::Mortal => "inference",
                },
                message: format!("restart failed: {}", spawn_err),
                occurred_at: Utc::now().to_rfc3339(),
            };
            if let Err(emit_err) = app.emit("recognition-error", &payload) {
                warn!(target: "monitor", "emit recognition-error failed: {}", emit_err);
            }
        }
    }
}

/// 推奨候補から UI のプライマリ表示文を組み立てる。
///
/// - mortal が `action_label` を返している場合はそれを優先して尊重する
///   (フロント側の表現バリエーション「リーチ / ダマ / スルー」等を残せるため)。
/// - 無い場合は `action_type` ごとの定型文にフォールバックし、Discard だけは
///   `tile` と組み合わせて「N を切る」にする。`tile` が無い Discard は
///   想定外なので `None` を返してフロント側で他の手掛かり (recommended.tile
///   等) に任せる。
fn format_primary_label(rec: &RecommendationCandidate) -> Option<String> {
    if let Some(label) = rec.action_label.as_ref().filter(|s| !s.trim().is_empty()) {
        return Some(label.clone());
    }
    match rec.action_type {
        ActionType::Discard => rec.tile.as_ref().map(|t| format!("{} を切る", t)),
        ActionType::Riichi => Some("リーチ".into()),
        ActionType::Tsumo => Some("ツモ".into()),
        ActionType::Ron => Some("ロン".into()),
        ActionType::Pon => Some("ポン".into()),
        ActionType::Chi => Some("チー".into()),
        ActionType::Kan => Some("カン".into()),
        ActionType::Pass => Some("スルー".into()),
    }
}

/// recognition が返す tenhou_json から UI 用の盤面サマリを抽出する。
/// 必須フィールド (hand, self_wind, round_wind, turn, dora_indicators) が
/// 揃わない場合は `None` を返し、フロント側で「盤面なし」表示にフォールバックさせる。
fn build_board_summary(tenhou: &serde_json::Value) -> Option<GameBoardSummary> {
    let obj = tenhou.as_object()?;
    // 配列要素が全て文字列でない場合は、部分的な手牌で先に進むより
    // GameBoardSummary 全体を None にして「盤面なし」表示に倒した方が安全。
    // recognition 側のスキーマ崩れ (例: 数値が混ざる) を黙って通過させない。
    let hand: Vec<String> = obj
        .get("hand")?
        .as_array()?
        .iter()
        .map(|v| v.as_str().map(String::from))
        .collect::<Option<Vec<String>>>()?;
    let self_wind = obj.get("self_wind")?.as_str()?.to_string();
    let round_wind = obj.get("round_wind")?.as_str()?.to_string();
    let turn: u32 = obj.get("turn")?.as_u64()?.try_into().ok()?;
    let dora_indicators: Vec<String> = obj
        .get("dora_indicators")?
        .as_array()?
        .iter()
        .map(|v| v.as_str().map(String::from))
        .collect::<Option<Vec<String>>>()?;
    // tenhou 形式の `scores` は座順 (東→南→西→北) に並んでいる前提で、
    // 自分のスコアを `self_wind` から引く。先頭固定だと起家以外で誤った値が
    // フロントに渡るため。インデックスを引けない場合は `None` にして
    // 「持ち点不明」フォールバックさせる。
    let score = self_wind_index(&self_wind).and_then(|idx| {
        obj.get("scores")
            .and_then(|v| v.as_array())
            .and_then(|arr| arr.get(idx))
            .and_then(|v| v.as_i64())
    });
    let round_label = obj
        .get("round_label")
        .and_then(|v| v.as_str())
        .map(str::to_string);
    // issue #15: my_turn / available_actions は寛容に抽出する。recognition が
    // 旧スキーマで返した (= フィールドが存在しない) 場合は false / 空配列に
    // 倒すことで、Phase C 前の tenhou_json も壊さずに通過させる。ただし「フィールド
    // 欠落」状態では Rust 側で「skip しない」(= 従来通り mortal を呼ぶ) 方向に
    // フェイルセーフする (`should_skip_inference` 参照)。
    let my_turn = obj
        .get("my_turn")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let available_actions: Vec<String> = obj
        .get("available_actions")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();
    Some(GameBoardSummary {
        hand,
        self_wind,
        round_wind,
        turn,
        dora_indicators,
        score,
        round_label,
        my_turn,
        available_actions,
    })
}

/// `my_turn=false` または `available_actions` が空のとき、mortal 推論を
/// スキップして monitor ループが軽くなるようにする (issue #15)。
///
/// `board` が None のとき (= recognition のスキーマ崩れ / 必須フィールド欠落)
/// は「念のため mortal を呼ぶ」側に倒す。これがないと、recognition だけが
/// 一時的に壊れた状態で UI が完全に IdleBody に張り付き、デバッグしにくくなる。
fn should_skip_inference(board: Option<&GameBoardSummary>) -> bool {
    match board {
        None => false,
        Some(b) => !b.my_turn || b.available_actions.is_empty(),
    }
}

/// `request_line_with_filter` のクロージャから呼ぶための受信応答 id 判定。
///
/// 「自分宛ではない」と確信できる応答だけ reject する:
///   - パース可能 + `id` が整数 + 期待値と異なる → 直前フレームの stale
///
/// それ以外 (JSON パース失敗 / `id` 欠損 / 非整数 / 期待値と一致) は accept し、
/// `run_cycle` 側のチェック (`*ParseFail` / `*Invalid`) に委ねる。malformed な
/// 応答を「stale」と見なして捨ててしまうと、本来即座にエラーを返すべきなのに
/// `PYTHON_RECV_TIMEOUT` (3 秒) ぶら下がってから `*Timeout` を出すことになり、
/// プロトコル退行が隠れる + 1 フレーム遅延する (Codex P2 on PR #40)。
fn response_id_matches(line: &str, expected: i64) -> bool {
    let Ok(value) = serde_json::from_str::<serde_json::Value>(line.trim()) else {
        // パース不能 → run_cycle で *ParseFail として即時失敗させる
        return true;
    };
    match value.get("id").and_then(|x| x.as_i64()) {
        // id が一致 → 期待した応答
        Some(id) if id == expected => true,
        // id が異なる整数 → 別フレームの stale とみなして次の応答へ
        Some(_) => false,
        // id 欠損 / 非整数 → run_cycle の Invalid ガードに任せる
        None => true,
    }
}

/// 自家の風 (`self_wind`) を `scores` 配列のインデックスへ写像する。
/// tenhou 形式では座順 = 東 (0), 南 (1), 西 (2), 北 (3) で並ぶ。
fn self_wind_index(self_wind: &str) -> Option<usize> {
    match self_wind {
        "東" => Some(0),
        "南" => Some(1),
        "西" => Some(2),
        "北" => Some(3),
        _ => None,
    }
}

/// 1 往復の ping/pong を同期実行する。本ループ突入前に呼ばれる前提なので、
/// `request_line` の roundtrip ロックを握っても誰も待っていない。応答が無く
/// 詰まった場合でも stop() の kill() が `recv_line` を解錠して戻ってくる。
///
/// 戻り値は warm-up 健全性の判定に使う。起動時のスモーク (start() 直下) では
/// 結果を捨てて UI への影響を抑え、再起動後のスモーク (handle_cycle_error)
/// では失敗を「不健全な再起動」として扱い直すために `Err` を返す。
fn smoke_ping(proc: &PythonProcess, label: &str) -> Result<(), String> {
    let resp = match proc.request_line_timeout(
        &format!(r#"{{"type":"ping","id":{}}}"#, SMOKE_PING_ID),
        SMOKE_RECV_TIMEOUT,
    ) {
        Ok(s) => s,
        Err(e) => {
            warn!(target: "monitor", "smoke ping/pong [{}] failed: {}", label, e);
            return Err(e.to_string());
        }
    };
    let trimmed = resp.trim();
    match validate_pong(trimmed) {
        Ok(()) => {
            info!(target: "monitor", "smoke ping/pong [{}]: ok ({})", label, trimmed);
            Ok(())
        }
        Err(reason) => {
            warn!(
                target: "monitor",
                "smoke ping/pong [{}]: {} (raw: {})",
                label, reason, trimmed
            );
            Err(reason)
        }
    }
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

/// ROI 矩形が `[0..1]` の単位空間に収まり、幅高さが正の有限値であることを判定する。
///
/// Python 側のクロップは `(x, y, w, h)` を画像サイズに掛け合わせて切り出す前提で、
/// 範囲外や負値が来るとそこで例外になる。ここで弾けば「設定が壊れていれば
/// 全画面フォールバック」に倒せる。
fn valid_rect(r: &RoiRect) -> bool {
    r.x.is_finite()
        && r.y.is_finite()
        && r.w.is_finite()
        && r.h.is_finite()
        && r.w > 0.0
        && r.h > 0.0
        && r.x >= 0.0
        && r.y >= 0.0
        && r.x + r.w <= 1.0
        && r.y + r.h <= 1.0
}

fn sanitize_opt_rect(v: Option<RoiRect>) -> Option<RoiRect> {
    v.filter(valid_rect)
}

/// `roi_calibration` を recognition に送る直前のサニタイズ。
/// 無効値はフィールドごと `None` に落として、Python 側を全画面フォールバック
/// 経路に倒す (CodeRabbit Major on PR #42)。
fn sanitize_roi_calibration(src: &RoiCalibration) -> RoiCalibration {
    RoiCalibration {
        hand: sanitize_opt_rect(src.hand),
        doras: sanitize_opt_rect(src.doras),
        rivers: RiverRois {
            self_seat: sanitize_opt_rect(src.rivers.self_seat),
            right: sanitize_opt_rect(src.rivers.right),
            across: sanitize_opt_rect(src.rivers.across),
            left: sanitize_opt_rect(src.rivers.left),
        },
        melds: MeldRois {
            self_seat: sanitize_opt_rect(src.melds.self_seat),
            right: sanitize_opt_rect(src.melds.right),
            across: sanitize_opt_rect(src.melds.across),
            left: sanitize_opt_rect(src.melds.left),
        },
        round_info: sanitize_opt_rect(src.round_info),
        self_wind: sanitize_opt_rect(src.self_wind),
        scores: sanitize_opt_rect(src.scores),
        turn_counter: sanitize_opt_rect(src.turn_counter),
        action_buttons: sanitize_opt_rect(src.action_buttons),
        turn_timer: sanitize_opt_rect(src.turn_timer),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cycle_error_kinds_map_to_app_error_types() {
        // capture 系は全て "capture"
        assert_eq!(CycleError::NoTarget.kind(), "capture");
        assert_eq!(CycleError::Capture("x".into()).kind(), "capture");
        assert_eq!(CycleError::PngEncode("x".into()).kind(), "capture");

        // recognition 系 (timeout / parse fail / died / io) は全て "recognition"
        assert_eq!(CycleError::RecognitionTimeout.kind(), "recognition");
        assert_eq!(CycleError::RecognitionDied.kind(), "recognition");
        assert_eq!(
            CycleError::RecognitionParseFail("x".into()).kind(),
            "recognition"
        );
        assert_eq!(
            CycleError::RecognitionInvalid("x".into()).kind(),
            "recognition"
        );
        assert_eq!(
            CycleError::RecognitionIo(PythonProcError::SpawnFailed(std::io::Error::other("x")))
                .kind(),
            "recognition"
        );

        // mortal 系は全て "inference" (フロント側 AppError.type の語彙)
        assert_eq!(CycleError::MortalTimeout.kind(), "inference");
        assert_eq!(CycleError::MortalDied.kind(), "inference");
        assert_eq!(CycleError::MortalParseFail("x".into()).kind(), "inference");
        assert_eq!(CycleError::MortalInvalid("x".into()).kind(), "inference");
        assert_eq!(
            CycleError::MortalIo(PythonProcError::SpawnFailed(std::io::Error::other("x"))).kind(),
            "inference"
        );
    }

    #[test]
    fn dead_process_only_set_for_died_variants() {
        assert_eq!(
            CycleError::RecognitionDied.dead_process(),
            Some(DeadProcess::Recognition)
        );
        assert_eq!(
            CycleError::MortalDied.dead_process(),
            Some(DeadProcess::Mortal)
        );
        assert!(CycleError::RecognitionTimeout.dead_process().is_none());
        assert!(CycleError::MortalTimeout.dead_process().is_none());
        assert!(CycleError::RecognitionParseFail("x".into())
            .dead_process()
            .is_none());
    }

    #[test]
    fn map_recognition_error_classifies_timeout_died_io() {
        let timeout = map_recognition_error(PythonProcError::Timeout(Duration::from_secs(1)));
        assert!(matches!(timeout, CycleError::RecognitionTimeout));

        let died = map_recognition_error(PythonProcError::Terminated);
        assert!(matches!(died, CycleError::RecognitionDied));

        let broken_pipe = map_recognition_error(PythonProcError::SpawnFailed(std::io::Error::new(
            std::io::ErrorKind::BrokenPipe,
            "pipe",
        )));
        assert!(matches!(broken_pipe, CycleError::RecognitionDied));

        let other =
            map_recognition_error(PythonProcError::SpawnFailed(std::io::Error::other("boom")));
        assert!(matches!(other, CycleError::RecognitionIo(_)));
    }

    #[test]
    fn map_mortal_error_classifies_timeout_died_io() {
        let timeout = map_mortal_error(PythonProcError::Timeout(Duration::from_secs(1)));
        assert!(matches!(timeout, CycleError::MortalTimeout));

        let died = map_mortal_error(PythonProcError::Terminated);
        assert!(matches!(died, CycleError::MortalDied));

        let broken_pipe = map_mortal_error(PythonProcError::SpawnFailed(std::io::Error::new(
            std::io::ErrorKind::BrokenPipe,
            "pipe",
        )));
        assert!(matches!(broken_pipe, CycleError::MortalDied));
    }

    /// `response_id_matches` は「自分宛ではないと確信できる応答」(=別 id の
    /// 整数を持つもの) だけを reject し、それ以外 (一致 / パース不能 / id 欠損 /
    /// 非整数) は accept する。malformed を reject すると run_cycle が timeout
    /// まで待つことになり、本来即時に *ParseFail / *Invalid を返すべき
    /// プロトコル退行を隠してしまう (Codex P2 on PR #40)。
    #[test]
    fn response_id_matches_accepts_matching_and_malformed_rejects_other_id() {
        // 一致 → accept
        assert!(response_id_matches(r#"{"id":7,"type":"result"}"#, 7));
        // 別 id → reject
        assert!(!response_id_matches(r#"{"id":6,"type":"result"}"#, 7));
        // パース不能 → accept (run_cycle で *ParseFail にする)
        assert!(response_id_matches("not json {", 7));
        // id 欠損 → accept (run_cycle の Invalid ガードで拾う)
        assert!(response_id_matches(r#"{"type":"result"}"#, 7));
        // 非整数 id → accept (同上)
        assert!(response_id_matches(r#"{"id":"7","type":"result"}"#, 7));
    }

    /// `valid_rect` は単位空間 [0,1] に収まる正の矩形だけを受け入れる
    /// (CodeRabbit Major on PR #42)。
    #[test]
    fn valid_rect_accepts_unit_square_and_rejects_out_of_range() {
        // 正常系
        assert!(valid_rect(&RoiRect {
            x: 0.0,
            y: 0.0,
            w: 1.0,
            h: 1.0
        }));
        assert!(valid_rect(&RoiRect {
            x: 0.1,
            y: 0.2,
            w: 0.3,
            h: 0.4
        }));
        // 負の x / y
        assert!(!valid_rect(&RoiRect {
            x: -0.01,
            y: 0.0,
            w: 0.5,
            h: 0.5
        }));
        assert!(!valid_rect(&RoiRect {
            x: 0.0,
            y: -0.01,
            w: 0.5,
            h: 0.5
        }));
        // 0 幅 / 0 高さ
        assert!(!valid_rect(&RoiRect {
            x: 0.1,
            y: 0.1,
            w: 0.0,
            h: 0.5
        }));
        assert!(!valid_rect(&RoiRect {
            x: 0.1,
            y: 0.1,
            w: 0.5,
            h: 0.0
        }));
        // 端越え
        assert!(!valid_rect(&RoiRect {
            x: 0.6,
            y: 0.0,
            w: 0.5,
            h: 0.5
        }));
        assert!(!valid_rect(&RoiRect {
            x: 0.0,
            y: 0.6,
            w: 0.5,
            h: 0.5
        }));
        // NaN / Infinity
        assert!(!valid_rect(&RoiRect {
            x: f64::NAN,
            y: 0.0,
            w: 0.5,
            h: 0.5
        }));
        assert!(!valid_rect(&RoiRect {
            x: 0.0,
            y: 0.0,
            w: f64::INFINITY,
            h: 0.5
        }));
    }

    /// `sanitize_roi_calibration` は無効値を `None` に落とし、有効値はそのまま残す。
    /// 鍵となるのは「壊れた値 1 つでフルキャリブレーションが捨てられない」こと。
    #[test]
    fn sanitize_roi_calibration_drops_invalid_only() {
        let mut roi = RoiCalibration::default();
        let good = RoiRect {
            x: 0.1,
            y: 0.1,
            w: 0.2,
            h: 0.2,
        };
        let bad_negative = RoiRect {
            x: -0.1,
            y: 0.0,
            w: 0.5,
            h: 0.5,
        };
        let bad_overflow = RoiRect {
            x: 0.8,
            y: 0.0,
            w: 0.5,
            h: 0.5,
        };
        roi.hand = Some(good);
        roi.doras = Some(bad_negative);
        roi.rivers.self_seat = Some(good);
        roi.rivers.right = Some(bad_overflow);
        roi.round_info = Some(good);
        roi.self_wind = None;
        // issue #12: scores / turn_counter も同じく無効値は落とす
        roi.scores = Some(bad_negative);
        roi.turn_counter = Some(good);
        // issue #15: action_buttons / turn_timer も同じく無効値は落とす
        roi.action_buttons = Some(good);
        roi.turn_timer = Some(bad_overflow);

        let cleaned = sanitize_roi_calibration(&roi);
        assert!(cleaned.hand.is_some());
        assert!(cleaned.doras.is_none(), "negative x must be dropped");
        assert!(cleaned.rivers.self_seat.is_some());
        assert!(cleaned.rivers.right.is_none(), "x+w > 1.0 must be dropped");
        assert!(cleaned.round_info.is_some());
        assert!(cleaned.self_wind.is_none());
        assert!(
            cleaned.scores.is_none(),
            "scores with negative x must be dropped"
        );
        assert!(cleaned.turn_counter.is_some());
        assert!(cleaned.action_buttons.is_some());
        assert!(
            cleaned.turn_timer.is_none(),
            "turn_timer with x+w > 1.0 must be dropped"
        );
    }

    /// issue #15: `build_board_summary` は `my_turn` / `available_actions` を
    /// tenhou_json から抽出し、欠落時は安全な既定値 (false / 空配列) で埋める。
    #[test]
    fn build_board_summary_extracts_my_turn_and_actions() {
        let tenhou = serde_json::json!({
            "hand": ["1m", "2m"],
            "self_wind": "東",
            "round_wind": "東",
            "turn": 3,
            "dora_indicators": ["5p"],
            "scores": [25000, 25000, 25000, 25000],
            "my_turn": true,
            "available_actions": ["discard", "riichi"],
        });
        let board = build_board_summary(&tenhou).expect("summary");
        assert!(board.my_turn);
        assert_eq!(board.available_actions, vec!["discard", "riichi"]);
    }

    /// issue #15: 旧 tenhou_json (my_turn / available_actions 無し) は
    /// `(false, vec![])` で埋まる (互換性レグレッション)。
    #[test]
    fn build_board_summary_defaults_when_my_turn_missing() {
        let tenhou = serde_json::json!({
            "hand": [],
            "self_wind": "東",
            "round_wind": "東",
            "turn": 1,
            "dora_indicators": [],
            "scores": [25000, 25000, 25000, 25000],
        });
        let board = build_board_summary(&tenhou).expect("summary");
        assert!(!board.my_turn);
        assert!(board.available_actions.is_empty());
    }

    /// issue #15: `should_skip_inference` の判定マトリクス。
    /// - board None → false (フェイルセーフで mortal を呼ぶ側)
    /// - my_turn=false → true
    /// - my_turn=true + actions empty → true
    /// - my_turn=true + actions 非空 → false
    #[test]
    fn should_skip_inference_decision_matrix() {
        assert!(!should_skip_inference(None));

        let opponent_turn = GameBoardSummary {
            hand: vec![],
            self_wind: "東".into(),
            round_wind: "東".into(),
            turn: 1,
            dora_indicators: vec![],
            score: None,
            round_label: None,
            my_turn: false,
            available_actions: vec!["discard".into()],
        };
        assert!(should_skip_inference(Some(&opponent_turn)));

        let my_turn_no_actions = GameBoardSummary {
            my_turn: true,
            available_actions: vec![],
            ..opponent_turn.clone()
        };
        assert!(should_skip_inference(Some(&my_turn_no_actions)));

        let my_turn_with_action = GameBoardSummary {
            my_turn: true,
            available_actions: vec!["discard".into()],
            ..opponent_turn
        };
        assert!(!should_skip_inference(Some(&my_turn_with_action)));
    }
}
