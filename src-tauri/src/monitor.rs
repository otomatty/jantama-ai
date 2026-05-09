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
use crate::types::{ActionType, GameBoardSummary, InferenceResult, RecommendationCandidate};
use base64::engine::general_purpose::STANDARD as B64;
use base64::Engine as _;
use chrono::Utc;
use serde::Serialize;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
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

/// `inference-result` イベントの payload。
/// フロント側 (`src/types/index.ts`) の `InferenceResult` + `GameBoardSummary` に対応。
#[derive(Debug, Clone, Serialize)]
pub struct InferenceEventPayload {
    pub inference: InferenceResult,
    pub board: Option<GameBoardSummary>,
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
    let rec_for_thread = recognition.clone();
    let mortal_for_thread = mortal.clone();
    let app_for_thread = app.clone();

    let join = std::thread::spawn(move || {
        info!(target: "monitor", "monitor loop started");

        // スモーク ping/pong は本ループの前に済ませる。
        // 並行に走らせると `PythonProcess::request_line` の roundtrip ロックを
        // スモーク側が握ったまま recv で詰まり、本ループのフレーム要求が
        // 出せなくなる (#36 review by codex)。recognition と mortal は別プロセスなので
        // 双方のスモークだけはスレッド分割して並列化し、起動時間を短縮する。
        let rec_smoke = {
            let p = rec_for_thread.clone();
            std::thread::spawn(move || smoke_ping(p.as_ref(), "recognition"))
        };
        let mortal_smoke = {
            let p = mortal_for_thread.clone();
            std::thread::spawn(move || smoke_ping(p.as_ref(), "mortal"))
        };
        let _ = rec_smoke.join();
        let _ = mortal_smoke.join();

        // SMOKE_PING_ID (=0) と衝突しないよう 1 から開始。
        let mut frame_id: i64 = SMOKE_PING_ID + 1;
        while !stop_for_thread.load(Ordering::SeqCst) {
            let cycle_start = Instant::now();
            match run_cycle(
                &capture_target,
                frame_id,
                rec_for_thread.as_ref(),
                mortal_for_thread.as_ref(),
            ) {
                Ok((inference, board)) => {
                    let payload = InferenceEventPayload { inference, board };
                    if let Err(e) = app_for_thread.emit("inference-result", &payload) {
                        warn!(target: "monitor", "emit inference-result failed: {}", e);
                    }
                }
                Err(e) => {
                    warn!(target: "monitor", "cycle failed: {}", e);
                    let payload = RecognitionErrorPayload {
                        kind: e.kind(),
                        message: e.to_string(),
                        occurred_at: Utc::now().to_rfc3339(),
                    };
                    if let Err(emit_err) = app_for_thread.emit("recognition-error", &payload) {
                        warn!(target: "monitor", "emit recognition-error failed: {}", emit_err);
                    }
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
enum CycleError {
    #[error("capture target window id is empty")]
    NoTarget,
    #[error("capture failed: {0}")]
    Capture(String),
    #[error("png encode failed: {0}")]
    PngEncode(String),
    #[error("recognition io failed: {0}")]
    RecognitionIo(PythonProcError),
    #[error("recognition response invalid: {0}")]
    RecognitionInvalid(String),
    #[error("mortal io failed: {0}")]
    MortalIo(PythonProcError),
    #[error("mortal response invalid: {0}")]
    MortalInvalid(String),
}

impl CycleError {
    /// フロント側 `AppError.type` (src/types/index.ts) と整合する分類を返す。
    fn kind(&self) -> &'static str {
        match self {
            Self::NoTarget | Self::Capture(_) | Self::PngEncode(_) => "capture",
            Self::RecognitionIo(_) | Self::RecognitionInvalid(_) => "recognition",
            Self::MortalIo(_) | Self::MortalInvalid(_) => "inference",
        }
    }
}

/// 1 フレーム分の capture → recognition → mortal を実行する。
fn run_cycle(
    capture_target: &str,
    frame_id: i64,
    recognition: &PythonProcess,
    mortal: &PythonProcess,
) -> Result<(InferenceResult, Option<GameBoardSummary>), CycleError> {
    if capture_target.trim().is_empty() {
        return Err(CycleError::NoTarget);
    }

    let img =
        capture::capture_window(capture_target).map_err(|e| CycleError::Capture(e.to_string()))?;
    let png = encode_png(&img).map_err(CycleError::PngEncode)?;
    let image_b64 = B64.encode(&png);

    // recognition: フレーム送信 → tenhou_json 受信
    let frame_req = serde_json::json!({
        "type": "frame",
        "id": frame_id,
        "image_b64": image_b64,
    });
    let rec_line = recognition
        .request_line(&frame_req.to_string())
        .map_err(CycleError::RecognitionIo)?;
    let rec_parsed: serde_json::Value = serde_json::from_str(rec_line.trim())
        .map_err(|e| CycleError::RecognitionInvalid(format!("json: {}", e)))?;
    if rec_parsed.get("type").and_then(|v| v.as_str()) != Some("result") {
        return Err(CycleError::RecognitionInvalid(format!(
            "unexpected type: {}",
            rec_parsed
                .get("type")
                .and_then(|v| v.as_str())
                .unwrap_or("")
        )));
    }
    let tenhou_json = rec_parsed
        .get("tenhou_json")
        .cloned()
        .ok_or_else(|| CycleError::RecognitionInvalid("missing tenhou_json".into()))?;
    let board = build_board_summary(&tenhou_json);

    // mortal: tenhou_json 送信 → 推奨候補受信
    let infer_req = serde_json::json!({
        "type": "infer",
        "id": frame_id,
        "tenhou_json": tenhou_json,
    });
    let infer_line = mortal
        .request_line(&infer_req.to_string())
        .map_err(CycleError::MortalIo)?;
    let infer_parsed: serde_json::Value = serde_json::from_str(infer_line.trim())
        .map_err(|e| CycleError::MortalInvalid(format!("json: {}", e)))?;
    if infer_parsed.get("type").and_then(|v| v.as_str()) != Some("result") {
        return Err(CycleError::MortalInvalid(format!(
            "unexpected type: {}",
            infer_parsed
                .get("type")
                .and_then(|v| v.as_str())
                .unwrap_or("")
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
        timestamp,
        primary_label,
        reason: None,
        danger: None,
        safe: None,
    };

    Ok((inference, board))
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

/// `RgbaImage` を PNG にエンコードしてバイト列で返す。
fn encode_png(img: &xcap::image::RgbaImage) -> Result<Vec<u8>, String> {
    use std::io::Cursor;
    let mut buf = Vec::new();
    img.write_to(&mut Cursor::new(&mut buf), xcap::image::ImageFormat::Png)
        .map_err(|e| e.to_string())?;
    Ok(buf)
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
    Some(GameBoardSummary {
        hand,
        self_wind,
        round_wind,
        turn,
        dora_indicators,
        score,
        round_label,
    })
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
fn smoke_ping(proc: &PythonProcess, label: &'static str) {
    let resp = match proc.request_line(&format!(r#"{{"type":"ping","id":{}}}"#, SMOKE_PING_ID)) {
        Ok(s) => s,
        Err(e) => {
            warn!(target: "monitor", "smoke ping/pong [{}] failed: {}", label, e);
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
