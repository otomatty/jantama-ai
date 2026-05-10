// Python サブプロセス管理
//
// PRD §8.1 全体構成図: Tauri (Rust) ↔ Python の間は stdin/stdout で
// JSON-lines 通信を行う。recognition / mortal の 2 プロセスを管理。

use crate::types::InferenceBackend;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::sync::Mutex;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};
// `Manager` は release ビルドでだけ `app.path()` を解決するために必要。
// debug ビルドでは未使用となるため条件付きで取り込む。
#[cfg(not(debug_assertions))]
use tauri::Manager;
use tauri::{AppHandle, Runtime};
use thiserror::Error;
use tracing::{debug, info};

#[derive(Debug, Error)]
pub enum PythonProcError {
    #[error("Pythonプロセスが見つかりません: {0}")]
    NotFound(String),
    #[error("プロセス起動に失敗しました: {0}")]
    SpawnFailed(#[from] std::io::Error),
    #[error("プロセスが終了しています")]
    Terminated,
    #[error("応答がタイムアウトしました ({}ms)", .0.as_millis())]
    Timeout(Duration),
    #[error("JSON parse error: {0}")]
    Json(#[from] serde_json::Error),
}

/// バックグラウンド reader スレッドが受け取った 1 イベント。
///
/// Python プロセスの stdout は 1 行ずつ JSON が流れてくる前提で、
/// `read_line` の戻り値ごとに `Line` / `Eof` / `Io` の 3 種に分類して
/// チャネルへ流す。受信側 (`recv_line_timeout`) はチャネルに何が
/// 入っているかで「正常応答 / プロセス死亡 / I/O 障害」を区別できる。
enum ReaderEvent {
    Line(String),
    Eof,
    Io(std::io::Error),
}

/// reader→consumer チャネルの上限。
///
/// 元実装は `read_line` を consumer スレッドが直接呼んでいたため、OS の pipe
/// バッファ (Linux で 64KiB 程度) が天井として働いていた。バックグラウンド
/// reader + 無制限 mpsc に切り替えると、Python が暴走して stdout に書き続けた
/// 場合キューが青天井に伸びてアプリが OOM する可能性がある (Codex P2)。
/// `sync_channel` で上限を設け、reader はブロッキング `send` を使う:
///
///   - 通常運用: 1 リクエスト = 1 レスポンスなのでキューは常にほぼ空
///   - 暴走時: reader が send で詰まる → 64 個目以降は pipe に残る → Python の
///     write が pipe バッファ満杯でブロック (= 元の OS 由来のバックプレッシャ)
///
/// try_send + drop で溢れを捨てる方式は、待っている応答そのものが drop されると
/// 偽タイムアウトを誘発する (Codex P1) ので採らない。
const READER_QUEUE_BOUND: usize = 64;

pub struct PythonProcess {
    child: Mutex<Child>,
    stdin: Mutex<ChildStdin>,
    /// stdout を読む専用スレッドが流すイベントの受け口。
    /// `recv_line` 系で取り出す。
    rx: Mutex<Receiver<ReaderEvent>>,
    /// reader スレッドのハンドル。`kill` 時に join するため保持する。
    reader_join: Mutex<Option<JoinHandle<()>>>,
    /// 送信→受信を 1 つのトランザクションとして直列化するためのロック。
    /// stdin / stdout のロックは個別なので、複数スレッドが
    /// `send_line` → `recv_line` を呼ぶと別スレッドの応答を
    /// 取り違える race が起き得る (#36 review)。`request_line` で
    /// このロックを取って往復をくくることで防ぐ。
    roundtrip: Mutex<()>,
    label: String,
}

impl PythonProcess {
    /// Python プロセスを起動する。
    ///
    /// `program`: 実行ファイル (PyInstaller でバンドル後の .exe を指定するか、
    ///            開発時は `uv` を渡し `args` で `["run", "jantama-recognition"]` 等)。
    /// `args`: コマンドライン引数。
    /// `cwd`: 作業ディレクトリ。dev で `uv run` する場合は `python/` を指定する。
    pub fn spawn(
        label: impl Into<String>,
        program: &Path,
        args: &[&str],
        cwd: Option<&Path>,
    ) -> Result<Self, PythonProcError> {
        let label = label.into();
        info!(
            target: "python_proc",
            "spawning {} ({} {})",
            label,
            program.display(),
            args.join(" ")
        );

        let mut command = Command::new(program);
        command
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            // stderr は read しないため pipe するとバッファ満杯でデッドロックし得る。
            // Phase D で構造化ログとして取り込むまでは親プロセスへ継承する。
            .stderr(Stdio::inherit());
        if let Some(dir) = cwd {
            command.current_dir(dir);
        }

        let mut child = command.spawn()?;

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| PythonProcError::NotFound("stdin".into()))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| PythonProcError::NotFound("stdout".into()))?;

        let (tx, rx) = mpsc::sync_channel::<ReaderEvent>(READER_QUEUE_BOUND);
        let label_for_thread = label.clone();
        let reader_join = std::thread::spawn(move || {
            reader_loop(stdout, tx, label_for_thread);
        });

        Ok(Self {
            child: Mutex::new(child),
            stdin: Mutex::new(stdin),
            rx: Mutex::new(rx),
            reader_join: Mutex::new(Some(reader_join)),
            roundtrip: Mutex::new(()),
            label,
        })
    }

    /// 1 行送って 1 行受け取る同期トランザクション (タイムアウト無し)。
    /// 互換性のために残してあるが、本番フローでは
    /// [`request_line_timeout`] を使うことで Python が無応答になっても
    /// 監視ループが止まらないようにする。
    pub fn request_line(&self, line: &str) -> Result<String, PythonProcError> {
        let _guard = self.roundtrip.lock().unwrap();
        self.send_line(line)?;
        self.recv_line()
    }

    /// タイムアウト付きの送信→受信 (フィルタ無し版)。
    ///
    /// `request_line_with_filter` を使えば前往復の遅延応答が混入しても
    /// 自動でスキップできるが、この互換 API はフィルタを持たないため、
    /// 「直前の往復がタイムアウトで終わったが応答だけ後から来た」状況では
    /// stale を返す可能性がある。新規コードは `request_line_with_filter`
    /// を使うこと。
    pub fn request_line_timeout(
        &self,
        line: &str,
        timeout: Duration,
    ) -> Result<String, PythonProcError> {
        self.request_line_with_filter(line, timeout, |_| true)
    }

    /// タイムアウト付きの送信→受信。レスポンスごとに `accept` を呼び、
    /// `false` を返した行は捨てて受信を続ける。
    ///
    /// 用途: 直前の `request_line_*` がタイムアウトで戻った後、Python 側が
    /// その応答を遅れて書き出してくると、次の往復ではその stale が
    /// チャネル先頭に詰まる。送信前に `drain_pending` で消費前の stale を
    /// 落としつつ、`accept` (例: 期待した `id` に一致するか) で送信後に
    /// 入ってきた stale もスキップしていく。これにより
    ///
    ///   - drain と recv の隙間に stale が滑り込んでも検出できる
    ///   - 永続的に Python が遅い場合は overall timeout で抜ける
    ///
    /// transport 層 (`python_proc`) は `accept` の中身を知らないので
    /// JSON / 独自プロトコル等は呼び出し側 (`run_cycle`) のクロージャに閉じる。
    pub fn request_line_with_filter<F>(
        &self,
        line: &str,
        timeout: Duration,
        accept: F,
    ) -> Result<String, PythonProcError>
    where
        F: Fn(&str) -> bool,
    {
        let _guard = self.roundtrip.lock().unwrap();
        self.drain_pending();
        self.send_line(line)?;
        let deadline = Instant::now() + timeout;
        loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return Err(PythonProcError::Timeout(timeout));
            }
            let response = self.recv_line_timeout(remaining)?;
            if accept(&response) {
                return Ok(response);
            }
            debug!(
                target: "python_proc",
                "[{}] filter rejected (likely stale): {}",
                self.label,
                response.trim()
            );
        }
    }

    /// チャネルに溜まっている未読イベントを破棄する。
    /// `roundtrip` ロック保持中に呼ばれる前提で、reader スレッドが
    /// 直前のタイムアウト応答を流し終えていればここでまとめて捨てる。
    fn drain_pending(&self) {
        let rx = self.rx.lock().unwrap();
        loop {
            match rx.try_recv() {
                Ok(ReaderEvent::Line(s)) => {
                    debug!(target: "python_proc", "[{}] drop stale: {}", self.label, s.trim());
                }
                Ok(ReaderEvent::Eof) | Ok(ReaderEvent::Io(_)) => {
                    // reader スレッドは Eof / Io を流した直後に終了するので、
                    // それ以降チャネルへ何も流れてこない。Disconnected を待つ
                    // までもなくここで切り上げる (次の send_line /
                    // recv_line_timeout が改めて Terminated を返す)。
                    break;
                }
                Err(_) => break,
            }
        }
    }

    /// 1 行 JSON を送信する (末尾 \n を自動付与)。
    pub fn send_line(&self, line: &str) -> Result<(), PythonProcError> {
        let mut stdin = self.stdin.lock().unwrap();
        stdin.write_all(line.as_bytes())?;
        if !line.ends_with('\n') {
            stdin.write_all(b"\n")?;
        }
        stdin.flush()?;
        debug!(target: "python_proc", "[{}] -> {}", self.label, redact_for_log(line));
        Ok(())
    }

    /// 1 行 JSON を受信する (タイムアウト無し、ブロッキング)。
    /// プロセスが死んでチャネルが切断された場合は
    /// [`PythonProcError::Terminated`] を返す。
    pub fn recv_line(&self) -> Result<String, PythonProcError> {
        let rx = self.rx.lock().unwrap();
        match rx.recv() {
            Ok(ReaderEvent::Line(s)) => {
                debug!(target: "python_proc", "[{}] <- {}", self.label, s.trim());
                Ok(s)
            }
            Ok(ReaderEvent::Eof) => Err(PythonProcError::Terminated),
            Ok(ReaderEvent::Io(e)) => Err(PythonProcError::SpawnFailed(e)),
            Err(_) => Err(PythonProcError::Terminated),
        }
    }

    /// タイムアウト付きで 1 行 JSON を受信する。
    ///
    /// - 応答到達: `Ok(line)`
    /// - `timeout` 経過: [`PythonProcError::Timeout`]
    /// - プロセス死亡 (stdout が EOF / チャネル切断): [`PythonProcError::Terminated`]
    /// - reader 側 I/O 失敗: [`PythonProcError::SpawnFailed`] (io::Error)
    pub fn recv_line_timeout(&self, timeout: Duration) -> Result<String, PythonProcError> {
        let rx = self.rx.lock().unwrap();
        match rx.recv_timeout(timeout) {
            Ok(ReaderEvent::Line(s)) => {
                debug!(target: "python_proc", "[{}] <- {}", self.label, s.trim());
                Ok(s)
            }
            Ok(ReaderEvent::Eof) => Err(PythonProcError::Terminated),
            Ok(ReaderEvent::Io(e)) => Err(PythonProcError::SpawnFailed(e)),
            Err(RecvTimeoutError::Timeout) => Err(PythonProcError::Timeout(timeout)),
            Err(RecvTimeoutError::Disconnected) => Err(PythonProcError::Terminated),
        }
    }

    /// プロセスを終了させる。Drop 時にも呼ばれるため二重 kill 安全。
    /// MonitorHandle::stop からも明示的に呼び、recv_line でブロック中の
    /// 監視スレッドを解放する。
    pub fn kill(&self) {
        if let Ok(mut child) = self.child.lock() {
            let _ = child.kill();
            let _ = child.wait();
        }
        // reader はブロッキング `send` を使うので、consumer がキューを引かない
        // まま child を kill するだけだと、send で詰まったまま join が永久に
        // 戻らない。受信側を切り替えて元の Receiver を drop し、reader 側の
        // send を Err(Disconnected) で起こすことで shutdown を進める。
        // 切り替え後の rx (空 + tx 無し) は recv で即 Disconnected を返し、
        // 上位で `Terminated` にマップされる (= プロセス死亡時の正規挙動)。
        if let Ok(mut rx_guard) = self.rx.lock() {
            let (_dead_tx, dead_rx) = mpsc::sync_channel::<ReaderEvent>(1);
            drop(_dead_tx);
            *rx_guard = dead_rx;
        }
        // child が落ちると stdout が EOF になり reader_loop が抜けるので、
        // ここで join して reader スレッドのリークを防ぐ。
        if let Ok(mut join_opt) = self.reader_join.lock() {
            if let Some(join) = join_opt.take() {
                let _ = join.join();
            }
        }
    }

    /// recognition プロセスを高レベル API で起動する。
    /// dev/release の差は `resolve_python_command` が吸収する。
    pub fn spawn_recognition<R: Runtime>(app: &AppHandle<R>) -> Result<Self, PythonProcError> {
        let cmd = resolve_python_command(app, "recognition")?;
        let arg_refs: Vec<&str> = cmd.args.iter().map(String::as_str).collect();
        Self::spawn("recognition", &cmd.program, &arg_refs, cmd.cwd.as_deref())
    }

    /// mortal プロセスを高レベル API で起動する。
    ///
    /// `model_path` が空文字列なら `--stub` モードで起動し、それ以外は
    /// `--model <path>` を渡す。`backend` は将来 Phase D で Python 側の
    /// モデルロードに反映する想定で、現時点ではコマンド構築には影響しない
    /// (将来の互換性のため API シグネチャに含めている)。
    pub fn spawn_mortal<R: Runtime>(
        app: &AppHandle<R>,
        model_path: &str,
        backend: InferenceBackend,
    ) -> Result<Self, PythonProcError> {
        // Phase D で Python 側 CLI に渡す予定。現状はバックエンドを参照しない。
        let _ = backend;
        let mut cmd = resolve_python_command(app, "mortal")?;
        // 設定値に前後空白が混ざっても有効なパスとして扱えるよう trim する。
        // trim せずに argparse へ渡すとファイルが見つからないと誤解されがち。
        let model_path = model_path.trim();
        if model_path.is_empty() {
            cmd.args.push("--stub".into());
        } else {
            cmd.args.push("--model".into());
            cmd.args.push(model_path.to_string());
        }
        let arg_refs: Vec<&str> = cmd.args.iter().map(String::as_str).collect();
        Self::spawn("mortal", &cmd.program, &arg_refs, cmd.cwd.as_deref())
    }
}

/// stdout を 1 行ずつ読み、`tx` へイベントとして流す reader スレッド本体。
/// プロセスが死ぬと `read_line` が 0 を返すので `Eof` を送って終了する。
///
/// ブロッキング `send` を使う: try_send で溢れを捨てると、求めている応答自体が
/// (consumer が一時的に遅れたタイミングで) drop されて偽タイムアウト → 不要な
/// 自動再起動を誘発する可能性がある (Codex P1 on PR #40)。bounded sync_channel
/// で send が満杯になれば reader が pipe を読まなくなり、Python 側の write が
/// pipe バッファ満杯でブロックする = 元の OS pipe ベースの自然な
/// バックプレッシャに戻る。`kill()` 側は rx を差し替えて Disconnected を発火させ、
/// 詰まった send を解放してから join するので shutdown の deadlock は起きない。
fn reader_loop(stdout: ChildStdout, tx: mpsc::SyncSender<ReaderEvent>, label: String) {
    let mut reader = BufReader::new(stdout);
    loop {
        let mut buf = String::new();
        let event = match reader.read_line(&mut buf) {
            Ok(0) => ReaderEvent::Eof,
            Ok(_) => ReaderEvent::Line(buf),
            Err(e) => ReaderEvent::Io(e),
        };
        let is_terminal = matches!(event, ReaderEvent::Eof | ReaderEvent::Io(_));
        // send は consumer が空きを作るまでブロックする。受信側が drop されて
        // いれば Err(SendError) で抜ける (= shutdown 経路)。
        if tx.send(event).is_err() {
            break;
        }
        if is_terminal {
            break;
        }
    }
    debug!(target: "python_proc", "[{}] reader thread exited", label);
}

/// `resolve_python_command` の結果。dev では `uv` を `python/` で起動し、
/// release では同梱 exe を直接起動するため、cwd の有無を含めてまとめて返す。
#[derive(Debug, Clone)]
pub struct ResolvedCommand {
    pub program: PathBuf,
    pub args: Vec<String>,
    pub cwd: Option<PathBuf>,
}

/// dev/release を一本化した Python サブプロセスのコマンド解決。
///
/// `label` には Python 側スクリプト名 (`recognition` / `mortal`) を渡す。
/// - dev (`debug_assertions`): PATH から `uv` を探し、
///   `uv run jantama-<label>` を `python/` ディレクトリで起動するための
///   コマンドを返す。`uv` が見つからない場合は `NotFound` を返す。
/// - release: tauri リソースディレクトリに同梱された
///   `jantama-<label>(.exe)` のパスを返す。同梱が無い場合は `NotFound`
///   を返し、Phase F (バンドル) が未完了であることを示すメッセージを
///   付ける (フォールバックはしない: release == 同梱 exe の契約を維持)。
#[cfg(debug_assertions)]
pub fn resolve_python_command<R: Runtime>(
    _app: &AppHandle<R>,
    label: &str,
) -> Result<ResolvedCommand, PythonProcError> {
    let uv = find_uv_executable()?;
    let cwd = resolve_python_project_dir().ok_or_else(|| {
        PythonProcError::NotFound(
            "python/pyproject.toml が見つかりません (uv run には python/ ディレクトリが必要)"
                .into(),
        )
    })?;
    Ok(ResolvedCommand {
        program: uv,
        args: vec!["run".into(), format!("jantama-{}", label)],
        cwd: Some(cwd),
    })
}

#[cfg(not(debug_assertions))]
pub fn resolve_python_command<R: Runtime>(
    app: &AppHandle<R>,
    label: &str,
) -> Result<ResolvedCommand, PythonProcError> {
    // Phase F (バンドル) で `jantama-<label>(.exe)` を同梱する想定。
    // Unix 系ではリソース同梱時に実行ビットが落ちると "Permission denied"
    // になるため、Phase F のバンドル手順で実行ビットを保つ (もしくは
    // Tauri Sidecar に切り替える) こと。
    let resource_dir = app
        .path()
        .resource_dir()
        .map_err(|e| PythonProcError::NotFound(format!("resource_dir: {}", e)))?;
    let exe_name = if cfg!(windows) {
        format!("jantama-{}.exe", label)
    } else {
        format!("jantama-{}", label)
    };
    let exe_path = resource_dir.join(&exe_name);
    // 事前に存在確認しないと、起動時は OS の汎用 ENOENT エラーになり
    // 「Phase F のバンドル待ち」だと気付きにくい。明示的に NotFound を返す。
    if !exe_path.is_file() {
        return Err(PythonProcError::NotFound(format!(
            "{} が同梱されていません (Phase F のバンドルが未完了の可能性)",
            exe_path.display()
        )));
    }
    // Unix では同梱時に実行ビットが落ちる経路があり、そのまま spawn すると
    // "Permission denied" になって Phase F の問題と気付きにくい。事前に
    // 実行ビットを確認して明示的なエラーで返す。
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if let Ok(meta) = std::fs::metadata(&exe_path) {
            if meta.permissions().mode() & 0o111 == 0 {
                return Err(PythonProcError::NotFound(format!(
                    "{} に実行権限がありません (Phase F のバンドル手順または chmod +x を確認)",
                    exe_path.display()
                )));
            }
        }
    }
    Ok(ResolvedCommand {
        program: exe_path,
        args: Vec::new(),
        cwd: None,
    })
}

/// PATH から `uv` 実行ファイルを探す。
/// `which` クレートに委譲することで OS のプロセス起動と同じ
/// ルックアップ規則 (Unix の実行ビット、Windows の PATHEXT による
/// `uv.exe` / `uv.cmd` 等) に従う。手書きの PATH 走査では非実行ファイル
/// を拾ったり Windows のラッパー (`.cmd`) を見落とす可能性があるため。
/// dev ビルド (および `cargo test`) でのみ呼び出される。
#[cfg(debug_assertions)]
fn find_uv_executable() -> Result<PathBuf, PythonProcError> {
    find_uv_executable_in(std::env::var_os("PATH").as_deref())
}

/// `find_uv_executable` の本体。テスト用に PATH を引数で差し替えられるよう
/// 分離している (グローバル `std::env::set_var` を触ると並列テストで競合するため)。
#[cfg(debug_assertions)]
fn find_uv_executable_in(path: Option<&std::ffi::OsStr>) -> Result<PathBuf, PythonProcError> {
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    which::which_in("uv", path, cwd).map_err(|e| {
        PythonProcError::NotFound(format!(
            "`uv` が PATH にありません ({}). https://docs.astral.sh/uv/ からインストールしてください",
            e
        ))
    })
}

/// 開発時に `python/pyproject.toml` がある場所を解決する。
/// `cargo tauri dev` ではバイナリが `src-tauri/target/debug/` に生成されるため、
/// cwd か exe からの上位ディレクトリ走査で `python/` を探す。
/// dev ビルド (および `cargo test`) でのみ呼び出される。
#[cfg(debug_assertions)]
fn resolve_python_project_dir() -> Option<PathBuf> {
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

impl Drop for PythonProcess {
    fn drop(&mut self) {
        self.kill();
    }
}

/// `debug!` ログ出力用に 1 行 JSON をサニタイズする。
///
/// frame リクエストには `image_b64` (画面キャプチャの base64) が乗るため、
/// そのままログに垂れ流すとスクリーンショットがファイル/ターミナルに溜まる。
/// JSON として解釈できれば該当フィールドをサイズだけ残して伏せ、解釈
/// できなければ元の文字列を返す。
fn redact_for_log(line: &str) -> String {
    let trimmed = line.trim();
    let mut value: serde_json::Value = match serde_json::from_str(trimmed) {
        Ok(v) => v,
        Err(_) => return trimmed.to_string(),
    };
    if let Some(obj) = value.as_object_mut() {
        if let Some(img) = obj.get_mut("image_b64") {
            let len = img.as_str().map(str::len).unwrap_or(0);
            *img = serde_json::Value::String(format!("<redacted {} bytes>", len));
        }
    }
    value.to_string()
}

// `find_uv_executable_in` は `debug_assertions` でだけ定義されるため、
// テストモジュールも同条件でゲートしないと `cargo test --release` で
// コンパイルエラーになる。
#[cfg(all(test, debug_assertions))]
mod tests {
    use super::*;
    use std::ffi::OsStr;

    /// PATH に `uv` がない状態では NotFound エラーが返ることを確認する。
    /// プロセス全体の `PATH` を書き換えると並列テストで競合するため、
    /// PATH を引数で受け取る `find_uv_executable_in` を直接呼んで
    /// グローバル状態に触れない。
    #[test]
    fn find_uv_returns_not_found_when_path_is_empty() {
        let result = find_uv_executable_in(Some(OsStr::new("")));
        assert!(matches!(result, Err(PythonProcError::NotFound(_))));
    }

    /// 即時応答する Python ワンライナーで、`request_line_timeout` が
    /// 設定したタイムアウト内に応答を取得できることを確認する。
    #[test]
    fn request_line_timeout_returns_response() {
        let Some(python) = which::which("python3")
            .ok()
            .or_else(|| which::which("python").ok())
        else {
            eprintln!("skipping: python3/python not on PATH");
            return;
        };
        let proc = PythonProcess::spawn(
            "echo",
            &python,
            &[
                "-u",
                "-c",
                "import sys\nfor line in sys.stdin:\n    sys.stdout.write(line)\n    sys.stdout.flush()\n",
            ],
            None,
        )
        .expect("spawn echo python");

        let resp = proc
            .request_line_timeout("hello\n", Duration::from_secs(5))
            .expect("response within timeout");
        assert_eq!(resp.trim(), "hello");
    }

    /// 応答を返さない Python に対して `request_line_timeout` が
    /// `Timeout` を返すことを確認する。
    #[test]
    fn request_line_timeout_fires_when_python_silent() {
        let Some(python) = which::which("python3")
            .ok()
            .or_else(|| which::which("python").ok())
        else {
            eprintln!("skipping: python3/python not on PATH");
            return;
        };
        let proc = PythonProcess::spawn(
            "silent",
            &python,
            &[
                "-u",
                "-c",
                // stdin から行を読みつつも何も書き返さない。stdin が閉じるまで
                // ぶら下がるので親側のタイムアウトを試せる。
                "import sys, time\nfor _ in sys.stdin:\n    time.sleep(60)\n",
            ],
            None,
        )
        .expect("spawn silent python");

        let result = proc.request_line_timeout("ping\n", Duration::from_millis(300));
        assert!(
            matches!(result, Err(PythonProcError::Timeout(_))),
            "got {:?}",
            result
        );
    }

    /// `request_line_with_filter` がフィルタで弾かれた応答をスキップして
    /// 受信を続け、許可された応答だけを返すことを確認する。
    /// stale 応答対策 (Codex review on PR #40) のリグレッションテスト。
    #[test]
    fn request_line_with_filter_skips_rejected_responses() {
        let Some(python) = which::which("python3")
            .ok()
            .or_else(|| which::which("python").ok())
        else {
            eprintln!("skipping: python3/python not on PATH");
            return;
        };
        // 1 行入力を受け取る前に "stale" を 2 行流し、入力が来たら "fresh" を返す。
        // フィルタが stale を捨てて fresh だけを採用できるかをチェック。
        let script = "import sys\n\
                      sys.stdout.write('stale-1\\n')\n\
                      sys.stdout.write('stale-2\\n')\n\
                      sys.stdout.flush()\n\
                      for line in sys.stdin:\n    \
                          sys.stdout.write('fresh\\n')\n    \
                          sys.stdout.flush()\n    \
                          break\n";
        let proc = PythonProcess::spawn("filter", &python, &["-u", "-c", script], None)
            .expect("spawn filter python");

        // stale が reader 側に流れ込むまで少し待つ。
        std::thread::sleep(Duration::from_millis(100));

        let resp = proc
            .request_line_with_filter("go\n", Duration::from_secs(5), |line| {
                line.trim() == "fresh"
            })
            .expect("got fresh response");
        assert_eq!(resp.trim(), "fresh");
    }

    /// reader がブロッキング send で詰まっている状態でも `kill()` が
    /// deadlock せずに戻ることを確認する。Codex P1 (PR #40) で try_send +
    /// drop から blocking send + kill 時 rx 差し替えに切り替えた挙動の
    /// リグレッションテスト。
    #[test]
    fn kill_returns_when_reader_is_blocked_on_send() {
        let Some(python) = which::which("python3")
            .ok()
            .or_else(|| which::which("python").ok())
        else {
            eprintln!("skipping: python3/python not on PATH");
            return;
        };
        // READER_QUEUE_BOUND を大きく超える行数を一気に書いて、reader を
        // blocking send で詰まらせる。consumer が引かないので channel は満杯。
        let burst = READER_QUEUE_BOUND * 4;
        let script = format!(
            "import sys\n\
             for i in range({}):\n    \
                 sys.stdout.write(f'line-{{i}}\\n')\n\
             sys.stdout.flush()\n\
             # stdin が閉じるまでぶら下がる (kill 経由で stdout が閉じる契機を作る)\n\
             sys.stdin.read()\n",
            burst
        );
        let proc = PythonProcess::spawn("burst", &python, &["-u", "-c", &script], None)
            .expect("spawn burst python");

        // Python が書き終わって reader が send で詰まるまで待つ。
        std::thread::sleep(Duration::from_millis(300));

        let started = Instant::now();
        proc.kill();
        let elapsed = started.elapsed();
        assert!(
            elapsed < Duration::from_secs(5),
            "kill() should not deadlock; took {:?}",
            elapsed
        );
    }

    /// 即終了する Python を相手に `request_line_timeout` が
    /// `Terminated` を返すことを確認する。
    #[test]
    fn request_line_timeout_returns_terminated_on_dead_process() {
        let Some(python) = which::which("python3")
            .ok()
            .or_else(|| which::which("python").ok())
        else {
            eprintln!("skipping: python3/python not on PATH");
            return;
        };
        let proc = PythonProcess::spawn(
            "dies",
            &python,
            &["-u", "-c", "import sys; sys.exit(0)"],
            None,
        )
        .expect("spawn dying python");

        // 子プロセスが終了して reader_loop が EOF を流すまで少し待つ
        std::thread::sleep(Duration::from_millis(200));
        let result = proc.request_line_timeout("anything\n", Duration::from_secs(2));
        assert!(
            matches!(
                result,
                Err(PythonProcError::Terminated) | Err(PythonProcError::SpawnFailed(_))
            ),
            "got {:?}",
            result
        );
    }
}
