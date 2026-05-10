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
use std::time::Duration;
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

        let (tx, rx) = mpsc::channel::<ReaderEvent>();
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

    /// タイムアウト付きの送信→受信。
    ///
    /// `timeout` を超えても応答が来なければ [`PythonProcError::Timeout`]
    /// を返す。応答が遅れている = 認識/推論が詰まっているケースの
    /// バックストップで、PRD §7.4「Mortal 推論失敗」「認識失敗」に対応。
    ///
    /// 直前の `request_line_timeout` がタイムアウトで戻った場合、Python 側は
    /// その応答を後から書き出してくる可能性がある。次の往復ではその応答が
    /// 本来欲しい応答より先にチャネルへ届き、`id` ミスマッチで永遠に
    /// 1 サイクルずれた応答を読み続ける羽目になるので、送信前にチャネルへ
    /// 溜まった「古い応答」を drain しておく。
    pub fn request_line_timeout(
        &self,
        line: &str,
        timeout: Duration,
    ) -> Result<String, PythonProcError> {
        let _guard = self.roundtrip.lock().unwrap();
        self.drain_pending();
        self.send_line(line)?;
        self.recv_line_timeout(timeout)
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
                    // reader が既に死んでいる場合はその痕跡を捨ててよい。
                    // 次の send_line / recv_line_timeout で改めて
                    // Terminated を返すことになる。
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
fn reader_loop(stdout: ChildStdout, tx: mpsc::Sender<ReaderEvent>, label: String) {
    let mut reader = BufReader::new(stdout);
    loop {
        let mut buf = String::new();
        match reader.read_line(&mut buf) {
            Ok(0) => {
                // 受信側が既にチャネルを drop している場合 (＝PythonProcess
                // が破棄された) は send が失敗するが、そのまま終了して問題ない。
                let _ = tx.send(ReaderEvent::Eof);
                break;
            }
            Ok(_) => {
                if tx.send(ReaderEvent::Line(buf)).is_err() {
                    // 受信側が消えていればこれ以上読む意味は無い。
                    break;
                }
            }
            Err(e) => {
                let _ = tx.send(ReaderEvent::Io(e));
                break;
            }
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
