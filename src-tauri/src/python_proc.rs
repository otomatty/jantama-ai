// Python サブプロセス管理
//
// PRD §8.1 全体構成図: Tauri (Rust) ↔ Python の間は stdin/stdout で
// JSON-lines 通信を行う。recognition / mortal の 2 プロセスを管理。

use crate::types::InferenceBackend;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::Mutex;
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
    #[error("JSON parse error: {0}")]
    Json(#[from] serde_json::Error),
}

pub struct PythonProcess {
    child: Mutex<Child>,
    stdin: Mutex<ChildStdin>,
    stdout: Mutex<BufReader<ChildStdout>>,
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

        Ok(Self {
            child: Mutex::new(child),
            stdin: Mutex::new(stdin),
            stdout: Mutex::new(BufReader::new(stdout)),
            roundtrip: Mutex::new(()),
            label,
        })
    }

    /// 1 行送って 1 行受け取る同期トランザクション。
    ///
    /// 複数スレッドからこのメソッドを呼んでも、`roundtrip` ロックで
    /// 直列化されるので応答の取り違えは発生しない。スモークと監視ループの
    /// ように同一プロセスへ並行アクセスするコードはこちらを使う。
    pub fn request_line(&self, line: &str) -> Result<String, PythonProcError> {
        let _guard = self.roundtrip.lock().unwrap();
        self.send_line(line)?;
        self.recv_line()
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

    /// 1 行 JSON を受信する。
    pub fn recv_line(&self) -> Result<String, PythonProcError> {
        let mut stdout = self.stdout.lock().unwrap();
        let mut buf = String::new();
        let n = stdout.read_line(&mut buf)?;
        if n == 0 {
            return Err(PythonProcError::Terminated);
        }
        debug!(target: "python_proc", "[{}] <- {}", self.label, buf.trim());
        Ok(buf)
    }

    /// プロセスを終了させる。Drop 時にも呼ばれるため二重 kill 安全。
    /// MonitorHandle::stop からも明示的に呼び、recv_line でブロック中の
    /// 監視スレッドを解放する。
    pub fn kill(&self) {
        if let Ok(mut child) = self.child.lock() {
            let _ = child.kill();
            let _ = child.wait();
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
}
