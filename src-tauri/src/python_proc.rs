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
        if model_path.trim().is_empty() {
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
///   `jantama-<label>(.exe)` のパスを返す (Phase F で同梱)。
pub fn resolve_python_command<R: Runtime>(
    app: &AppHandle<R>,
    label: &str,
) -> Result<ResolvedCommand, PythonProcError> {
    #[cfg(debug_assertions)]
    {
        // release ビルド以外では AppHandle は使わない。
        let _ = app;
        let uv = find_uv_executable()?;
        let cwd = resolve_python_project_dir().ok_or_else(|| {
            PythonProcError::NotFound(
                "python/pyproject.toml が見つかりません (dev 起動には python/ ディレクトリが必要)"
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
    {
        let resource_dir = app
            .path()
            .resource_dir()
            .map_err(|e| PythonProcError::NotFound(format!("resource_dir: {}", e)))?;
        let exe_name = if cfg!(windows) {
            format!("jantama-{}.exe", label)
        } else {
            format!("jantama-{}", label)
        };
        Ok(ResolvedCommand {
            program: resource_dir.join(exe_name),
            args: Vec::new(),
            cwd: None,
        })
    }
}

/// PATH から `uv` 実行ファイルを探す。Windows では `uv.exe` も対象にする。
/// 見つからない場合は `PythonProcError::NotFound` を返す。
/// dev ビルド (および `cargo test`) でのみ呼び出される。
#[cfg(debug_assertions)]
fn find_uv_executable() -> Result<PathBuf, PythonProcError> {
    let names: &[&str] = if cfg!(windows) {
        &["uv.exe", "uv"]
    } else {
        &["uv"]
    };
    if let Some(paths) = std::env::var_os("PATH") {
        for dir in std::env::split_paths(&paths) {
            for name in names {
                let candidate = dir.join(name);
                if candidate.is_file() {
                    return Ok(candidate);
                }
            }
        }
    }
    Err(PythonProcError::NotFound(
        "`uv` が PATH にありません。https://docs.astral.sh/uv/ からインストールしてください".into(),
    ))
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

#[cfg(test)]
mod tests {
    use super::*;

    /// PATH に `uv` がない状態では NotFound エラーが返ることを確認する。
    /// 元の PATH は revert する。
    #[test]
    fn find_uv_returns_not_found_when_path_is_empty() {
        let original = std::env::var_os("PATH");
        // SAFETY: テスト並列実行時に他テストが PATH を読むと壊れるが、
        // 本クレートは PATH をテスト目的で参照する箇所が他にないため、
        // 明示的にこのテストでだけ操作する。
        std::env::set_var("PATH", "");
        let result = find_uv_executable();
        if let Some(p) = original {
            std::env::set_var("PATH", p);
        } else {
            std::env::remove_var("PATH");
        }
        assert!(matches!(result, Err(PythonProcError::NotFound(_))));
    }
}
