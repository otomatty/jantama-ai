// Python サブプロセス管理
//
// PRD §8.1 全体構成図: Tauri (Rust) ↔ Python の間は stdin/stdout で
// JSON-lines 通信を行う。recognition / mortal の 2 プロセスを管理。
//
// Phase D で配線するスケルトン。現状は呼び出し側が未実装のため module 全体に
// dead_code 許可を付けている。
#![allow(dead_code)]

use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::Mutex;
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
    label: String,
}

impl PythonProcess {
    /// Python プロセスを起動する。
    ///
    /// `python_path`: 実行ファイル (PyInstaller でバンドル後の .exe を指定するか、
    ///                開発時は uv 経由で `uv run python -m <module>` を使う想定)。
    /// `args`: コマンドライン引数。
    pub fn spawn(
        label: impl Into<String>,
        python_path: &PathBuf,
        args: &[&str],
    ) -> Result<Self, PythonProcError> {
        let label = label.into();
        info!(target: "python_proc", "spawning {} ({})", label, python_path.display());

        let mut child = Command::new(python_path)
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            // stderr は read しないため pipe するとバッファ満杯でデッドロックし得る。
            // Phase D で構造化ログとして取り込むまでは親プロセスへ継承する。
            .stderr(Stdio::inherit())
            .spawn()?;

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
            label,
        })
    }

    /// 1 行 JSON を送信する (末尾 \n を自動付与)。
    pub fn send_line(&self, line: &str) -> Result<(), PythonProcError> {
        let mut stdin = self.stdin.lock().unwrap();
        stdin.write_all(line.as_bytes())?;
        if !line.ends_with('\n') {
            stdin.write_all(b"\n")?;
        }
        stdin.flush()?;
        debug!(target: "python_proc", "[{}] -> {}", self.label, line);
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
    pub fn kill(&self) {
        if let Ok(mut child) = self.child.lock() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

impl Drop for PythonProcess {
    fn drop(&mut self) {
        self.kill();
    }
}
