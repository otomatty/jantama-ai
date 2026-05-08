/**
 * Tauri Backend (Rust) との RPC ラッパー。
 *
 * Tauri コマンド名と対応する Rust 関数 (src-tauri/src/lib.rs) は同名で揃える。
 * 実行環境がブラウザの場合 (Vite 単体起動時) はスタブ値を返してフォールバックする。
 */

import { invoke } from "@tauri-apps/api/core";
import type {
  AppSettings,
  CaptureWindow,
  GameBoardSummary,
  InferenceResult,
} from "@/types";
import { nextStubScenario } from "@/lib/scenarios";

function isTauri(): boolean {
  const w = window as unknown as { __TAURI_INTERNALS__?: unknown };
  return typeof w.__TAURI_INTERNALS__ !== "undefined";
}

/**
 * 起動中のウィンドウ一覧を取得する。設定画面で雀魂のウィンドウを選ぶときに使う。
 */
export async function listCaptureWindows(): Promise<CaptureWindow[]> {
  if (!isTauri()) {
    return [
      {
        id: "stub-1",
        title: "雀魂 - Mahjong Soul (Steam)",
        app_name: "jantama.exe",
        is_minimized: false,
      },
      {
        id: "stub-2",
        title: "Mahjong Soul - Google Chrome",
        app_name: "chrome.exe",
        is_minimized: false,
      },
    ];
  }
  return invoke<CaptureWindow[]>("list_capture_windows");
}

/**
 * 設定の取得・保存。tauri-plugin-store にて永続化。
 */
export async function loadSettings(): Promise<AppSettings | null> {
  if (!isTauri()) return null;
  return invoke<AppSettings | null>("load_settings");
}

export async function saveSettings(settings: AppSettings): Promise<void> {
  if (!isTauri()) return;
  return invoke<void>("save_settings", { settings });
}

/**
 * 監視ループの開始/停止。Rust 側で別スレッドを起動し、
 * 結果は Tauri Event 経由 (`inference-result`, `recognition-error`) で返ってくる。
 */
export async function startMonitoring(): Promise<void> {
  if (!isTauri()) return;
  return invoke<void>("start_monitoring");
}

export async function stopMonitoring(): Promise<void> {
  if (!isTauri()) return;
  return invoke<void>("stop_monitoring");
}

/**
 * デバッグ・E2E 確認用: ダミー推論を 1 回実行して結果を返す。
 * ブラウザ実行時は scenarios のローテーションを返す。
 *
 * Tauri モードではバックエンドが盤面サマリ (board) を返すようになるまで
 * board は `null` を返す。以前は scenarios.board を併用していたが、
 * 5 秒毎のポーリングで推論ペイロードと無関係な手牌・局・巡目が
 * 切り替わってしまい、UI 上で推奨と盤面が乖離するため廃止した。
 */
export async function runStubInference(): Promise<{
  inference: InferenceResult;
  board: GameBoardSummary | null;
}> {
  if (!isTauri()) {
    return nextStubScenario();
  }
  const inference = await invoke<InferenceResult>("run_stub_inference");
  return { inference, board: null };
}
