/**
 * Tauri Backend (Rust) との RPC ラッパー。
 *
 * Tauri コマンド名と対応する Rust 関数 (src-tauri/src/lib.rs) は同名で揃える。
 * 実行環境がブラウザの場合 (Vite 単体起動時) はスタブ値を返してフォールバックする。
 */

import { invoke } from "@tauri-apps/api/core";
import type { AppSettings, CaptureWindow, GameBoardSummary, InferenceResult } from "@/types";
import { EMPTY_ROI_CALIBRATION } from "@/types";
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
 *
 * Rust 側の serde は `roi_calibration` の欠落を `Default` でフォールバックさせる
 * (= 全領域 None) が、古い設定ファイルや手作業で編集された JSON が
 * `roi_calibration` 自体を持たないケースに備えて、TS 側でも欠落フィールドを
 * 明示的に `null` で埋めて返す。これがないと CalibrationScreen 側の
 * `=== null` 判定が `undefined` をすり抜けて誤動作する。
 */
export async function loadSettings(): Promise<AppSettings | null> {
  if (!isTauri()) return null;
  const raw = await invoke<AppSettings | null>("load_settings");
  if (!raw) return null;
  return normalizeSettings(raw);
}

function normalizeSettings(raw: AppSettings): AppSettings {
  const roi = raw.roi_calibration ?? EMPTY_ROI_CALIBRATION;
  return {
    ...raw,
    roi_calibration: {
      hand: roi.hand ?? null,
      doras: roi.doras ?? null,
      rivers: {
        self: roi.rivers?.self ?? null,
        right: roi.rivers?.right ?? null,
        across: roi.rivers?.across ?? null,
        left: roi.rivers?.left ?? null,
      },
      round_info: roi.round_info ?? null,
      self_wind: roi.self_wind ?? null,
    },
  };
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

/**
 * 設定画面の ROI キャリブレーション UI 用に、対象ウィンドウを 1 枚キャプチャして
 * `<canvas>` に描画できるよう base64 PNG で返す (issue #10)。
 *
 * ブラウザ実行 (Tauri 外) の場合は、デザイン版の動作確認用として
 * 仮の透明 1x1 PNG を返す。実際の雀魂ウィンドウは Tauri ビルドでのみ取得できる。
 */
export interface CalibrationCapture {
  width: number;
  height: number;
  image_b64: string;
}

const STUB_CAPTURE_PNG_B64 =
  // 1920x1080 風のサイズで返したいが、ブラウザ動作時は雀魂が無いので
  // 単色グラデーションの軽量 PNG を埋め込んで UI を確認できるようにする。
  // (1x1 PNG transparent → CSS で stretching: 雀魂を起動した実機で再確認すること)
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNgYAAAAAMAAWgmWQ0AAAAASUVORK5CYII=";

export async function captureWindowForCalibration(windowId: string): Promise<CalibrationCapture> {
  if (!isTauri()) {
    return {
      width: 1920,
      height: 1080,
      image_b64: STUB_CAPTURE_PNG_B64,
    };
  }
  return invoke<CalibrationCapture>("capture_window_for_calibration", {
    windowId,
  });
}
