/**
 * 雀魂AIアシスタント 共通型定義
 *
 * PRD §6 (データモデル) に対応する型を定義する。
 */

// ============================================================
// 状態管理
// ============================================================

/** PRD §3.2 に定義された状態遷移 */
export type AppPhase =
  | "uninitialized" // 未設定
  | "idle" // 待機中
  | "watching_no_board" // 監視中・盤面なし
  | "watching_recommend" // 監視中・推奨表示中
  | "error"; // エラー

// ============================================================
// 設定 (AppSettings) - PRD §6.2
// ============================================================

export interface AppSettings {
  capture_target_window_id: string | null;
  capture_target_window_title: string | null;
  mortal_model_path: string | null;
  window_position: { x: number; y: number } | null;
  window_size: { width: number; height: number } | null;
  data_retention_days: {
    inference_log: number;
    tile_image: number;
    error_log: number;
  };
  hotkey_settings?: Record<string, string>;
}

export const DEFAULT_SETTINGS: AppSettings = {
  capture_target_window_id: null,
  capture_target_window_title: null,
  mortal_model_path: null,
  window_position: null,
  window_size: null,
  data_retention_days: {
    inference_log: 30,
    tile_image: 7,
    error_log: 90,
  },
};

// ============================================================
// 画面キャプチャ対象ウィンドウ情報
// ============================================================

export interface CaptureWindow {
  id: string;
  title: string;
  app_name: string | null;
  is_minimized: boolean;
}

// ============================================================
// 推論結果
// ============================================================

/** PRD §4.1 F-09 に定義されたアクション種別 */
export type ActionType =
  | "discard" // 打牌
  | "riichi" // リーチ
  | "chi" // チー
  | "pon" // ポン
  | "kan" // カン
  | "ron" // ロン
  | "tsumo" // ツモ
  | "pass"; // パス

export interface RecommendationCandidate {
  /** 牌表記 (例: "6m", "1z") またはアクション識別子 */
  tile?: string;
  action_type: ActionType;
  expected_value: number;
  /** 任意の補足文字列 (例: "鳴き対象牌: 5p") */
  detail?: string;
}

export interface InferenceResult {
  recommended: RecommendationCandidate;
  candidates: RecommendationCandidate[];
  /** 推論実行時刻 (ISO8601) */
  timestamp: string;
}

// ============================================================
// 盤面認識結果サマリー
// ============================================================

export interface GameBoardSummary {
  hand: string[]; // 例: ["1m", "2m", ...]
  self_wind: "東" | "南" | "西" | "北";
  round_wind: "東" | "南" | "西" | "北";
  turn: number;
  dora_indicators: string[];
}

// ============================================================
// ステータス情報 (画面上部に表示する内容)
// ============================================================

export interface MonitoringStatus {
  watching: boolean;
  capture_target_window_title: string | null;
  last_recognized_at: string | null; // ISO8601
}

// ============================================================
// エラー情報
// ============================================================

export interface AppError {
  type: "recognition" | "inference" | "capture" | "config" | "unknown";
  message: string;
  occurred_at: string; // ISO8601
}
