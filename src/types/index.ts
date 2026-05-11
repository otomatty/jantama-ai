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

export type InferenceBackend = "rocm" | "cpu";

/**
 * ROI 矩形 (PRD §9 リスク表 / issue #10)。
 *
 * 雀魂のウィンドウサイズ・解像度に依存しないように、左上原点での
 * 0.0〜1.0 比率で保存する。キャプチャサイズが変わってもそのまま使える。
 */
export interface RoiRect {
  x: number;
  y: number;
  w: number;
  h: number;
}

/** ROI キャリブレーション領域 ID。UI とロジックで共有する。 */
export type RoiRegionId =
  | "hand"
  | "doras"
  | "river_self"
  | "river_right"
  | "river_across"
  | "river_left"
  | "meld_self"
  | "meld_right"
  | "meld_across"
  | "meld_left"
  | "round_info"
  | "self_wind"
  | "scores"
  | "turn_counter";

/** ROI キャリブレーション結果 (issue #10 / #12 / #14)。未指定領域は `null`。 */
export interface RoiCalibration {
  hand: RoiRect | null;
  doras: RoiRect | null;
  rivers: {
    self: RoiRect | null;
    right: RoiRect | null;
    across: RoiRect | null;
    left: RoiRect | null;
  };
  /**
   * 4 家分の副露 (鳴き) 領域 (issue #14)。雀魂では自家は手牌の右、他家は
   * 対応する位置に表示される。加槓の積み牌を捉えるため、ROI は upright 方向の
   * 1 段ぶん上に余白を含めて切り出すこと。
   */
  melds: {
    self: RoiRect | null;
    right: RoiRect | null;
    across: RoiRect | null;
    left: RoiRect | null;
  };
  round_info: RoiRect | null;
  self_wind: RoiRect | null;
  /** 4 家分の点棒数字が並んだ帯。内側を 4 等分して OCR (issue #12)。 */
  scores: RoiRect | null;
  /** 巡目カウンタの数字 (issue #12)。 */
  turn_counter: RoiRect | null;
}

export const EMPTY_ROI_CALIBRATION: RoiCalibration = {
  hand: null,
  doras: null,
  rivers: { self: null, right: null, across: null, left: null },
  melds: { self: null, right: null, across: null, left: null },
  round_info: null,
  self_wind: null,
  scores: null,
  turn_counter: null,
};

export interface AppSettings {
  capture_target_window_id: string | null;
  capture_target_window_title: string | null;
  mortal_model_path: string | null;
  inference_backend: InferenceBackend;
  show_llm_reason: boolean;
  show_danger_safe: boolean;
  window_position: { x: number; y: number } | null;
  window_size: { width: number; height: number } | null;
  data_retention_days: {
    inference_log: number;
    tile_image: number;
    error_log: number;
  };
  hotkey_settings?: Record<string, string>;
  /** ROI キャリブレーション結果 (issue #10)。未キャリブレーション時は全 `null`。 */
  roi_calibration: RoiCalibration;
}

export const DEFAULT_SETTINGS: AppSettings = {
  capture_target_window_id: null,
  capture_target_window_title: null,
  mortal_model_path: null,
  inference_backend: "rocm",
  show_llm_reason: true,
  show_danger_safe: true,
  window_position: null,
  window_size: null,
  data_retention_days: {
    inference_log: 30,
    tile_image: 7,
    error_log: 90,
  },
  roi_calibration: EMPTY_ROI_CALIBRATION,
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
  /** 表示用アクション名 (例: "リーチ" / "ダマ" / "スルー") */
  action_label?: string;
  action_type: ActionType;
  expected_value: number;
  /** 確信度 (0..1) */
  probability?: number;
  /** 任意の補足文字列 */
  detail?: string;
}

export interface DangerTile {
  tile: string;
  level: "high" | "mid" | "low";
}

export interface InferenceResult {
  /** 推論実行時刻 (ISO8601) */
  timestamp: string;
  recommended: RecommendationCandidate;
  candidates: RecommendationCandidate[];
  /** プライマリ表示文 (例: "6m を切る") */
  primary_label?: string;
  /** LLM 生成の打牌理由 (S-01) */
  reason?: string;
  /** S-02: 危険牌 */
  danger?: DangerTile[];
  /** S-02: 安全牌 */
  safe?: string[];
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
  /** 持ち点 */
  score?: number;
  /** 局名 (例: "東1局") */
  round_label?: string;
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

// ============================================================
// Python サブプロセスからの構造化ログ
// ============================================================

/**
 * Rust 側 (`src-tauri/src/python_proc.rs`) が emit する `python-log` の payload。
 * Python の stderr を `{level}\t{logger}\t{message}` の TSV としてパースした結果。
 * S-03 デバッグビューでログテーブルを描画する際の入力になる。
 */
export interface PythonLogEvent {
  /** 発生プロセス識別子 (例: "recognition", "mortal") */
  source: string;
  /** Python logging の levelname (INFO / WARNING / ERROR / DEBUG など) */
  level: string;
  /** logger 名 */
  logger: string;
  /** 1 行に正規化されたメッセージ (改行は `\\n` にエスケープ済み) */
  message: string;
  /** Rust 側が受信した時刻 (RFC3339) */
  timestamp: string;
}
