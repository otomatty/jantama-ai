-- 雀魂AIアシスタント DB 初期マイグレーション
-- PRD §6 (データモデル) に基づく
-- tauri-plugin-sql で適用される

PRAGMA foreign_keys = ON;

-- ============================================================
-- 推論履歴 (PRD §6.2 InferenceLog)
-- 30日保持。F-11 で自動削除する。
-- ============================================================
CREATE TABLE IF NOT EXISTS inference_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,                       -- ISO8601
    game_state_snapshot TEXT NOT NULL,             -- 天鳳JSON
    mortal_output TEXT NOT NULL,                   -- 推奨候補・期待値 JSON
    recommended_action_type TEXT NOT NULL          -- 'discard'|'riichi'|'chi'|'pon'|'kan'|'ron'|'tsumo'|'pass'
);

CREATE INDEX IF NOT EXISTS idx_inference_log_timestamp
    ON inference_log (timestamp);

-- ============================================================
-- 牌画像サンプル (PRD §6.2 TileImageSample)
-- 7日保持。F-12 で自動削除する。
-- ============================================================
CREATE TABLE IF NOT EXISTS tile_image_sample (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    image_file_path TEXT NOT NULL,
    recognition_result TEXT,
    confidence_score REAL
);

CREATE INDEX IF NOT EXISTS idx_tile_image_sample_timestamp
    ON tile_image_sample (timestamp);

-- ============================================================
-- エラーログ (PRD §6.2 ErrorLog)
-- 90日保持。F-13 で自動削除する。
-- ============================================================
CREATE TABLE IF NOT EXISTS error_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    error_type TEXT NOT NULL,                       -- 'recognition'|'inference'|'capture'|'config'|'unknown'
    message TEXT NOT NULL,
    stack_trace TEXT,
    related_game_state TEXT                         -- 任意
);

CREATE INDEX IF NOT EXISTS idx_error_log_timestamp
    ON error_log (timestamp);
