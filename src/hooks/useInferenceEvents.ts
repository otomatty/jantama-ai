import { useEffect } from "react";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import { runStubInference } from "@/lib/tauriCommands";
import type { AppError, GameBoardSummary, InferenceResult } from "@/types";

/**
 * Rust 側 (`src-tauri/src/monitor.rs`) が emit する `inference-result` の payload。
 * `InferenceEventPayload` (Rust) と同形。
 */
interface InferenceEventPayload {
  inference: InferenceResult;
  board: GameBoardSummary | null;
}

interface UseInferenceEventsOptions {
  /** 監視 ON の間だけ listener を購読する */
  watching: boolean;
  /** `inference-result` 受信時 / ブラウザ単体起動時のスタブ受信時に呼ばれる */
  onInference: (inference: InferenceResult, board: GameBoardSummary | null) => void;
  /** `recognition-error` 受信時に呼ばれる */
  onError: (error: AppError) => void;
}

const STUB_INTERVAL_MS = 5000;

function isTauri(): boolean {
  const w = window as unknown as { __TAURI_INTERNALS__?: unknown };
  return typeof w.__TAURI_INTERNALS__ !== "undefined";
}

/**
 * 監視中に Rust から飛んでくる Tauri Event を listen し、
 * 推論結果 / 認識エラーを上位に流す。
 *
 * - Tauri 環境では `@tauri-apps/api/event` の `listen` で
 *   `inference-result` / `recognition-error` を購読。
 * - ブラウザ単体起動 (`npm run dev`) ではフォールバックとして
 *   従来の 5 秒スタブ (`runStubInference`) を回す。
 * - `watching` が false に戻った時点、もしくは unmount 時に
 *   listener / interval を確実に解除する。
 */
export function useInferenceEvents({
  watching,
  onInference,
  onError,
}: UseInferenceEventsOptions): void {
  useEffect(() => {
    if (!watching) return;

    if (!isTauri()) {
      // ブラウザ単体動作確認用: 5 秒に 1 回スタブ推論を流す
      let cancelled = false;
      const fire = async () => {
        const { inference, board } = await runStubInference();
        if (!cancelled) onInference(inference, board);
      };
      fire();
      const id = setInterval(fire, STUB_INTERVAL_MS);
      return () => {
        cancelled = true;
        clearInterval(id);
      };
    }

    let cancelled = false;
    let inferenceUnlisten: UnlistenFn | null = null;
    let errorUnlisten: UnlistenFn | null = null;

    (async () => {
      const [inferenceUn, errorUn] = await Promise.all([
        listen<InferenceEventPayload>("inference-result", (event) => {
          onInference(event.payload.inference, event.payload.board ?? null);
        }),
        listen<AppError>("recognition-error", (event) => {
          onError(event.payload);
        }),
      ]);
      // listen の登録中に watching が false に戻った場合は即解除
      if (cancelled) {
        inferenceUn();
        errorUn();
        return;
      }
      inferenceUnlisten = inferenceUn;
      errorUnlisten = errorUn;
    })();

    return () => {
      cancelled = true;
      inferenceUnlisten?.();
      errorUnlisten?.();
    };
  }, [watching, onInference, onError]);
}
