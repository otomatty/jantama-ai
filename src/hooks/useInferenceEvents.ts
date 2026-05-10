import { useEffect } from "react";
import { isTauri } from "@tauri-apps/api/core";
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

function toAppError(kind: AppError["type"], fallbackMessage: string, e: unknown): AppError {
  return {
    type: kind,
    message: e instanceof Error ? e.message : fallbackMessage,
    occurred_at: new Date().toISOString(),
  };
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
        try {
          const { inference, board } = await runStubInference();
          if (!cancelled) onInference(inference, board);
        } catch (e) {
          if (!cancelled) {
            onError(toAppError("unknown", "stub inference failed", e));
          }
        }
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
      // Promise.all で並列登録すると 2 つ目が reject した時に
      // 1 つ目の UnlistenFn が捨てられて listener がリークするため、
      // 順次登録して部分登録のクリーンアップを担保する。
      let tempInferenceUn: UnlistenFn | null = null;
      try {
        tempInferenceUn = await listen<InferenceEventPayload>(
          "inference-result",
          (event) => {
            // 監視 OFF と listener 解除の競合で古いイベントが届くことがあるため
            // コールバック側でも cancelled を確認する
            if (cancelled) return;
            onInference(event.payload.inference, event.payload.board ?? null);
          },
        );
        if (cancelled) {
          tempInferenceUn();
          return;
        }
        const tempErrorUn = await listen<AppError>("recognition-error", (event) => {
          if (cancelled) return;
          onError(event.payload);
        });
        if (cancelled) {
          tempInferenceUn();
          tempErrorUn();
          return;
        }
        inferenceUnlisten = tempInferenceUn;
        errorUnlisten = tempErrorUn;
      } catch (e) {
        tempInferenceUn?.();
        if (!cancelled) {
          onError(toAppError("unknown", "failed to subscribe tauri events", e));
        }
      }
    })();

    return () => {
      cancelled = true;
      inferenceUnlisten?.();
      errorUnlisten?.();
    };
  }, [watching, onInference, onError]);
}
