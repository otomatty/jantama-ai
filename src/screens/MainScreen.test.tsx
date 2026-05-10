import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { act, render, screen } from "@testing-library/react";
import { MainScreen } from "./MainScreen";
import { INITIAL_APP_STATE } from "@/state/appState";
import { SCENARIO_FIXTURES } from "@/lib/scenarios";
import type { AppError, GameBoardSummary, InferenceResult } from "@/types";

// `useInferenceEvents` 内部の `@tauri-apps/api/event#listen` を差し替えて、
// 任意のタイミングで Rust 側からの emit を再現できるようにする。
type Handler<T> = (event: { payload: T }) => void;
const handlers = {
  inference: null as Handler<{ inference: InferenceResult; board: GameBoardSummary | null }> | null,
  error: null as Handler<AppError> | null,
};

vi.mock("@tauri-apps/api/event", () => ({
  listen: vi.fn(async (event: string, handler: Handler<unknown>) => {
    if (event === "inference-result") {
      handlers.inference = handler as typeof handlers.inference;
    } else if (event === "recognition-error") {
      handlers.error = handler as typeof handlers.error;
    }
    return () => {
      if (event === "inference-result") handlers.inference = null;
      if (event === "recognition-error") handlers.error = null;
    };
  }),
}));

function setTauriEnv(enabled: boolean) {
  const w = window as unknown as { __TAURI_INTERNALS__?: unknown };
  if (enabled) {
    w.__TAURI_INTERNALS__ = {};
  } else {
    delete w.__TAURI_INTERNALS__;
  }
}

async function flushAsync() {
  // listen() が解決した直後の state 反映を待つ
  await act(async () => {
    await Promise.resolve();
  });
}

describe("MainScreen", () => {
  beforeEach(() => {
    handlers.inference = null;
    handlers.error = null;
    setTauriEnv(false);
  });

  afterEach(() => {
    setTauriEnv(false);
  });

  it("phase が idle のとき待機メッセージを表示", () => {
    render(
      <MainScreen
        state={{ ...INITIAL_APP_STATE, phase: "idle" }}
        onOpenSettings={vi.fn()}
        onMonitoringChange={vi.fn()}
        onInferenceUpdate={vi.fn()}
        onRecognitionError={vi.fn()}
      />,
    );
    expect(screen.getByText("対局を待機中")).toBeInTheDocument();
  });

  it("推論結果がある場合 Hero レイアウトに primary_label と EV を表示", () => {
    const fixture = SCENARIO_FIXTURES.dahai;
    render(
      <MainScreen
        state={{
          ...INITIAL_APP_STATE,
          phase: "watching_recommend",
          monitoring: {
            watching: true,
            capture_target_window_title: "雀魂",
            last_recognized_at: null,
          },
          inference: fixture.inference,
          board: fixture.board,
        }}
        onOpenSettings={vi.fn()}
        onMonitoringChange={vi.fn()}
        onInferenceUpdate={vi.fn()}
        onRecognitionError={vi.fn()}
      />,
    );
    expect(screen.getByText("6m を切る")).toBeInTheDocument();
    expect(screen.getByText("+0.32")).toBeInTheDocument();
    expect(screen.getByText("RANK 1")).toBeInTheDocument();
  });

  it("Tauri 環境で inference-result を受信すると onInferenceUpdate に payload を渡す", async () => {
    setTauriEnv(true);
    const fixture = SCENARIO_FIXTURES.dahai;
    const onInferenceUpdate = vi.fn();
    render(
      <MainScreen
        state={{
          ...INITIAL_APP_STATE,
          phase: "watching_no_board",
          monitoring: {
            watching: true,
            capture_target_window_title: "雀魂",
            last_recognized_at: null,
          },
        }}
        onOpenSettings={vi.fn()}
        onMonitoringChange={vi.fn()}
        onInferenceUpdate={onInferenceUpdate}
        onRecognitionError={vi.fn()}
      />,
    );
    await flushAsync();
    expect(handlers.inference).not.toBeNull();

    act(() => {
      handlers.inference!({ payload: { inference: fixture.inference, board: fixture.board } });
    });
    expect(onInferenceUpdate).toHaveBeenCalledWith(fixture.inference, fixture.board);
  });

  it("Tauri 環境で recognition-error を受信すると onRecognitionError に AppError を渡す", async () => {
    setTauriEnv(true);
    const onRecognitionError = vi.fn();
    render(
      <MainScreen
        state={{
          ...INITIAL_APP_STATE,
          phase: "watching_no_board",
          monitoring: {
            watching: true,
            capture_target_window_title: "雀魂",
            last_recognized_at: null,
          },
        }}
        onOpenSettings={vi.fn()}
        onMonitoringChange={vi.fn()}
        onInferenceUpdate={vi.fn()}
        onRecognitionError={onRecognitionError}
      />,
    );
    await flushAsync();
    expect(handlers.error).not.toBeNull();

    const errorPayload: AppError = {
      type: "recognition",
      message: "frame decode failed",
      occurred_at: "2026-01-01T00:00:00Z",
    };
    act(() => {
      handlers.error!({ payload: errorPayload });
    });
    expect(onRecognitionError).toHaveBeenCalledWith(errorPayload);
  });

  it("監視 OFF に戻ると Tauri Event の listener を unsubscribe する", async () => {
    setTauriEnv(true);
    const { rerender } = render(
      <MainScreen
        state={{
          ...INITIAL_APP_STATE,
          phase: "watching_no_board",
          monitoring: {
            watching: true,
            capture_target_window_title: "雀魂",
            last_recognized_at: null,
          },
        }}
        onOpenSettings={vi.fn()}
        onMonitoringChange={vi.fn()}
        onInferenceUpdate={vi.fn()}
        onRecognitionError={vi.fn()}
      />,
    );
    await flushAsync();
    expect(handlers.inference).not.toBeNull();
    expect(handlers.error).not.toBeNull();

    rerender(
      <MainScreen
        state={{
          ...INITIAL_APP_STATE,
          phase: "idle",
          monitoring: {
            watching: false,
            capture_target_window_title: "雀魂",
            last_recognized_at: null,
          },
        }}
        onOpenSettings={vi.fn()}
        onMonitoringChange={vi.fn()}
        onInferenceUpdate={vi.fn()}
        onRecognitionError={vi.fn()}
      />,
    );
    expect(handlers.inference).toBeNull();
    expect(handlers.error).toBeNull();
  });
});
