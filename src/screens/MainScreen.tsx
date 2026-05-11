import { useState } from "react";
import { ContextBar } from "@/components/ContextBar";
import { DangerSafeBlock } from "@/components/DangerSafeBlock";
import { ErrorBody } from "@/components/ErrorBody";
import { HandRow } from "@/components/HandRow";
import { HeroLayout } from "@/components/HeroLayout";
import { IdleBody } from "@/components/IdleBody";
import { MonitorButton } from "@/components/MonitorButton";
import { ReasonBlock } from "@/components/ReasonBlock";
import { StatusBar } from "@/components/StatusBar";
import { useInferenceEvents } from "@/hooks/useInferenceEvents";
import { startMonitoring, stopMonitoring } from "@/lib/tauriCommands";
import type { AppState } from "@/state/appState";
import type { AppError, GameBoardSummary, InferenceResult } from "@/types";

interface MainScreenProps {
  state: AppState;
  onOpenSettings: () => void;
  onMonitoringChange: (watching: boolean) => void;
  onInferenceUpdate: (
    inference: InferenceResult | null,
    board: GameBoardSummary | null,
    timestamp: string,
  ) => void;
  onRecognitionError: (error: AppError) => void;
}

export function MainScreen({
  state,
  onOpenSettings,
  onMonitoringChange,
  onInferenceUpdate,
  onRecognitionError,
}: MainScreenProps) {
  const [busy, setBusy] = useState(false);

  const handleToggleWatching = async () => {
    setBusy(true);
    try {
      if (state.monitoring.watching) {
        await stopMonitoring();
        onMonitoringChange(false);
      } else {
        await startMonitoring();
        onMonitoringChange(true);
      }
    } finally {
      setBusy(false);
    }
  };

  useInferenceEvents({
    watching: state.monitoring.watching,
    onInference: onInferenceUpdate,
    onError: onRecognitionError,
  });

  return (
    <div
      className="mx-auto flex h-full w-full max-w-[480px] flex-col overflow-hidden border border-ink-200 bg-ink-50 font-jp"
      style={{
        boxShadow: "0 24px 60px rgba(15,15,30,0.14), 0 4px 12px rgba(15,15,30,0.06)",
      }}
    >
      <StatusBar monitoring={state.monitoring.watching} onOpenSettings={onOpenSettings} />
      {/* issue #15: ContextBar は手番外でも盤面サマリ (局・巡目・点棒) を出し続ける。
          mortal がスキップされている opponent turn でも recognition は走るので
          state.board は最新値が入っている。 */}
      <ContextBar board={state.monitoring.watching ? state.board : null} />

      <main className="flex flex-1 flex-col overflow-y-auto">
        <MainBody state={state} onOpenSettings={onOpenSettings} />
      </main>

      {state.monitoring.watching && state.inference && state.board && isMyTurn(state.board) && (
        <HandRow board={state.board} inference={state.inference} />
      )}

      <div className="border-t border-ink-200 bg-white px-3 py-3">
        <MonitorButton
          on={state.monitoring.watching}
          disabled={busy || state.phase === "uninitialized"}
          onClick={handleToggleWatching}
        />
      </div>
    </div>
  );
}

function MainBody({ state, onOpenSettings }: { state: AppState; onOpenSettings: () => void }) {
  if (state.phase === "uninitialized") {
    return (
      <ErrorBody
        error={{
          type: "config",
          message: "設定画面で雀魂のウィンドウと Mortal モデルを選んでください。",
          occurred_at: new Date().toISOString(),
        }}
        onOpenSettings={onOpenSettings}
      />
    );
  }

  if (state.phase === "error" && state.error) {
    return <ErrorBody error={state.error} onOpenSettings={onOpenSettings} />;
  }

  // issue #15: `inference` が無い (手番でない / mortal スキップ) または
  // board の my_turn が立っていない場合は IdleBody を表示する。Rust 側で
  // mortal をスキップしているので、ここに来る `state.inference` 非 null は
  // 「手番である」と Rust が判定したフレームのみだが、フェイルセーフとして
  // フロント側でも `isMyTurn` でゲートする。
  if (!state.monitoring.watching || !state.inference || !isMyTurn(state.board)) {
    return <IdleBody />;
  }

  return (
    <div className="flex flex-col gap-3 p-3.5">
      <HeroLayout inference={state.inference} />
      {state.settings.show_llm_reason && <ReasonBlock reason={state.inference.reason} />}
      {state.settings.show_danger_safe && (
        <DangerSafeBlock danger={state.inference.danger} safe={state.inference.safe} />
      )}
    </div>
  );
}

/**
 * 盤面サマリから「今 UI が推奨表示を出す状態か」を判定する (issue #15)。
 *
 * `my_turn` / `available_actions` が `null` / `undefined` のとき (旧 Rust
 * ペイロードや recognition が新スキーマ未対応の過渡期) は、Rust 側の
 * `should_skip_inference` が「mortal を呼ぶ」側に倒しているのと整合させ、
 * ここでも `true` 扱いで通す (= 推奨表示を抑制しない)。
 * 明示的に `false` または空配列のときだけ IdleBody に倒す。
 */
function isMyTurn(board: GameBoardSummary | null): boolean {
  if (!board) return false;
  if (board.my_turn === false) return false;
  const actions = board.available_actions;
  // 明示的に空配列のときだけ抑制。`null` / `undefined` は通す。
  if (Array.isArray(actions) && actions.length === 0) return false;
  return true;
}
