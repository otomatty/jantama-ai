import { useEffect, useState } from "react";
import { ContextBar } from "@/components/ContextBar";
import { DangerSafeBlock } from "@/components/DangerSafeBlock";
import { ErrorBody } from "@/components/ErrorBody";
import { HandRow } from "@/components/HandRow";
import { HeroLayout } from "@/components/HeroLayout";
import { IdleBody } from "@/components/IdleBody";
import { MonitorButton } from "@/components/MonitorButton";
import { ReasonBlock } from "@/components/ReasonBlock";
import { StatusBar } from "@/components/StatusBar";
import {
  runStubInference,
  startMonitoring,
  stopMonitoring,
} from "@/lib/tauriCommands";
import type { AppState } from "@/state/appState";
import type { GameBoardSummary, InferenceResult } from "@/types";

interface MainScreenProps {
  state: AppState;
  onOpenSettings: () => void;
  onMonitoringChange: (watching: boolean) => void;
  onInferenceUpdate: (
    inference: InferenceResult,
    board: GameBoardSummary | null,
  ) => void;
}

export function MainScreen({
  state,
  onOpenSettings,
  onMonitoringChange,
  onInferenceUpdate,
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

  // ブラウザ動作確認用: 監視 ON 中に 5秒に1回スタブ推論を流す
  useEffect(() => {
    if (!state.monitoring.watching) return;
    let cancelled = false;
    const fire = async () => {
      const { inference, board } = await runStubInference();
      if (!cancelled) onInferenceUpdate(inference, board);
    };
    fire();
    const id = setInterval(fire, 5000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [state.monitoring.watching, onInferenceUpdate]);

  return (
    <div
      className="mx-auto flex h-full w-full max-w-[480px] flex-col overflow-hidden border border-ink-200 bg-ink-50 font-jp"
      style={{
        boxShadow:
          "0 24px 60px rgba(15,15,30,0.14), 0 4px 12px rgba(15,15,30,0.06)",
      }}
    >
      <StatusBar
        monitoring={state.monitoring.watching}
        onOpenSettings={onOpenSettings}
      />
      <ContextBar
        board={state.monitoring.watching && state.inference ? state.board : null}
      />

      <main className="flex flex-1 flex-col overflow-y-auto">
        <MainBody
          state={state}
          onOpenSettings={onOpenSettings}
        />
      </main>

      {state.monitoring.watching && state.inference && state.board && (
        <HandRow board={state.board} inference={state.inference} />
      )}

      <div className="border-t border-ink-200 bg-white px-3 py-3">
        <MonitorButton
          on={state.monitoring.watching}
          disabled={busy || state.phase === "uninitialized" || state.phase === "error"}
          onClick={handleToggleWatching}
        />
      </div>
    </div>
  );
}

function MainBody({
  state,
  onOpenSettings,
}: {
  state: AppState;
  onOpenSettings: () => void;
}) {
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

  if (!state.monitoring.watching || !state.inference) {
    return <IdleBody />;
  }

  return (
    <div className="flex flex-col gap-3 p-3.5">
      <HeroLayout inference={state.inference} />
      {state.settings.show_llm_reason && (
        <ReasonBlock reason={state.inference.reason} />
      )}
      {state.settings.show_danger_safe && (
        <DangerSafeBlock
          danger={state.inference.danger}
          safe={state.inference.safe}
        />
      )}
    </div>
  );
}
