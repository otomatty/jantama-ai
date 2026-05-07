import { useEffect, useState } from "react";
import { Settings, Play, Pause, AlertCircle, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import type { AppState } from "@/state/appState";
import {
  startMonitoring,
  stopMonitoring,
  runStubInference,
} from "@/lib/tauriCommands";
import type { ActionType } from "@/types";

interface MainScreenProps {
  state: AppState;
  onOpenSettings: () => void;
  onMonitoringChange: (watching: boolean) => void;
  onInferenceUpdate: (result: Awaited<ReturnType<typeof runStubInference>>) => void;
}

const ACTION_LABEL: Record<ActionType, string> = {
  discard: "打牌",
  riichi: "リーチ",
  chi: "チー",
  pon: "ポン",
  kan: "カン",
  ron: "ロン",
  tsumo: "ツモ",
  pass: "パス",
};

export function MainScreen({
  state,
  onOpenSettings,
  onMonitoringChange,
  onInferenceUpdate,
}: MainScreenProps) {
  const [busy, setBusy] = useState(false);

  // 監視 ON/OFF ハンドラ
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
    const id = setInterval(async () => {
      const result = await runStubInference();
      onInferenceUpdate(result);
    }, 5000);
    // 初回も即実行
    runStubInference().then(onInferenceUpdate);
    return () => clearInterval(id);
  }, [state.monitoring.watching, onInferenceUpdate]);

  return (
    <div className="flex h-full flex-col">
      {/* ============== 上部ステータスバー ============== */}
      <header className="flex items-center justify-between border-b border-[var(--color-border)] px-4 py-3">
        <div className="flex items-center gap-3">
          {state.monitoring.watching ? (
            <Badge variant="success">監視中</Badge>
          ) : (
            <Badge variant="secondary">停止中</Badge>
          )}
          <span className="text-sm text-[var(--color-muted-foreground)]">
            対象: {state.monitoring.capture_target_window_title ?? "未設定"}
          </span>
          {state.monitoring.last_recognized_at && (
            <span className="text-xs text-[var(--color-muted-foreground)]">
              最終認識: {formatTime(state.monitoring.last_recognized_at)}
            </span>
          )}
        </div>
        <Button variant="ghost" size="sm" onClick={onOpenSettings}>
          <Settings />
          設定
        </Button>
      </header>

      {/* ============== 中央メインエリア ============== */}
      <main className="flex-1 overflow-auto p-6">
        <CenterContent state={state} />
      </main>

      {/* ============== 下部ボードサマリー ============== */}
      {state.board && (
        <section className="border-t border-[var(--color-border)] px-4 py-3 text-xs text-[var(--color-muted-foreground)]">
          <div>手牌: {state.board.hand.join(" ")}</div>
          <div className="mt-1">
            自風: {state.board.self_wind} / 場風: {state.board.round_wind} / 巡目:{" "}
            {state.board.turn} / ドラ: {state.board.dora_indicators.join(" ")}
          </div>
        </section>
      )}

      {/* ============== 監視 ON/OFF ============== */}
      <footer className="flex items-center justify-center border-t border-[var(--color-border)] px-4 py-3">
        <Button
          size="lg"
          variant={state.monitoring.watching ? "destructive" : "default"}
          disabled={busy || state.phase === "uninitialized"}
          onClick={handleToggleWatching}
        >
          {state.monitoring.watching ? (
            <>
              <Pause />
              監視 OFF
            </>
          ) : (
            <>
              <Play />
              監視 ON
            </>
          )}
        </Button>
      </footer>
    </div>
  );

  function CenterContent({ state }: { state: AppState }) {
    // PRD §5.2 の状態別表示テーブルに対応
    if (state.phase === "uninitialized") {
      return (
        <Card className="mx-auto max-w-xl">
          <CardContent className="flex flex-col items-center gap-4 py-10">
            <p className="text-center text-[var(--color-muted-foreground)]">
              設定画面で初期設定を行ってください
            </p>
            <Button onClick={onOpenSettings}>設定画面へ</Button>
          </CardContent>
        </Card>
      );
    }

    if (state.phase === "error" && state.error) {
      return (
        <Card className="mx-auto max-w-xl">
          <CardContent className="flex flex-col items-center gap-4 py-10">
            <AlertCircle className="size-10 text-[var(--color-destructive)]" />
            <p className="text-center font-medium">{state.error.message}</p>
            <p className="text-xs text-[var(--color-muted-foreground)]">
              発生時刻: {formatTime(state.error.occurred_at)} / 種別: {state.error.type}
            </p>
            <Button variant="outline" onClick={onOpenSettings}>
              設定を確認
            </Button>
          </CardContent>
        </Card>
      );
    }

    if (!state.monitoring.watching) {
      return (
        <Card className="mx-auto max-w-xl">
          <CardContent className="flex flex-col items-center gap-4 py-10">
            <p className="text-center text-[var(--color-muted-foreground)]">
              監視を開始してください
            </p>
          </CardContent>
        </Card>
      );
    }

    if (state.monitoring.watching && !state.inference) {
      return (
        <Card className="mx-auto max-w-xl">
          <CardContent className="flex flex-col items-center gap-4 py-10">
            <Loader2 className="size-8 animate-spin text-[var(--color-muted-foreground)]" />
            <p className="text-center text-[var(--color-muted-foreground)]">
              対局を待機中...
            </p>
          </CardContent>
        </Card>
      );
    }

    // 監視中・推奨表示中
    if (state.inference) {
      const { recommended, candidates } = state.inference;
      return (
        <div className="mx-auto flex max-w-2xl flex-col gap-6">
          <div className="text-center">
            <div className="mb-2 text-sm text-[var(--color-muted-foreground)]">
              推奨アクション
            </div>
            <div className="text-5xl font-bold tracking-wide">
              {recommended.tile ?? ACTION_LABEL[recommended.action_type]}
              <span className="ml-3 text-2xl font-normal text-[var(--color-muted-foreground)]">
                ({ACTION_LABEL[recommended.action_type]})
              </span>
            </div>
            <div className="mt-2 text-sm text-[var(--color-muted-foreground)]">
              期待値: {formatExpectedValue(recommended.expected_value)}
            </div>
          </div>

          <Card>
            <CardContent className="py-4">
              <div className="mb-2 text-xs uppercase text-[var(--color-muted-foreground)]">
                候補
              </div>
              <ul className="space-y-2">
                {candidates.map((c, i) => (
                  <li key={`${c.tile ?? c.action_type}-${i}`} className="flex items-center justify-between">
                    <span className="font-mono text-base">
                      {i + 1}. {c.tile ?? ACTION_LABEL[c.action_type]}
                      <span className="ml-2 text-xs text-[var(--color-muted-foreground)]">
                        [{ACTION_LABEL[c.action_type]}]
                      </span>
                    </span>
                    <span className="font-mono text-sm">
                      期待値: {formatExpectedValue(c.expected_value)}
                    </span>
                  </li>
                ))}
              </ul>
            </CardContent>
          </Card>
        </div>
      );
    }

    return null;
  }
}

function formatTime(iso: string): string {
  const d = new Date(iso);
  return d.toLocaleTimeString("ja-JP", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function formatExpectedValue(v: number): string {
  const sign = v >= 0 ? "+" : "";
  return `${sign}${v.toFixed(2)}`;
}
