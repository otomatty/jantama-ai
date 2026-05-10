import { useCallback, useEffect, useState } from "react";
import { MainScreen } from "@/screens/MainScreen";
import { SettingsScreen } from "@/screens/SettingsScreen";
import { useAppState } from "@/state/appState";
import { loadSettings } from "@/lib/tauriCommands";
import type { AppError, GameBoardSummary, InferenceResult } from "@/types";

type Screen = "main" | "settings";

function App() {
  const { state, setPhase, setSettings, setMonitoring, setInference, setBoard, setError } =
    useAppState();
  const [screen, setScreen] = useState<Screen>("main");

  // 起動時に保存済み設定を読み込む
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const loaded = await loadSettings();
        if (cancelled) return;
        if (loaded) {
          setSettings(loaded);
          setPhase(
            loaded.capture_target_window_id && loaded.mortal_model_path ? "idle" : "uninitialized",
          );
        } else {
          setPhase("uninitialized");
        }
      } catch {
        // 設定ロード失敗時は未設定扱いにフォールバックし、画面遷移は決定論的に保つ
        if (!cancelled) setPhase("uninitialized");
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [setSettings, setPhase]);

  const handleOpenSettings = useCallback(() => setScreen("settings"), []);

  const handleMonitoringChange = useCallback(
    (watching: boolean) => {
      // 監視開始直後は推論がまだ走っていないので last_recognized_at は null。
      // 実際の時刻は onInferenceUpdate で推論結果が届いた時に更新する。
      setMonitoring({
        watching,
        capture_target_window_title: state.settings.capture_target_window_title,
        last_recognized_at: null,
      });
      // 監視 ON/OFF どちらの操作も「ユーザーがエラーを了承して仕切り直し」と
      // みなし、phase=error に張り付かないようリセットする。これがないと
      // 永続的な recognition-error (例: 対象ウィンドウ最小化) で次の成功推論
      // が来ない限り MonitorButton 経由で停止できなくなる。
      setError(null);
      if (watching) {
        setPhase("watching_no_board");
      } else {
        setInference(null);
        setBoard(null);
        setPhase("idle");
      }
    },
    [
      setMonitoring,
      setInference,
      setBoard,
      setError,
      setPhase,
      state.settings.capture_target_window_title,
    ],
  );

  const handleInferenceUpdate = useCallback(
    (inference: InferenceResult, board: GameBoardSummary | null) => {
      setInference(inference);
      setBoard(board);
      setMonitoring({ last_recognized_at: inference.timestamp });
      // 直前のサイクルが recognition-error で phase=error に固定されていても、
      // 次に成功推論が届いたら自動復帰させる。
      // 一過性のキャプチャ失敗で UI が永続的にエラー画面に張り付くのを防ぐ。
      setError(null);
      setPhase(board ? "watching_recommend" : "watching_no_board");
    },
    [setInference, setBoard, setMonitoring, setError, setPhase],
  );

  const handleRecognitionError = useCallback(
    (error: AppError) => {
      setError(error);
      setPhase("error");
    },
    [setError, setPhase],
  );

  if (screen === "settings") {
    return (
      <SettingsScreen
        initialSettings={state.settings}
        onBack={() => setScreen("main")}
        onSaved={(next) => {
          setSettings(next);
          setPhase(
            next.capture_target_window_id && next.mortal_model_path ? "idle" : "uninitialized",
          );
          setScreen("main");
        }}
      />
    );
  }

  return (
    <MainScreen
      state={state}
      onOpenSettings={handleOpenSettings}
      onMonitoringChange={handleMonitoringChange}
      onInferenceUpdate={handleInferenceUpdate}
      onRecognitionError={handleRecognitionError}
    />
  );
}

export default App;
