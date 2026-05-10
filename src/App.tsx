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
      if (!watching) {
        setInference(null);
        setBoard(null);
      }
    },
    [setMonitoring, setInference, setBoard, state.settings.capture_target_window_title],
  );

  const handleInferenceUpdate = useCallback(
    (inference: InferenceResult, board: GameBoardSummary | null) => {
      setInference(inference);
      setBoard(board);
      setMonitoring({ last_recognized_at: inference.timestamp });
    },
    [setInference, setBoard, setMonitoring],
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
