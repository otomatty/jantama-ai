import { useEffect, useState } from "react";
import { MainScreen } from "@/screens/MainScreen";
import { SettingsScreen } from "@/screens/SettingsScreen";
import { useAppState } from "@/state/appState";
import { loadSettings } from "@/lib/tauriCommands";

type Screen = "main" | "settings";

function App() {
  const { state, setPhase, setSettings, setMonitoring, setInference } =
    useAppState();
  const [screen, setScreen] = useState<Screen>("main");

  // 起動時に保存済み設定を読み込む
  useEffect(() => {
    (async () => {
      const loaded = await loadSettings();
      if (loaded) {
        setSettings(loaded);
        setPhase(
          loaded.capture_target_window_id && loaded.mortal_model_path
            ? "idle"
            : "uninitialized",
        );
      } else {
        setPhase("uninitialized");
      }
    })();
  }, [setSettings, setPhase]);

  if (screen === "settings") {
    return (
      <SettingsScreen
        initialSettings={state.settings}
        onBack={() => setScreen("main")}
        onSaved={(next) => {
          setSettings(next);
          setPhase(
            next.capture_target_window_id && next.mortal_model_path
              ? "idle"
              : "uninitialized",
          );
          setScreen("main");
        }}
      />
    );
  }

  return (
    <MainScreen
      state={state}
      onOpenSettings={() => setScreen("settings")}
      onMonitoringChange={(watching) => {
        setMonitoring({
          watching,
          capture_target_window_title: state.settings.capture_target_window_title,
          last_recognized_at: watching ? new Date().toISOString() : null,
        });
        if (!watching) setInference(null);
      }}
      onInferenceUpdate={(result) => {
        setInference(result);
        setMonitoring({ last_recognized_at: result.timestamp });
      }}
    />
  );
}

export default App;
