import { useCallback, useEffect, useState } from "react";
import { MainScreen } from "@/screens/MainScreen";
import { SettingsScreen } from "@/screens/SettingsScreen";
import { CalibrationScreen } from "@/screens/CalibrationScreen";
import { useAppState } from "@/state/appState";
import { loadSettings } from "@/lib/tauriCommands";
import type { AppError, AppPhase, AppSettings, GameBoardSummary, InferenceResult } from "@/types";

type Screen = "main" | "settings" | "calibration";

/**
 * 設定の必須項目 (キャプチャ対象 + Mortal モデルパス) が揃っているかで
 * 静的フェーズを決める。監視中・推論中・エラー等は別経路で上書きされる前提。
 *
 * 設定がアプリ state に昇格 (= App の `state.settings` 更新) されるたびに
 * これを呼んで phase を再計算する。`onOpenCalibration` で settings だけ更新して
 * phase を放置すると、必須項目の追加/削除を伴う編集後にメイン画面が古い phase
 * (`uninitialized` / `idle`) のまま貼り付き、監視ボタンの活性が UI と乖離する
 * (Codex P1 on PR #42)。
 */
function phaseForSettings(settings: AppSettings): AppPhase {
  return settings.capture_target_window_id && settings.mortal_model_path ? "idle" : "uninitialized";
}

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
          setPhase(phaseForSettings(loaded));
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
    (inference: InferenceResult | null, board: GameBoardSummary | null, timestamp: string) => {
      setInference(inference);
      setBoard(board);
      // issue #15: 手番でないフレームでも timestamp は届くので、
      // last_recognized_at は payload の timestamp を使う (= heartbeat)。
      setMonitoring({ last_recognized_at: timestamp });
      // 直前のサイクルが recognition-error で phase=error に固定されていても、
      // 次に推奨またはアイドルフレームが届いたら自動復帰させる。
      // 一過性のキャプチャ失敗で UI が永続的にエラー画面に張り付くのを防ぐ。
      setError(null);
      // issue #15: inference が null (手番でない) のときは watching_no_board に倒す。
      // inference 付きでも board.my_turn が false のケースは Rust 側でスキップ
      // するので発生しない想定 (フェイルセーフは MainScreen 側の MainBody ゲート)。
      setPhase(inference ? "watching_recommend" : "watching_no_board");
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
        onOpenCalibration={(current) => {
          // 設定画面が永続化した最新値を親 state に反映してから遷移する。
          // これがないと CalibrationScreen が古い settings を見て onSaved で
          // クロバーする (Codex P1 on PR #42)。
          // phase も再計算する: 必須項目の追加/削除を伴う編集後に
          // calibration → main で古い phase に張り付くのを防ぐ (Codex P1 続報)。
          setSettings(current);
          setPhase(phaseForSettings(current));
          setScreen("calibration");
        }}
        onSaved={(next) => {
          setSettings(next);
          setPhase(phaseForSettings(next));
          setScreen("main");
        }}
      />
    );
  }

  if (screen === "calibration") {
    return (
      <CalibrationScreen
        settings={state.settings}
        onBack={() => setScreen("settings")}
        onSaved={(next: AppSettings) => {
          // 永続化は CalibrationScreen 側で完了済みの前提でここに来る (失敗時は
          // 画面側がエラーを表示してこの onSaved を呼ばないので、UI / store 不整合
          // が生まれない。Codex P1 / CodeRabbit Major on PR #42)。
          setSettings(next);
          // ROI 編集だけでは phase は変わらない (capture_target / mortal_model は
          // 維持される) が、設定昇格は常に phase 再計算とセットにする方針なので
          // 同じヘルパを通して整合性を保つ。
          setPhase(phaseForSettings(next));
          setScreen("settings");
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
