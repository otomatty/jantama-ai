import { useReducer, useCallback } from "react";
import {
  type AppPhase,
  type AppSettings,
  type AppError,
  type InferenceResult,
  type GameBoardSummary,
  type MonitoringStatus,
  DEFAULT_SETTINGS,
} from "@/types";

// ============================================================
// アプリ全体の状態
// ============================================================

export interface AppState {
  phase: AppPhase;
  settings: AppSettings;
  monitoring: MonitoringStatus;
  inference: InferenceResult | null;
  board: GameBoardSummary | null;
  error: AppError | null;
}

export const INITIAL_APP_STATE: AppState = {
  phase: "uninitialized",
  settings: DEFAULT_SETTINGS,
  monitoring: {
    watching: false,
    capture_target_window_title: null,
    last_recognized_at: null,
  },
  inference: null,
  board: null,
  error: null,
};

// ============================================================
// Action
// ============================================================

export type AppAction =
  | { type: "SET_PHASE"; phase: AppPhase }
  | { type: "SET_SETTINGS"; settings: AppSettings }
  | { type: "SET_MONITORING"; monitoring: Partial<MonitoringStatus> }
  | { type: "SET_INFERENCE"; inference: InferenceResult | null }
  | { type: "SET_BOARD"; board: GameBoardSummary | null }
  | { type: "SET_ERROR"; error: AppError | null }
  | { type: "RESET" };

export function appReducer(state: AppState, action: AppAction): AppState {
  switch (action.type) {
    case "SET_PHASE":
      return { ...state, phase: action.phase };
    case "SET_SETTINGS":
      return { ...state, settings: action.settings };
    case "SET_MONITORING":
      return {
        ...state,
        monitoring: { ...state.monitoring, ...action.monitoring },
      };
    case "SET_INFERENCE":
      return { ...state, inference: action.inference };
    case "SET_BOARD":
      return { ...state, board: action.board };
    case "SET_ERROR":
      return { ...state, error: action.error };
    case "RESET":
      return INITIAL_APP_STATE;
    default:
      return state;
  }
}

// ============================================================
// Hook
// ============================================================

export function useAppState() {
  const [state, dispatch] = useReducer(appReducer, INITIAL_APP_STATE);

  const setPhase = useCallback((phase: AppPhase) => dispatch({ type: "SET_PHASE", phase }), []);
  const setSettings = useCallback(
    (settings: AppSettings) => dispatch({ type: "SET_SETTINGS", settings }),
    [],
  );
  const setMonitoring = useCallback(
    (monitoring: Partial<MonitoringStatus>) => dispatch({ type: "SET_MONITORING", monitoring }),
    [],
  );
  const setInference = useCallback(
    (inference: InferenceResult | null) => dispatch({ type: "SET_INFERENCE", inference }),
    [],
  );
  const setBoard = useCallback(
    (board: GameBoardSummary | null) => dispatch({ type: "SET_BOARD", board }),
    [],
  );
  const setError = useCallback(
    (error: AppError | null) => dispatch({ type: "SET_ERROR", error }),
    [],
  );

  return {
    state,
    setPhase,
    setSettings,
    setMonitoring,
    setInference,
    setBoard,
    setError,
  };
}
