import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { MainScreen } from "./MainScreen";
import { INITIAL_APP_STATE } from "@/state/appState";
import { SCENARIO_FIXTURES } from "@/lib/scenarios";

describe("MainScreen", () => {
  it("phase が idle のとき待機メッセージを表示", () => {
    render(
      <MainScreen
        state={{ ...INITIAL_APP_STATE, phase: "idle" }}
        onOpenSettings={vi.fn()}
        onMonitoringChange={vi.fn()}
        onInferenceUpdate={vi.fn()}
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
      />,
    );
    expect(screen.getByText("6m を切る")).toBeInTheDocument();
    expect(screen.getByText("+0.32")).toBeInTheDocument();
    expect(screen.getByText("RANK 1")).toBeInTheDocument();
  });
});
