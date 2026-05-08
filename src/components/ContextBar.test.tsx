import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { ContextBar } from "./ContextBar";
import type { GameBoardSummary } from "@/types";

describe("ContextBar", () => {
  it("board が null のときは数値を捏造せずプレースホルダーを表示", () => {
    render(<ContextBar board={null} />);
    // — が複数 (round / turn / 自風 / 持点 / Dora) で出る
    const dashes = screen.getAllByText("—");
    expect(dashes.length).toBeGreaterThanOrEqual(4);
    // 旧バグ回帰防止: 偽の値が出ないこと
    expect(screen.queryByText("東1局")).not.toBeInTheDocument();
    expect(screen.queryByText(/25,000/)).not.toBeInTheDocument();
  });

  it("board が与えられたら局・巡目・自風・持点を表示", () => {
    const board: GameBoardSummary = {
      hand: [],
      self_wind: "南",
      round_wind: "東",
      turn: 4,
      dora_indicators: [],
      score: 32100,
      round_label: "東2局",
    };
    render(<ContextBar board={board} />);
    expect(screen.getByText("東2局")).toBeInTheDocument();
    expect(screen.getByText(/4巡目/)).toBeInTheDocument();
    expect(screen.getByText("南")).toBeInTheDocument();
    expect(screen.getByText("32,100")).toBeInTheDocument();
  });
});
