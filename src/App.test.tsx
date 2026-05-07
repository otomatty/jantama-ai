import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import App from "./App";

describe("App", () => {
  it("初期表示で雀tama AI ロゴと OFF (停止中) インジケータが見える", () => {
    render(<App />);
    expect(screen.getByText("雀")).toBeInTheDocument();
    expect(screen.getByText("tama")).toBeInTheDocument();
    // 監視中バッジ — 起動直後は停止中なので OFF
    expect(screen.getByText("OFF")).toBeInTheDocument();
  });
});
