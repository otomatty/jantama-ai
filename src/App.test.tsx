import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import App from "./App";

describe("App", () => {
  it("ヘッダーが描画される (停止中状態)", () => {
    render(<App />);
    // 初期表示は uninitialized -> 「設定画面で初期設定を行ってください」
    // または idle -> 「監視を開始してください」
    expect(screen.getByText(/対象:/)).toBeInTheDocument();
  });
});
