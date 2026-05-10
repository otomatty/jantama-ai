import { describe, expect, it, vi, beforeEach } from "vitest";
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { CalibrationScreen } from "./CalibrationScreen";
import { DEFAULT_SETTINGS } from "@/types";
import * as tauriCommands from "@/lib/tauriCommands";

vi.mock("@/lib/tauriCommands", async () => {
  const actual = await vi.importActual<typeof import("@/lib/tauriCommands")>("@/lib/tauriCommands");
  return {
    ...actual,
    captureWindowForCalibration: vi.fn(actual.captureWindowForCalibration),
  };
});

beforeEach(() => {
  vi.mocked(tauriCommands.captureWindowForCalibration).mockResolvedValue({
    width: 1920,
    height: 1080,
    image_b64:
      "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNgYAAAAAMAAWgmWQ0AAAAASUVORK5CYII=",
  });
});

// jsdom は getBoundingClientRect に 0 を返すため、ドラッグ計算用に固定値を返す。
function stubCanvasRect() {
  const orig = HTMLDivElement.prototype.getBoundingClientRect;
  HTMLDivElement.prototype.getBoundingClientRect = function () {
    return {
      x: 0,
      y: 0,
      left: 0,
      top: 0,
      right: 200,
      bottom: 100,
      width: 200,
      height: 100,
      toJSON() {
        return {};
      },
    } as DOMRect;
  };
  return () => {
    HTMLDivElement.prototype.getBoundingClientRect = orig;
  };
}

describe("CalibrationScreen", () => {
  it("起動時に capture を取りに行き、未設定時はプレースホルダから始まる", async () => {
    render(<CalibrationScreen settings={DEFAULT_SETTINGS} onBack={() => {}} onSaved={() => {}} />);
    await waitFor(() => {
      expect(tauriCommands.captureWindowForCalibration).toHaveBeenCalled();
    });
    // 初期表示: 1 領域目 = 手牌が選択中 (タブのボタンに「手牌」)
    expect(screen.getByRole("button", { name: /手牌/ })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "再キャプチャ" })).toBeInTheDocument();
  });

  it("ドラッグで矩形を確定すると比率が onSaved に渡る", async () => {
    const onSaved = vi.fn();
    render(<CalibrationScreen settings={DEFAULT_SETTINGS} onBack={() => {}} onSaved={onSaved} />);
    await waitFor(() => {
      expect(tauriCommands.captureWindowForCalibration).toHaveBeenCalled();
    });
    // pointer capture とラジアル計算用に rect をスタブする
    const restore = stubCanvasRect();
    try {
      const canvas = screen.getByTestId("roi-canvas");
      // 200x100 の中で (40,20)〜(120,60) → ratio (0.2, 0.2, 0.4, 0.4)
      // 各 fireEvent は内部で act() ラップされるが、念のため個別の act で
      // 区切って state 更新を確実に flush させる (drag closure の取り違え防止)。
      act(() => {
        fireEvent.mouseDown(canvas, { clientX: 40, clientY: 20 });
      });
      act(() => {
        fireEvent.mouseMove(canvas, { clientX: 120, clientY: 60 });
      });
      act(() => {
        fireEvent.mouseUp(canvas, { clientX: 120, clientY: 60 });
      });

      // 確定後に「保存」を押すと onSaved に hand が乗った settings が来る
      fireEvent.click(screen.getByRole("button", { name: "保存" }));
      expect(onSaved).toHaveBeenCalledTimes(1);
      const next = onSaved.mock.calls[0][0];
      expect(next.roi_calibration.hand).toEqual({ x: 0.2, y: 0.2, w: 0.4, h: 0.4 });
    } finally {
      restore();
    }
  });

  it("極小ドラッグ (誤クリック) は矩形として確定しない", async () => {
    const onSaved = vi.fn();
    render(<CalibrationScreen settings={DEFAULT_SETTINGS} onBack={() => {}} onSaved={onSaved} />);
    await waitFor(() => {
      expect(tauriCommands.captureWindowForCalibration).toHaveBeenCalled();
    });
    const restore = stubCanvasRect();
    try {
      const canvas = screen.getByTestId("roi-canvas");
      act(() => {
        fireEvent.mouseDown(canvas, { clientX: 50, clientY: 50 });
      });
      act(() => {
        fireEvent.mouseUp(canvas, { clientX: 50, clientY: 50 });
      });
      fireEvent.click(screen.getByRole("button", { name: "保存" }));
      expect(onSaved).toHaveBeenCalledTimes(1);
      const next = onSaved.mock.calls[0][0];
      // 確定されていないので EMPTY のまま
      expect(next.roi_calibration.hand).toBeNull();
    } finally {
      restore();
    }
  });
});
