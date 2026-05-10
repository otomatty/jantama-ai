import { describe, expect, it } from "vitest";
import { REGION_DEFS, countRoi, getRegionRect, hasAnyRoi, setRegionRect } from "./roiCalibration";
import { EMPTY_ROI_CALIBRATION, type RoiCalibration } from "@/types";

const SAMPLE = { x: 0.1, y: 0.2, w: 0.3, h: 0.4 };

describe("roiCalibration helpers", () => {
  it("REGION_DEFS は issue #10 / #12 の 10 領域を網羅する", () => {
    const ids = REGION_DEFS.map((r) => r.id).sort();
    expect(ids).toEqual(
      [
        "doras",
        "hand",
        "river_across",
        "river_left",
        "river_right",
        "river_self",
        "round_info",
        "scores",
        "self_wind",
        "turn_counter",
      ].sort(),
    );
  });

  it("getRegionRect/setRegionRect は flat な ID で全領域を読み書きできる", () => {
    let calibration: RoiCalibration = EMPTY_ROI_CALIBRATION;
    for (const def of REGION_DEFS) {
      expect(getRegionRect(calibration, def.id)).toBeNull();
      calibration = setRegionRect(calibration, def.id, SAMPLE);
      expect(getRegionRect(calibration, def.id)).toEqual(SAMPLE);
    }
    expect(countRoi(calibration)).toBe(REGION_DEFS.length);
    expect(hasAnyRoi(calibration)).toBe(true);
  });

  it("setRegionRect は他領域に副作用を起こさない (immutable update)", () => {
    const before: RoiCalibration = EMPTY_ROI_CALIBRATION;
    const afterHand = setRegionRect(before, "hand", SAMPLE);
    // 元 (EMPTY) は不変
    expect(before.hand).toBeNull();
    expect(before.rivers.self).toBeNull();
    // 別領域は null のまま
    expect(afterHand.doras).toBeNull();
    expect(afterHand.rivers.self).toBeNull();
    // 1 領域だけ反映
    expect(afterHand.hand).toEqual(SAMPLE);

    const afterRiver = setRegionRect(afterHand, "river_self", SAMPLE);
    expect(afterRiver.hand).toEqual(SAMPLE);
    expect(afterRiver.rivers.self).toEqual(SAMPLE);
    expect(afterRiver.rivers.right).toBeNull();
  });

  it("setRegionRect に null を渡すとクリアできる", () => {
    const filled = setRegionRect(EMPTY_ROI_CALIBRATION, "doras", SAMPLE);
    expect(getRegionRect(filled, "doras")).toEqual(SAMPLE);
    const cleared = setRegionRect(filled, "doras", null);
    expect(getRegionRect(cleared, "doras")).toBeNull();
    expect(hasAnyRoi(cleared)).toBe(false);
  });

  it("countRoi/hasAnyRoi は null calibration を 0 / false で返す", () => {
    expect(countRoi(null)).toBe(0);
    expect(countRoi(undefined)).toBe(0);
    expect(hasAnyRoi(null)).toBe(false);
    expect(hasAnyRoi(undefined)).toBe(false);
    expect(hasAnyRoi(EMPTY_ROI_CALIBRATION)).toBe(false);
  });
});
