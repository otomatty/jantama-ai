import { describe, expect, it } from "vitest";
import { isTileCode } from "./Tile";

describe("isTileCode", () => {
  it("通常牌 1m..9m / 1p..9p / 1s..9s を許可", () => {
    for (const suit of ["m", "p", "s"]) {
      for (let n = 1; n <= 9; n++) {
        expect(isTileCode(`${n}${suit}`)).toBe(true);
      }
    }
  });

  it("字牌は 1z..7z のみ許可", () => {
    for (let n = 1; n <= 7; n++) {
      expect(isTileCode(`${n}z`)).toBe(true);
    }
    expect(isTileCode("8z")).toBe(false);
    expect(isTileCode("9z")).toBe(false);
  });

  it("不正値・アクション識別子は弾く", () => {
    expect(isTileCode("0m")).toBe(false);
    expect(isTileCode("riichi")).toBe(false);
    expect(isTileCode("ron")).toBe(false);
    expect(isTileCode("")).toBe(false);
  });
});
