/**
 * ROI キャリブレーション (issue #10) のヘルパ。
 *
 * UI とテストの両方から領域 ID ↔ rect のアクセスを 1 か所で扱うための薄いラッパ。
 * ネスト (rivers.{self,right,across,left}) を吸収して、領域 ID をフラットなキーで
 * 扱えるようにする。
 */

import type { RoiCalibration, RoiRect, RoiRegionId } from "@/types";

export interface RoiRegionDef {
  id: RoiRegionId;
  label: string;
}

/**
 * UI に並べる領域定義。順序がそのままキャリブレーション操作のフロー
 * (= 矩形確定後に自動で次の領域へ進む) になるので、雀魂の画面上で目につく順番で
 * 定義しておく。
 */
export const REGION_DEFS: readonly RoiRegionDef[] = [
  { id: "hand", label: "手牌" },
  { id: "doras", label: "ドラ表示" },
  { id: "river_self", label: "河 (自家)" },
  { id: "river_right", label: "河 (下家)" },
  { id: "river_across", label: "河 (対面)" },
  { id: "river_left", label: "河 (上家)" },
  // 副露 (issue #14)。加槓の積み牌を捉えるため、ROI は牌 1 段ぶん上に余白を含めて切る。
  { id: "meld_self", label: "副露 (自家)" },
  { id: "meld_right", label: "副露 (下家)" },
  { id: "meld_across", label: "副露 (対面)" },
  { id: "meld_left", label: "副露 (上家)" },
  { id: "round_info", label: "場況" },
  { id: "self_wind", label: "自風" },
  { id: "scores", label: "点棒" },
  { id: "turn_counter", label: "巡目" },
] as const;

/** 指定領域の rect を取得する。未指定なら `null`。 */
export function getRegionRect(calibration: RoiCalibration, region: RoiRegionId): RoiRect | null {
  switch (region) {
    case "hand":
      return calibration.hand;
    case "doras":
      return calibration.doras;
    case "round_info":
      return calibration.round_info;
    case "self_wind":
      return calibration.self_wind;
    case "scores":
      return calibration.scores;
    case "turn_counter":
      return calibration.turn_counter;
    case "river_self":
      return calibration.rivers.self;
    case "river_right":
      return calibration.rivers.right;
    case "river_across":
      return calibration.rivers.across;
    case "river_left":
      return calibration.rivers.left;
    case "meld_self":
      return calibration.melds.self;
    case "meld_right":
      return calibration.melds.right;
    case "meld_across":
      return calibration.melds.across;
    case "meld_left":
      return calibration.melds.left;
  }
}

/**
 * 指定領域の rect を差し替えた新しい `RoiCalibration` を返す (immutable)。
 *
 * `rect` に `null` を渡すとクリアになる。
 * `rivers` は構造を維持するためにフィールドごとに spread する。
 */
export function setRegionRect(
  calibration: RoiCalibration,
  region: RoiRegionId,
  rect: RoiRect | null,
): RoiCalibration {
  switch (region) {
    case "hand":
      return { ...calibration, hand: rect };
    case "doras":
      return { ...calibration, doras: rect };
    case "round_info":
      return { ...calibration, round_info: rect };
    case "self_wind":
      return { ...calibration, self_wind: rect };
    case "scores":
      return { ...calibration, scores: rect };
    case "turn_counter":
      return { ...calibration, turn_counter: rect };
    case "river_self":
      return { ...calibration, rivers: { ...calibration.rivers, self: rect } };
    case "river_right":
      return { ...calibration, rivers: { ...calibration.rivers, right: rect } };
    case "river_across":
      return { ...calibration, rivers: { ...calibration.rivers, across: rect } };
    case "river_left":
      return { ...calibration, rivers: { ...calibration.rivers, left: rect } };
    case "meld_self":
      return { ...calibration, melds: { ...calibration.melds, self: rect } };
    case "meld_right":
      return { ...calibration, melds: { ...calibration.melds, right: rect } };
    case "meld_across":
      return { ...calibration, melds: { ...calibration.melds, across: rect } };
    case "meld_left":
      return { ...calibration, melds: { ...calibration.melds, left: rect } };
  }
}

/** 1 個でも領域が設定されていれば `true`。設定画面のサマリ表示に使う。 */
export function hasAnyRoi(calibration: RoiCalibration | null | undefined): boolean {
  if (!calibration) return false;
  return REGION_DEFS.some((r) => getRegionRect(calibration, r.id) !== null);
}

/** 設定済み領域の数を返す。設定画面のバッジに使う。 */
export function countRoi(calibration: RoiCalibration | null | undefined): number {
  if (!calibration) return 0;
  return REGION_DEFS.reduce(
    (acc, r) => acc + (getRegionRect(calibration, r.id) !== null ? 1 : 0),
    0,
  );
}
