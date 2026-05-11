/**
 * モック対局シナリオ。
 * Tauri バックエンドと繋がっていないブラウザ実行時、設定画面のプレビュー、
 * `runStubInference` のフォールバックなどで利用する。
 */

import type { GameBoardSummary, InferenceResult } from "@/types";

export type ScenarioKey = "dahai" | "riichi" | "fuuro" | "agari";

interface ScenarioFixture {
  inference: InferenceResult;
  board: GameBoardSummary;
}

// Hand fixtures — same physical hands the design canvas uses.
const HAND_TENPAI = ["1m", "2m", "3m", "4m", "5m", "6m", "7p", "8p", "9p", "1z", "2z", "3z", "5m"];
const HAND_MENZEN = ["2m", "3m", "4m", "6m", "7m", "8m", "3p", "4p", "5p", "7s", "8s", "9s", "7z"];
const HAND_AGARI = ["1m", "2m", "3m", "4m", "5m", "6m", "7p", "8p", "9p", "1z", "1z", "1z", "5p"];

const COMMON_BOARD: GameBoardSummary = {
  hand: HAND_TENPAI,
  self_wind: "東",
  round_wind: "東",
  turn: 6,
  dora_indicators: ["5p"],
  score: 25000,
  round_label: "東1局",
  // issue #15: ブラウザスタブは「自分の手番」状態の固定値で動くので、
  // フロントの IdleBody ゲートを通過するよう my_turn=true を立てる。
  // シナリオ別に available_actions を上書きする (リーチ可能 / 鳴き / 和了)。
  my_turn: true,
  available_actions: ["discard"],
};

export const SCENARIO_FIXTURES: Record<ScenarioKey, ScenarioFixture> = {
  dahai: {
    inference: {
      timestamp: new Date().toISOString(),
      primary_label: "6m を切る",
      recommended: {
        tile: "6m",
        action_type: "discard",
        action_label: "打牌",
        expected_value: 0.32,
        probability: 0.61,
      },
      candidates: [
        {
          tile: "6m",
          action_type: "discard",
          action_label: "打牌",
          expected_value: 0.32,
          probability: 0.61,
        },
        {
          tile: "9p",
          action_type: "discard",
          action_label: "打牌",
          expected_value: 0.18,
          probability: 0.22,
        },
        {
          tile: "1z",
          action_type: "discard",
          action_label: "打牌",
          expected_value: -0.05,
          probability: 0.11,
        },
        {
          tile: "3s",
          action_type: "discard",
          action_label: "打牌",
          expected_value: -0.21,
          probability: 0.06,
        },
      ],
      reason:
        "受け入れ最大の打。6mを切るとピンフ・三色の両天秤で、シャンテン戻しなく17種56牌の有効牌が残る。9pは安全度が一段下がり、ドラ表示牌5pの周辺で他家リーチへの放銃率が上がる。",
      danger: [
        { tile: "7p", level: "high" },
        { tile: "3z", level: "mid" },
      ],
      safe: ["1z", "4z", "9s"],
    },
    board: { ...COMMON_BOARD, hand: HAND_TENPAI, available_actions: ["discard"] },
  },
  riichi: {
    inference: {
      timestamp: new Date().toISOString(),
      primary_label: "リーチを宣言",
      recommended: {
        action_type: "riichi",
        action_label: "リーチ",
        expected_value: 0.84,
        probability: 0.78,
      },
      candidates: [
        {
          action_type: "riichi",
          action_label: "リーチ",
          expected_value: 0.84,
          probability: 0.78,
        },
        {
          action_type: "discard",
          action_label: "ダマ",
          tile: "5p",
          expected_value: 0.21,
          probability: 0.14,
        },
        {
          action_type: "discard",
          action_label: "ダマ",
          tile: "2s",
          expected_value: 0.05,
          probability: 0.08,
        },
      ],
      reason:
        "良形テンパイ・打点上昇余地・順目に余裕があり、リーチ宣言の期待値が圧倒的に高い。三人の捨牌は無筋少なく、放銃リスクは中庸。裏ドラ・一発の打点込みで+0.84の優位。",
    },
    board: {
      ...COMMON_BOARD,
      hand: HAND_TENPAI,
      available_actions: ["discard", "riichi"],
    },
  },
  fuuro: {
    inference: {
      timestamp: new Date().toISOString(),
      primary_label: "ポンせずスルー",
      recommended: {
        action_type: "pass",
        action_label: "スルー",
        expected_value: 0.12,
        probability: 0.58,
      },
      candidates: [
        {
          action_type: "pass",
          action_label: "スルー",
          expected_value: 0.12,
          probability: 0.58,
        },
        {
          action_type: "pon",
          action_label: "ポン",
          expected_value: 0.04,
          probability: 0.27,
        },
        {
          action_type: "chi",
          action_label: "チー",
          expected_value: -0.08,
          probability: 0.15,
        },
      ],
      reason:
        "門前を維持した方が、リーチ込みの打点期待値が高い。中をポンしても役は確定するが、形が崩れて受け入れが14牌→9牌に減少。3巡目で局速はまだ十分。",
    },
    board: {
      ...COMMON_BOARD,
      hand: HAND_MENZEN,
      turn: 3,
      available_actions: ["pon", "chi", "pass"],
    },
  },
  agari: {
    inference: {
      timestamp: new Date().toISOString(),
      primary_label: "ロンで和了",
      recommended: {
        action_type: "ron",
        action_label: "ロン",
        expected_value: 1.0,
        probability: 0.99,
      },
      candidates: [
        {
          action_type: "ron",
          action_label: "ロン",
          expected_value: 1.0,
          probability: 0.99,
        },
        {
          action_type: "pass",
          action_label: "見逃し",
          expected_value: -0.21,
          probability: 0.01,
        },
      ],
      reason:
        "リーチ・ピンフ・ツモ無し・ドラ1で5,200点。トップ目との点差を考えると見逃しの価値はなく、即和了が最善。残り2局・親番なし。",
    },
    board: {
      ...COMMON_BOARD,
      hand: HAND_AGARI,
      available_actions: ["ron", "pass"],
    },
  },
};

const SCENARIO_ROTATION: ScenarioKey[] = ["dahai", "riichi", "fuuro", "agari"];

let cursor = 0;

/**
 * Demo 用にシナリオをローテートして返す (ブラウザ実行時のスタブ推論)。
 * 呼び出し側がレスポンスを変更してもフィクスチャ本体を汚染しないよう、
 * structuredClone でディープコピーして返す。
 */
export function nextStubScenario(): ScenarioFixture {
  const key = SCENARIO_ROTATION[cursor % SCENARIO_ROTATION.length];
  cursor += 1;
  return cloneScenario(SCENARIO_FIXTURES[key]);
}

export function getScenario(key: ScenarioKey): ScenarioFixture {
  return cloneScenario(SCENARIO_FIXTURES[key]);
}

function cloneScenario(base: ScenarioFixture): ScenarioFixture {
  const cloned = structuredClone(base);
  cloned.inference.timestamp = new Date().toISOString();
  return cloned;
}
