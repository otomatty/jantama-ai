import { Tile } from "@/components/Tile";
import type { GameBoardSummary } from "@/types";

interface ContextBarProps {
  board: GameBoardSummary | null;
}

const PLACEHOLDER = "—";

export function ContextBar({ board }: ContextBarProps) {
  // 盤面が認識されていないときは値を捏造せず、プレースホルダー (—) で
  // 「コンテキスト未取得」を明示する。
  const round = board?.round_label ?? PLACEHOLDER;
  const turn = board?.turn;
  const selfWind = board?.self_wind ?? PLACEHOLDER;
  const score = board?.score;
  const dora = board?.dora_indicators?.[0];

  return (
    <div className="flex items-center justify-between border-b border-ink-100 bg-ink-50 px-3.5 py-2 font-sans text-[11px]">
      <div className="flex items-center gap-3.5 text-ink-600">
        <span>
          <strong className="font-bold text-ink-900">{round}</strong>{" "}
          {turn !== undefined ? `${turn}巡目` : PLACEHOLDER}
        </span>
        <span>
          自風 <strong className="font-bold text-ink-900">{selfWind}</strong>
        </span>
        <span>
          持点{" "}
          <strong className="font-bold text-ink-900 tabular-nums">
            {score !== undefined ? score.toLocaleString("en-US") : PLACEHOLDER}
          </strong>
        </span>
      </div>
      <div className="flex items-center gap-1.5 text-ink-600">
        <span className="text-[10px] uppercase tracking-[0.1em]">Dora</span>
        {dora ? (
          <Tile code={dora} size="xs" />
        ) : (
          <span className="font-mono text-ink-400">{PLACEHOLDER}</span>
        )}
      </div>
    </div>
  );
}
