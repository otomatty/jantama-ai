import { Tile } from "@/components/Tile";
import type { GameBoardSummary } from "@/types";

interface ContextBarProps {
  board: GameBoardSummary | null;
}

export function ContextBar({ board }: ContextBarProps) {
  const round = board?.round_label ?? "東1局";
  const turn = board?.turn ?? 0;
  const selfWind = board?.self_wind ?? "東";
  const score = board?.score ?? 25000;
  const dora = board?.dora_indicators?.[0];

  return (
    <div className="flex items-center justify-between border-b border-ink-100 bg-ink-50 px-3.5 py-2 font-sans text-[11px]">
      <div className="flex items-center gap-3.5 text-ink-600">
        <span>
          <strong className="font-bold text-ink-900">{round}</strong> {turn}
          巡目
        </span>
        <span>
          自風 <strong className="font-bold text-ink-900">{selfWind}</strong>
        </span>
        <span>
          持点{" "}
          <strong className="font-bold text-ink-900 tabular-nums">
            {score.toLocaleString("en-US")}
          </strong>
        </span>
      </div>
      <div className="flex items-center gap-1.5 text-ink-600">
        <span className="text-[10px] uppercase tracking-[0.1em]">Dora</span>
        {dora && <Tile code={dora} size="xs" />}
      </div>
    </div>
  );
}
