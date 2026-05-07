import { ConfBar } from "@/components/ConfBar";
import { Tile, isTileCode } from "@/components/Tile";
import { ACTION_LABEL } from "@/lib/actionLabels";
import type { RecommendationCandidate } from "@/types";

interface CandidateListProps {
  candidates: RecommendationCandidate[];
  startRank?: number;
}

export function CandidateList({
  candidates,
  startRank = 2,
}: CandidateListProps) {
  if (candidates.length === 0) return null;
  return (
    <div>
      <div className="mb-2 pl-0.5 font-sans text-[12px] font-bold uppercase tracking-[0.18em] text-ink-500">
        次点候補
      </div>
      <div className="flex flex-col gap-1.5">
        {candidates.map((c, i) => (
          <CandidateRow
            key={`${c.tile ?? c.action_type}-${i}`}
            candidate={c}
            rank={startRank + i}
          />
        ))}
      </div>
    </div>
  );
}

function CandidateRow({
  candidate,
  rank,
}: {
  candidate: RecommendationCandidate;
  rank: number;
}) {
  const display =
    candidate.tile ?? candidate.action_label ?? ACTION_LABEL[candidate.action_type];
  const tileCode = candidate.tile && isTileCode(candidate.tile) ? candidate.tile : null;
  const ev = candidate.expected_value;
  return (
    <div className="flex items-center gap-2.5 rounded-lg border border-ink-200 bg-white px-3 py-2">
      <span className="w-3.5 font-mono text-[10px] tabular-nums text-ink-400">
        {String(rank).padStart(2, "0")}
      </span>
      <div className="flex w-[30px] justify-center">
        {tileCode ? (
          <Tile code={tileCode} size="sm" />
        ) : (
          <span className="font-jp text-[13px] font-bold text-ink-700">
            {display}
          </span>
        )}
      </div>
      <div className="flex-1">
        <ConfBar value={candidate.probability ?? 0} color="var(--color-ink-300)" height={3} />
      </div>
      <div
        className={
          "min-w-[50px] text-right font-sans text-[13px] font-bold tabular-nums " +
          (ev >= 0 ? "text-ink-900" : "text-ink-500")
        }
      >
        {ev >= 0 ? "+" : ""}
        {ev.toFixed(2)}
      </div>
    </div>
  );
}
