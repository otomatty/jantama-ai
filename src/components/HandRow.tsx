import { Tile } from "@/components/Tile";
import type { GameBoardSummary, InferenceResult } from "@/types";

interface HandRowProps {
  board: GameBoardSummary | null;
  inference: InferenceResult | null;
}

export function HandRow({ board, inference }: HandRowProps) {
  if (!board) return null;
  const hand = board.hand;
  const recommendedTile =
    inference?.recommended.action_type === "discard"
      ? inference.recommended.tile
      : null;

  return (
    <div className="border-t border-ink-200 bg-white px-3 py-2.5">
      <div className="mb-1.5 flex items-center justify-between">
        <span className="font-sans text-[12px] font-bold uppercase tracking-[0.18em] text-ink-500">
          手牌
        </span>
        <span className="font-mono text-[10px] tabular-nums text-ink-400">
          {hand.length} 牌
        </span>
      </div>
      <div className="flex items-end gap-0.5">
        {hand.slice(0, 13).map((t, i) => (
          <Tile
            key={`${t}-${i}`}
            code={t}
            size="sm"
            highlight={recommendedTile === t && hand.indexOf(recommendedTile) === i}
          />
        ))}
        {hand.length === 14 && (
          <>
            <div className="w-1" />
            <Tile code={hand[13]} size="sm" />
          </>
        )}
      </div>
    </div>
  );
}
