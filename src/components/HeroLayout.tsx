import { ActionPill } from "@/components/ActionPill";
import { CandidateList } from "@/components/CandidateList";
import { ConfBar } from "@/components/ConfBar";
import { PrimaryGlyph } from "@/components/PrimaryGlyph";
import type { InferenceResult, RecommendationCandidate } from "@/types";
import { ACTION_LABEL } from "@/lib/actionLabels";

interface HeroLayoutProps {
  inference: InferenceResult;
}

function glyphValue(c: RecommendationCandidate): string {
  return c.tile ?? c.action_label ?? ACTION_LABEL[c.action_type];
}

function actionLabel(c: RecommendationCandidate): string {
  return c.action_label ?? ACTION_LABEL[c.action_type];
}

export function HeroLayout({ inference }: HeroLayoutProps) {
  const { recommended, candidates } = inference;
  const top = candidates[0] ?? recommended;
  const probability = top.probability ?? recommended.probability ?? 0;
  const evScore = formatEv(recommended.expected_value);

  return (
    <div className="flex flex-col gap-3">
      {/* Top hero card */}
      <div
        className="relative flex items-center gap-3.5 overflow-hidden rounded-xl border border-ink-200 bg-white px-4 py-3.5"
        style={{ boxShadow: "var(--shadow-2)" }}
      >
        <div
          aria-hidden
          className="pointer-events-none absolute inset-0 opacity-50 gradient-halo"
        />
        <div className="relative z-10">
          <PrimaryGlyph value={glyphValue(recommended)} size="xl" />
        </div>
        <div className="relative z-10 flex min-w-0 flex-1 flex-col">
          <div className="mb-1.5 flex items-center gap-1.5">
            <ActionPill kind={actionLabel(recommended)} />
            <span className="font-sans text-[9px] font-bold tracking-[0.08em] text-ink-500">
              RANK 1
            </span>
          </div>
          <div className="mb-1.5 font-jp text-[20px] font-bold leading-tight text-ink-900">
            {inference.primary_label ?? actionLabel(recommended)}
          </div>
          <div className="flex items-baseline gap-1.5">
            <span className="font-sans text-[9px] font-bold tracking-[0.1em] text-ink-500">
              EV
            </span>
            <span className="gradient-text font-sans text-[26px] font-extrabold tracking-[-0.02em] tabular-nums">
              {evScore}
            </span>
          </div>
          <div className="mt-1.5">
            <ConfBar value={probability} />
            <div className="mt-[3px] flex items-center justify-between font-sans text-[9px] tracking-[0.04em] text-ink-500">
              <span>確信度</span>
              <span className="font-mono tabular-nums">
                {Math.round(probability * 100)}%
              </span>
            </div>
          </div>
        </div>
      </div>

      <CandidateList candidates={candidates.slice(1)} startRank={2} />
    </div>
  );
}

function formatEv(value: number): string {
  const sign = value >= 0 ? "+" : "";
  return `${sign}${value.toFixed(2)}`;
}
