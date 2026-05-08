interface ReasonBlockProps {
  reason: string | null | undefined;
}

export function ReasonBlock({ reason }: ReasonBlockProps) {
  if (!reason) return null;
  return (
    <div className="rounded-lg border border-ink-100 bg-ink-50 px-3 py-2.5">
      <div className="mb-1.5 flex items-center gap-1.5">
        <span
          className="inline-flex h-4 w-4 items-center justify-center rounded-[3px] font-sans text-[8px] font-extrabold uppercase tracking-[0.04em] text-white"
          style={{ background: "var(--gradient-acial)" }}
        >
          AI
        </span>
        <span className="font-sans text-[12px] font-bold uppercase tracking-[0.18em] text-ink-500">
          Reasoning
        </span>
      </div>
      <p className="font-jp text-[14px] leading-[1.6] text-ink-700">{reason}</p>
    </div>
  );
}
