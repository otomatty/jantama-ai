interface ConfBarProps {
  /** 0..1 の確信度 */
  value: number;
  color?: string;
  height?: number;
  className?: string;
}

export function ConfBar({
  value,
  color = "var(--color-ink-900)",
  height = 4,
  className,
}: ConfBarProps) {
  const pct = Math.max(0, Math.min(100, value * 100));
  return (
    <div
      className={"w-full overflow-hidden rounded-full bg-ink-100 " + (className ?? "")}
      style={{ height }}
    >
      <div
        className="h-full rounded-full transition-[width] duration-300"
        style={{
          width: `${pct}%`,
          background: color,
          transitionTimingFunction: "var(--ease-out)",
        }}
      />
    </div>
  );
}
