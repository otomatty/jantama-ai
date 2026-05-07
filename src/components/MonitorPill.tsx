import { cn } from "@/lib/utils";

interface MonitorPillProps {
  on: boolean;
  className?: string;
}

export function MonitorPill({ on, className }: MonitorPillProps) {
  return (
    <div
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full py-[3px] pl-[6px] pr-[8px] font-sans text-[10px] font-semibold uppercase tracking-[0.08em]",
        on ? "bg-ink-900 text-ink-50" : "bg-ink-100 text-ink-600",
        className,
      )}
    >
      <span
        className={cn(
          "h-1.5 w-1.5 rounded-full",
          on ? "bg-acial-red animate-jt-pulse" : "bg-ink-400",
        )}
        style={on ? { boxShadow: "0 0 0 3px rgba(255,38,0,0.2)" } : undefined}
      />
      <span>{on ? "LIVE" : "OFF"}</span>
    </div>
  );
}
