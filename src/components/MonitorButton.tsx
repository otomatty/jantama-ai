import { cn } from "@/lib/utils";

interface MonitorButtonProps {
  on: boolean;
  disabled?: boolean;
  onClick: () => void;
}

export function MonitorButton({ on, disabled, onClick }: MonitorButtonProps) {
  return (
    <button
      type="button"
      disabled={disabled}
      onClick={onClick}
      className={cn(
        "flex w-full items-center justify-center gap-2 rounded-lg px-4 py-3 font-jp text-[14px] font-bold tracking-[0.04em] text-ink-50 transition-opacity disabled:cursor-not-allowed disabled:opacity-50",
        on ? "bg-ink-900" : "",
      )}
      style={!on ? { background: "var(--gradient-acial)" } : undefined}
    >
      <span
        className="h-2 w-2 rounded-full"
        style={{ background: on ? "#FF2600" : "#fff" }}
      />
      {on ? "監視中 — 停止する" : "監視を開始する"}
    </button>
  );
}
