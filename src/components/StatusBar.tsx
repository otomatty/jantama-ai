import { Settings } from "lucide-react";
import { MonitorPill } from "@/components/MonitorPill";
import { useNow } from "@/hooks/useNow";

interface StatusBarProps {
  monitoring: boolean;
  onOpenSettings?: () => void;
}

export function StatusBar({ monitoring, onOpenSettings }: StatusBarProps) {
  const now = useNow();
  return (
    <div
      className="flex items-center justify-between border-b border-ink-200 px-3.5 py-2.5"
      style={{
        background: "rgba(255,255,255,0.92)",
        backdropFilter: "blur(12px) saturate(160%)",
        WebkitBackdropFilter: "blur(12px) saturate(160%)",
      }}
    >
      <div className="flex items-center gap-2.5">
        <div className="flex items-baseline gap-px font-sans font-black tracking-[-0.02em] leading-none">
          <span className="gradient-text text-[17px]">雀</span>
          <span className="text-ink-900 text-[14px]">tama</span>
          <span className="ml-1.5 text-[11px] font-medium tracking-[0.12em] text-ink-400">
            AI
          </span>
        </div>
        <span className="h-3 w-px bg-ink-200" />
        <MonitorPill on={monitoring} />
      </div>
      <div className="flex items-center gap-1.5 font-mono text-[11px] text-ink-500">
        <span className="tabular-nums">{now}</span>
        <span className="h-2.5 w-px bg-ink-200" />
        {onOpenSettings && (
          <button
            type="button"
            title="設定"
            onClick={onOpenSettings}
            className="inline-flex h-[22px] w-[22px] items-center justify-center rounded text-ink-600 transition-colors hover:bg-ink-100 hover:text-ink-900"
          >
            <Settings className="h-3.5 w-3.5" strokeWidth={1.6} />
          </button>
        )}
      </div>
    </div>
  );
}
