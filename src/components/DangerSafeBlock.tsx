import { Tile } from "@/components/Tile";
import type { DangerTile } from "@/types";

interface DangerSafeBlockProps {
  danger: DangerTile[] | undefined;
  safe: string[] | undefined;
}

export function DangerSafeBlock({ danger, safe }: DangerSafeBlockProps) {
  const hasDanger = danger && danger.length > 0;
  const hasSafe = safe && safe.length > 0;
  if (!hasDanger && !hasSafe) return null;

  return (
    <div
      className={
        "grid gap-2 " + (hasDanger && hasSafe ? "grid-cols-2" : "grid-cols-1")
      }
    >
      {hasDanger && (
        <div
          className="rounded-lg px-2.5 py-2"
          style={{
            background: "var(--color-danger-bg)",
            border: "1px solid rgba(199,27,0,0.18)",
          }}
        >
          <div className="mb-1.5 font-sans text-[12px] font-bold uppercase tracking-[0.18em] text-danger">
            危険牌
          </div>
          <div className="flex items-center gap-1">
            {danger!.map((d, i) => (
              <Tile key={`${d.tile}-${i}`} code={d.tile} size="sm" />
            ))}
          </div>
        </div>
      )}
      {hasSafe && (
        <div
          className="rounded-lg px-2.5 py-2"
          style={{
            background: "var(--color-success-bg)",
            border: "1px solid rgba(19,138,79,0.18)",
          }}
        >
          <div className="mb-1.5 font-sans text-[12px] font-bold uppercase tracking-[0.18em] text-success">
            安全牌
          </div>
          <div className="flex items-center gap-1">
            {safe!.map((t, i) => (
              <Tile key={`${t}-${i}`} code={t} size="sm" />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
