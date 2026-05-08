import { Tile, type TileSize, isTileCode } from "@/components/Tile";

interface PrimaryGlyphProps {
  /** 牌コード or 動詞 (例: "6m" / "リーチ" / "ロン") */
  value: string;
  size?: TileSize;
}

const GRADIENT_VERBS = new Set(["リーチ", "ロン", "ツモ"]);

export function PrimaryGlyph({ value, size = "xl" }: PrimaryGlyphProps) {
  if (isTileCode(value)) {
    return <Tile code={value} size={size} highlight />;
  }
  return <VerbCard label={value} size={size} />;
}

function VerbCard({ label, size = "xl" }: { label: string; size: TileSize }) {
  const dimsMap: Record<TileSize, number> = {
    xs: 28,
    sm: 40,
    md: 56,
    lg: 80,
    xl: 92,
    xxl: 140,
  };
  const dims = dimsMap[size];
  const isGradient = GRADIENT_VERBS.has(label);
  return (
    <div
      className="flex items-center justify-center rounded-[10px] font-jp font-extrabold tracking-[0.02em] text-white"
      style={{
        width: dims,
        height: dims * 1.3,
        background: isGradient
          ? "linear-gradient(135deg, #0432FF 0%, #FF2600 100%)"
          : "#0F0F1E",
        fontSize: dims * 0.32,
        boxShadow:
          "0 12px 40px rgba(15,15,30,0.18), 0 2px 6px rgba(15,15,30,0.08)",
      }}
    >
      {label}
    </div>
  );
}
