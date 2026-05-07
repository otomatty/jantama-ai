/**
 * Mahjong tile — SVG flat-style.
 *
 * Codes:
 *   - 1m..9m: 萬子 (man)
 *   - 1p..9p: 筒子 (pin)
 *   - 1s..9s: 索子 (sou)
 *   - 1z..7z: 字牌 (honors: E S W N P F C / 東南西北白發中)
 */

export type TileSize = "xs" | "sm" | "md" | "lg" | "xl" | "xxl";

const HONOR_LABEL: Record<string, string> = {
  "1z": "東",
  "2z": "南",
  "3z": "西",
  "4z": "北",
  "5z": "白",
  "6z": "發",
  "7z": "中",
};

const HONOR_COLOR: Record<string, string> = {
  "6z": "#138A4F",
  "7z": "#C71B00",
};

const MAN_NUM = ["一", "二", "三", "四", "五", "六", "七", "八", "九"];

const SIZES: Record<TileSize, { w: number; h: number }> = {
  xs: { w: 22, h: 30 },
  sm: { w: 30, h: 42 },
  md: { w: 42, h: 58 },
  lg: { w: 56, h: 76 },
  xl: { w: 80, h: 108 },
  xxl: { w: 110, h: 148 },
};

interface TileProps {
  code?: string;
  red?: boolean;
  dim?: boolean;
  ghost?: boolean;
  size?: TileSize;
  rotate?: number;
  back?: boolean;
  highlight?: boolean;
  className?: string;
}

function PinDots({ n, red }: { n: number; red?: boolean }) {
  const positions: Record<number, [number, number][]> = {
    1: [[0.5, 0.5]],
    2: [
      [0.5, 0.28],
      [0.5, 0.72],
    ],
    3: [
      [0.28, 0.28],
      [0.5, 0.5],
      [0.72, 0.72],
    ],
    4: [
      [0.28, 0.28],
      [0.72, 0.28],
      [0.28, 0.72],
      [0.72, 0.72],
    ],
    5: [
      [0.28, 0.28],
      [0.72, 0.28],
      [0.5, 0.5],
      [0.28, 0.72],
      [0.72, 0.72],
    ],
    6: [
      [0.28, 0.25],
      [0.72, 0.25],
      [0.28, 0.5],
      [0.72, 0.5],
      [0.28, 0.75],
      [0.72, 0.75],
    ],
    7: [
      [0.28, 0.22],
      [0.5, 0.22],
      [0.72, 0.22],
      [0.5, 0.5],
      [0.28, 0.78],
      [0.5, 0.78],
      [0.72, 0.78],
    ],
    8: [
      [0.28, 0.22],
      [0.72, 0.22],
      [0.28, 0.42],
      [0.72, 0.42],
      [0.28, 0.62],
      [0.72, 0.62],
      [0.28, 0.82],
      [0.72, 0.82],
    ],
    9: [
      [0.28, 0.22],
      [0.5, 0.22],
      [0.72, 0.22],
      [0.28, 0.5],
      [0.5, 0.5],
      [0.72, 0.5],
      [0.28, 0.78],
      [0.5, 0.78],
      [0.72, 0.78],
    ],
  };
  const pts = positions[n] ?? [];
  return (
    <>
      {pts.map(([x, y], i) => (
        <circle
          key={i}
          cx={x * 60}
          cy={y * 80}
          r={6.5}
          fill={red ? "#FF2600" : "#0F0F1E"}
        />
      ))}
    </>
  );
}

function SouLines({ n, red }: { n: number; red?: boolean }) {
  const c = red ? "#FF2600" : "#138A4F";
  if (n === 1) {
    return (
      <g>
        <rect x={26} y={20} width={8} height={36} rx={4} fill={c} />
        <circle cx={30} cy={14} r={5} fill="#C71B00" />
        <rect x={20} y={58} width={20} height={4} rx={2} fill={c} />
      </g>
    );
  }
  const stick = (cx: number, cy: number, key: string) => (
    <g key={key}>
      <rect x={cx - 2.5} y={cy - 9} width={5} height={18} rx={2} fill={c} />
      <circle cx={cx} cy={cy - 11} r={2.5} fill={c} />
      <circle cx={cx} cy={cy + 11} r={2.5} fill={c} />
    </g>
  );
  const layouts: Record<number, [number, number][]> = {
    2: [
      [30, 26],
      [30, 54],
    ],
    3: [
      [18, 40],
      [30, 40],
      [42, 40],
    ],
    4: [
      [20, 26],
      [40, 26],
      [20, 54],
      [40, 54],
    ],
    5: [
      [20, 26],
      [40, 26],
      [30, 40],
      [20, 54],
      [40, 54],
    ],
    6: [
      [18, 26],
      [30, 26],
      [42, 26],
      [18, 54],
      [30, 54],
      [42, 54],
    ],
    7: [
      [18, 22],
      [30, 22],
      [42, 22],
      [30, 40],
      [18, 58],
      [30, 58],
      [42, 58],
    ],
    8: [
      [18, 22],
      [30, 22],
      [42, 22],
      [18, 40],
      [42, 40],
      [18, 58],
      [30, 58],
      [42, 58],
    ],
    9: [
      [18, 22],
      [30, 22],
      [42, 22],
      [18, 40],
      [30, 40],
      [42, 40],
      [18, 58],
      [30, 58],
      [42, 58],
    ],
  };
  return <>{(layouts[n] ?? []).map(([x, y]) => stick(x, y, `${x}-${y}`))}</>;
}

function ManKanji({ n, red }: { n: number; red?: boolean }) {
  return (
    <g>
      <text
        x={30}
        y={32}
        textAnchor="middle"
        fontSize={22}
        fontWeight={700}
        fontFamily='"Noto Sans JP", serif'
        fill={red ? "#FF2600" : "#0F0F1E"}
      >
        {MAN_NUM[n - 1]}
      </text>
      <text
        x={30}
        y={62}
        textAnchor="middle"
        fontSize={18}
        fontWeight={700}
        fontFamily='"Noto Sans JP", serif'
        fill={red ? "#FF2600" : "#C71B00"}
      >
        萬
      </text>
    </g>
  );
}

function HonorGlyph({ code }: { code: string }) {
  if (code === "5z") {
    return (
      <rect
        x={14}
        y={16}
        width={32}
        height={48}
        rx={3}
        fill="none"
        stroke="#0F0F1E"
        strokeWidth={2.5}
      />
    );
  }
  const color = HONOR_COLOR[code] ?? "#0F0F1E";
  return (
    <text
      x={30}
      y={52}
      textAnchor="middle"
      fontSize={30}
      fontWeight={800}
      fontFamily='"Noto Sans JP", serif'
      fill={color}
    >
      {HONOR_LABEL[code]}
    </text>
  );
}

export function Tile({
  code,
  red,
  dim,
  ghost,
  size = "md",
  rotate = 0,
  back = false,
  highlight = false,
  className,
}: TileProps) {
  const { w, h } = SIZES[size] ?? SIZES.md;

  if (back) {
    return (
      <div
        className={className}
        style={{
          width: w,
          height: h,
          borderRadius: 6,
          background: "linear-gradient(135deg, #0432FF 0%, #FF2600 100%)",
          boxShadow:
            "0 1px 2px rgba(15,15,30,0.18), inset 0 -2px 0 rgba(0,0,0,0.15), inset 0 1px 0 rgba(255,255,255,0.18)",
          opacity: dim ? 0.4 : 1,
        }}
      />
    );
  }

  const suit = code ? code[1] : null;
  const num = code ? parseInt(code[0], 10) : null;

  const tileStroke = highlight ? "url(#tile-grad-stroke)" : "#0F0F1E";
  const strokeW = highlight ? 2.5 : 1.2;

  return (
    <div
      className={className}
      style={{
        width: w,
        height: h,
        opacity: dim ? 0.32 : 1,
        transform: rotate ? `rotate(${rotate}deg)` : undefined,
        transformOrigin: "center",
        filter: highlight
          ? "drop-shadow(0 4px 14px rgba(4,50,255,0.25))"
          : ghost
            ? "none"
            : "drop-shadow(0 1px 1px rgba(15,15,30,0.10))",
      }}
    >
      <svg
        viewBox="0 0 60 80"
        width={w}
        height={h}
        style={{ display: "block" }}
      >
        <defs>
          <linearGradient
            id="tile-grad-stroke"
            x1="0%"
            y1="0%"
            x2="100%"
            y2="100%"
          >
            <stop offset="0%" stopColor="#0432FF" />
            <stop offset="100%" stopColor="#FF2600" />
          </linearGradient>
          <linearGradient
            id="tile-face-shade"
            x1="0%"
            y1="0%"
            x2="0%"
            y2="100%"
          >
            <stop offset="0%" stopColor="#FFFFFF" />
            <stop offset="100%" stopColor="#F0EFE9" />
          </linearGradient>
        </defs>
        <rect
          x={1.5}
          y={1.5}
          width={57}
          height={77}
          rx={6}
          ry={6}
          fill={ghost ? "none" : "url(#tile-face-shade)"}
          stroke={tileStroke}
          strokeWidth={strokeW}
          strokeDasharray={ghost ? "3 3" : undefined}
        />
        {!ghost && (
          <rect
            x={3}
            y={3}
            width={54}
            height={74}
            rx={5}
            fill="none"
            stroke="rgba(15,15,30,0.06)"
            strokeWidth={1}
          />
        )}
        {suit === "m" && num !== null && <ManKanji n={num} red={red} />}
        {suit === "p" && num !== null && <PinDots n={num} red={red} />}
        {suit === "s" && num !== null && <SouLines n={num} red={red} />}
        {suit === "z" && code && <HonorGlyph code={code} />}
      </svg>
    </div>
  );
}

export function isTileCode(s: string): boolean {
  return /^[1-9][mpsz]$/.test(s);
}
