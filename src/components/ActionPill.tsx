interface ActionPillProps {
  kind: string;
}

interface PillStyle {
  background: string;
  color: string;
}

const PILL_MAP: Record<string, PillStyle> = {
  リーチ: { background: "linear-gradient(135deg, #0432FF, #FF2600)", color: "#fff" },
  和了: { background: "#138A4F", color: "#fff" },
  ロン: { background: "#138A4F", color: "#fff" },
  ツモ: { background: "#138A4F", color: "#fff" },
  打牌: { background: "#0F0F1E", color: "#fff" },
  ポン: { background: "#1F1F2E", color: "#fff" },
  チー: { background: "#1F1F2E", color: "#fff" },
  カン: { background: "#1F1F2E", color: "#fff" },
  スルー: { background: "#F2F1EC", color: "#0F0F1E" },
  ダマ: { background: "#F2F1EC", color: "#0F0F1E" },
  見逃し: { background: "#F2F1EC", color: "#7A7A8C" },
  鳴き選択: { background: "#0F0F1E", color: "#fff" },
};

export function ActionPill({ kind }: ActionPillProps) {
  const style = PILL_MAP[kind] ?? PILL_MAP["打牌"];
  return (
    <span
      className="inline-flex items-center rounded-full px-2.5 py-[3px] font-sans text-[10px] font-bold uppercase tracking-[0.1em]"
      style={{ background: style.background, color: style.color }}
    >
      {kind}
    </span>
  );
}
