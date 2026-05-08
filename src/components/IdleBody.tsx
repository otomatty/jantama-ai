/**
 * 監視中・盤面なし時に中央エリアへ表示する待機ビュー。
 */
export function IdleBody() {
  return (
    <div className="flex flex-1 flex-col items-center justify-center gap-4 px-8 py-12 text-center">
      <div className="flex h-20 w-20 items-center justify-center rounded-full bg-ink-100">
        <svg
          width={32}
          height={32}
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth={1.5}
          className="text-ink-500"
        >
          <circle cx={12} cy={12} r={9} />
          <path d="M12 7v5l3 2" />
        </svg>
      </div>
      <div className="font-jp text-[24px] font-bold leading-tight text-ink-900">
        対局を待機中
      </div>
      <div className="max-w-[280px] font-jp text-[14px] leading-[1.6] text-ink-600">
        雀魂のウィンドウを監視しています。自分の手番が来ると、ここに推奨アクションを表示します。
      </div>
      <div className="mt-2 flex gap-1">
        {[0, 1, 2].map((i) => (
          <span
            key={i}
            className="h-1.5 w-1.5 rounded-full bg-ink-300"
            style={{
              animation: `jt-pulse 1.4s ease-in-out ${i * 0.2}s infinite`,
            }}
          />
        ))}
      </div>
    </div>
  );
}
