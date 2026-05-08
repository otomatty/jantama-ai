import { useEffect, useState } from "react";

/**
 * 現在時刻を `HH:MM:SS` 形式で 1 秒刻みに返すフック。
 * StatusBar の右上時刻表示に使う。
 */
export function useNow(): string {
  const [now, setNow] = useState<string>(formatNow);

  useEffect(() => {
    const id = setInterval(() => setNow(formatNow()), 1000);
    return () => clearInterval(id);
  }, []);

  return now;
}

function formatNow(): string {
  const d = new Date();
  return d.toLocaleTimeString("ja-JP", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  });
}
