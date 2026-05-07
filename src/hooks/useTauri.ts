import { useEffect, useState } from "react";

/**
 * 実行環境が Tauri (デスクトップアプリ) か、ブラウザ単体か判定する Hook。
 * `npm run dev` でブラウザから動作確認する場合は false を返し、
 * Tauri 関連 API を呼ばないようにフォールバックする。
 */
export function useIsTauri(): boolean {
  const [isTauri, setIsTauri] = useState(false);

  useEffect(() => {
    // Tauri 2.x では window.__TAURI_INTERNALS__ で判定可能
    const w = window as unknown as { __TAURI_INTERNALS__?: unknown };
    setIsTauri(typeof w.__TAURI_INTERNALS__ !== "undefined");
  }, []);

  return isTauri;
}
