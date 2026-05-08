import type { AppError } from "@/types";

interface ErrorBodyProps {
  error: AppError | null;
  onRetry?: () => void;
  onOpenSettings?: () => void;
}

const ERROR_TITLE: Record<AppError["type"], string> = {
  recognition: "盤面を認識できません",
  inference: "推論に失敗しました",
  capture: "キャプチャ対象が見つかりません",
  config: "設定に不備があります",
  unknown: "不明なエラーが発生しました",
};

const FALLBACK_MESSAGE =
  "「雀魂 - Mahjong Soul」ウィンドウを再検出しています。雀魂が起動しているかご確認ください。";

export function ErrorBody({ error, onRetry, onOpenSettings }: ErrorBodyProps) {
  const title = error ? ERROR_TITLE[error.type] : ERROR_TITLE.unknown;
  const message = error?.message ?? FALLBACK_MESSAGE;
  return (
    <div className="flex flex-1 flex-col items-center justify-center gap-3.5 px-8 py-12 text-center">
      <div
        className="flex h-16 w-16 items-center justify-center rounded-full text-danger"
        style={{ background: "var(--color-danger-bg)" }}
      >
        <svg
          width={28}
          height={28}
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth={2}
        >
          <path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z" />
          <line x1={12} y1={9} x2={12} y2={13} />
          <line x1={12} y1={17} x2={12.01} y2={17} />
        </svg>
      </div>
      <div className="font-jp text-[20px] font-bold leading-tight text-ink-900">{title}</div>
      <p className="max-w-[300px] font-jp text-[14px] leading-[1.6] text-ink-600">{message}</p>
      <div className="mt-1.5 flex gap-2">
        {onRetry && (
          <button
            type="button"
            onClick={onRetry}
            className="cursor-pointer rounded-md border border-ink-200 bg-white px-3.5 py-2 font-jp text-[12px] font-semibold transition-colors hover:bg-ink-50"
          >
            再検出
          </button>
        )}
        {onOpenSettings && (
          <button
            type="button"
            onClick={onOpenSettings}
            className="cursor-pointer rounded-md border-0 bg-ink-900 px-3.5 py-2 font-jp text-[12px] font-semibold text-white transition-opacity hover:opacity-90"
          >
            設定を開く
          </button>
        )}
      </div>
    </div>
  );
}
