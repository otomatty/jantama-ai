import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ChevronLeft } from "lucide-react";
import {
  EMPTY_ROI_CALIBRATION,
  type AppSettings,
  type RoiCalibration,
  type RoiRect,
  type RoiRegionId,
} from "@/types";
import { captureWindowForCalibration } from "@/lib/tauriCommands";
import { REGION_DEFS, getRegionRect, setRegionRect } from "@/lib/roiCalibration";
import { cn } from "@/lib/utils";

interface CalibrationScreenProps {
  settings: AppSettings;
  onBack: () => void;
  onSaved: (next: AppSettings) => void;
}

type Capture = {
  width: number;
  height: number;
  imageDataUrl: string;
};

type DragState = {
  region: RoiRegionId;
  // canvas 上 (CSS px) の起点・現在位置
  start: { x: number; y: number };
  current: { x: number; y: number };
};

export function CalibrationScreen({ settings, onBack, onSaved }: CalibrationScreenProps) {
  const [calibration, setCalibration] = useState<RoiCalibration>(
    settings.roi_calibration ?? EMPTY_ROI_CALIBRATION,
  );
  const [activeRegion, setActiveRegion] = useState<RoiRegionId>("hand");
  const [capture, setCapture] = useState<Capture | null>(null);
  const [capturing, setCapturing] = useState(false);
  const [captureError, setCaptureError] = useState<string | null>(null);
  const [drag, setDrag] = useState<DragState | null>(null);
  const canvasRef = useRef<HTMLDivElement>(null);

  const requestCapture = useCallback(async () => {
    setCapturing(true);
    setCaptureError(null);
    try {
      const result = await captureWindowForCalibration(settings.capture_target_window_id ?? "");
      setCapture({
        width: result.width,
        height: result.height,
        imageDataUrl: `data:image/png;base64,${result.image_b64}`,
      });
    } catch (e) {
      setCaptureError(e instanceof Error ? e.message : String(e));
    } finally {
      setCapturing(false);
    }
  }, [settings.capture_target_window_id]);

  // マウントと同時に 1 度キャプチャを取りに行く。失敗してもユーザは「再キャプチャ」
  // ボタンで再試行できる。
  useEffect(() => {
    void requestCapture();
  }, [requestCapture]);

  const onMouseDown = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!capture || !canvasRef.current) return;
    e.preventDefault();
    const rect = canvasRef.current.getBoundingClientRect();
    const point = {
      x: clamp01((e.clientX - rect.left) / rect.width) * rect.width,
      y: clamp01((e.clientY - rect.top) / rect.height) * rect.height,
    };
    setDrag({
      region: activeRegion,
      start: point,
      current: point,
    });
  };

  const onMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!drag || !canvasRef.current) return;
    const rect = canvasRef.current.getBoundingClientRect();
    setDrag({
      ...drag,
      current: {
        x: clamp01((e.clientX - rect.left) / rect.width) * rect.width,
        y: clamp01((e.clientY - rect.top) / rect.height) * rect.height,
      },
    });
  };

  const onMouseUp = (_e: React.MouseEvent<HTMLDivElement>) => {
    if (!drag || !canvasRef.current) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const ratio = rectToRatio(drag.start, drag.current, rect.width, rect.height);
    setDrag(null);
    if (!ratio) return;
    setCalibration((prev) => setRegionRect(prev, drag.region, ratio));
    // 連続入力で次の領域へ自動的に進ませると操作が分断されにくい。最後の項目で
    // 止まる挙動はそのまま (= 「全部終わった」ことが見て分かる)。
    const idx = REGION_DEFS.findIndex((r) => r.id === drag.region);
    if (idx >= 0 && idx + 1 < REGION_DEFS.length) {
      setActiveRegion(REGION_DEFS[idx + 1].id);
    }
  };

  // ドラッグ中にキャンバス外で離した場合に取り残されないよう
  // window レベルで mouseup を拾う。state を握っているのは React 側なので
  // useEffect で listener を登録する。
  useEffect(() => {
    if (!drag) return;
    const onWindowMouseUp = () => {
      if (!canvasRef.current) {
        setDrag(null);
        return;
      }
      const rect = canvasRef.current.getBoundingClientRect();
      const ratio = rectToRatio(drag.start, drag.current, rect.width, rect.height);
      setDrag(null);
      if (!ratio) return;
      setCalibration((prev) => setRegionRect(prev, drag.region, ratio));
      const idx = REGION_DEFS.findIndex((r) => r.id === drag.region);
      if (idx >= 0 && idx + 1 < REGION_DEFS.length) {
        setActiveRegion(REGION_DEFS[idx + 1].id);
      }
    };
    window.addEventListener("mouseup", onWindowMouseUp);
    return () => window.removeEventListener("mouseup", onWindowMouseUp);
  }, [drag]);

  const handleClear = (region: RoiRegionId) => {
    setCalibration((prev) => setRegionRect(prev, region, null));
  };

  const handleSave = () => {
    onSaved({ ...settings, roi_calibration: calibration });
  };

  const handleResetAll = () => {
    setCalibration(EMPTY_ROI_CALIBRATION);
  };

  const completedCount = useMemo(
    () => REGION_DEFS.filter((r) => getRegionRect(calibration, r.id) !== null).length,
    [calibration],
  );

  // CSS px の表示用矩形を比率から組み立てる。canvasRef の現在サイズが必要なため
  // useMemo ではなく描画時に都度計算する。
  const renderRect = (region: RoiRegionId, rectRatio: RoiRect, isActive: boolean) => (
    <div
      key={region}
      className={cn(
        "pointer-events-none absolute font-mono text-[10px] font-bold uppercase tracking-wide",
        isActive ? "ring-2 ring-acial-blue" : "ring-1 ring-acial-red/70",
      )}
      style={{
        left: `${rectRatio.x * 100}%`,
        top: `${rectRatio.y * 100}%`,
        width: `${rectRatio.w * 100}%`,
        height: `${rectRatio.h * 100}%`,
        background: isActive ? "rgba(4,50,255,0.16)" : "rgba(255,38,0,0.12)",
        color: isActive ? "var(--color-acial-blue)" : "var(--color-acial-red)",
      }}
    >
      <span
        className="absolute -top-4 left-0 rounded-sm bg-white/95 px-1.5 py-0.5 shadow"
        style={{ color: isActive ? "var(--color-acial-blue)" : "var(--color-acial-red)" }}
      >
        {regionLabel(region)}
      </span>
    </div>
  );

  const dragOverlay = drag
    ? (() => {
        const rect = canvasRef.current?.getBoundingClientRect();
        if (!rect) return null;
        const left = Math.min(drag.start.x, drag.current.x);
        const top = Math.min(drag.start.y, drag.current.y);
        const width = Math.abs(drag.current.x - drag.start.x);
        const height = Math.abs(drag.current.y - drag.start.y);
        return (
          <div
            className="pointer-events-none absolute border-2 border-dashed border-acial-blue"
            style={{
              left,
              top,
              width,
              height,
              background: "rgba(4,50,255,0.10)",
            }}
          />
        );
      })()
    : null;

  return (
    <div
      className="mx-auto flex h-full w-full max-w-[480px] flex-col overflow-hidden border border-ink-200 bg-white font-jp"
      style={{
        boxShadow: "0 24px 60px rgba(15,15,30,0.14), 0 4px 12px rgba(15,15,30,0.06)",
      }}
    >
      {/* ヘッダー */}
      <div className="flex items-center gap-2.5 border-b border-ink-200 bg-white px-4 py-3.5">
        <button
          type="button"
          onClick={onBack}
          aria-label="戻る"
          className="flex h-7 w-7 cursor-pointer items-center justify-center rounded-md border border-ink-200 bg-white text-ink-700 transition-colors hover:bg-ink-50"
        >
          <ChevronLeft className="h-3.5 w-3.5" strokeWidth={2} />
        </button>
        <h2 className="m-0 font-jp text-[20px] font-bold leading-tight text-ink-900">
          ROI キャリブレーション
        </h2>
      </div>

      <div className="flex flex-1 flex-col gap-3 overflow-y-auto p-4">
        <div className="flex items-center justify-between gap-2">
          <button
            type="button"
            onClick={requestCapture}
            disabled={capturing}
            className="cursor-pointer rounded-md border border-ink-200 bg-white px-3 py-1.5 font-jp text-[12px] font-semibold text-ink-900 transition-colors hover:bg-ink-50 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {capturing ? "キャプチャ中..." : capture ? "再キャプチャ" : "キャプチャ"}
          </button>
          <div className="font-jp text-[11px] text-ink-500">
            設定済み: {completedCount} / {REGION_DEFS.length}
          </div>
        </div>

        {captureError && (
          <div
            role="alert"
            className="rounded-md px-3 py-2 font-jp text-[12px] font-semibold text-danger"
            style={{
              background: "var(--color-danger-bg)",
              border: "1px solid rgba(199,27,0,0.18)",
            }}
          >
            {captureError}
          </div>
        )}

        {/* 操作対象選択 */}
        <div className="flex flex-wrap gap-1.5">
          {REGION_DEFS.map((r) => {
            const has = getRegionRect(calibration, r.id) !== null;
            const isActive = activeRegion === r.id;
            return (
              <button
                key={r.id}
                type="button"
                onClick={() => setActiveRegion(r.id)}
                className={cn(
                  "cursor-pointer rounded-full border px-3 py-1 font-jp text-[11px] font-semibold transition-colors",
                  isActive
                    ? "border-acial-blue bg-acial-blue text-white"
                    : has
                      ? "border-acial-red/30 bg-red-50 text-red-700"
                      : "border-ink-200 bg-white text-ink-700 hover:bg-ink-50",
                )}
              >
                {r.label}
                {has && !isActive && <span className="ml-1">●</span>}
              </button>
            );
          })}
        </div>

        {/* キャンバス */}
        <div
          ref={canvasRef}
          onMouseDown={onMouseDown}
          onMouseMove={onMouseMove}
          onMouseUp={onMouseUp}
          className="relative aspect-video w-full select-none overflow-hidden rounded-md border border-ink-200 bg-ink-50"
          style={{
            backgroundImage: capture ? `url(${capture.imageDataUrl})` : undefined,
            backgroundSize: "100% 100%",
            backgroundRepeat: "no-repeat",
            cursor: capture ? "crosshair" : "default",
            touchAction: "none",
          }}
          aria-label="キャプチャしたウィンドウ画像。ドラッグで矩形を指定する"
          data-testid="roi-canvas"
        >
          {!capture && (
            <div className="flex h-full w-full items-center justify-center font-jp text-[12px] text-ink-500">
              {capturing ? "キャプチャ中..." : "キャプチャを取得してください"}
            </div>
          )}
          {capture &&
            REGION_DEFS.map((r) => {
              const rect = getRegionRect(calibration, r.id);
              if (!rect) return null;
              return renderRect(r.id, rect, activeRegion === r.id);
            })}
          {dragOverlay}
        </div>

        <div className="font-jp text-[11px] text-ink-500">
          選択中: <span className="font-semibold text-ink-900">{regionLabel(activeRegion)}</span> —
          画像上をドラッグして矩形を指定。確定すると次の領域へ進みます。
        </div>

        {/* 設定済み一覧 */}
        <div className="flex flex-col gap-1.5">
          {REGION_DEFS.map((r) => {
            const rect = getRegionRect(calibration, r.id);
            if (!rect) return null;
            return (
              <div
                key={r.id}
                className="flex items-center justify-between rounded-md border border-ink-200 bg-white px-3 py-2"
              >
                <div className="flex flex-col gap-0.5">
                  <span className="font-jp text-[12px] font-semibold text-ink-900">{r.label}</span>
                  <span className="font-mono text-[10px] text-ink-500">
                    x {rect.x.toFixed(3)} y {rect.y.toFixed(3)} w {rect.w.toFixed(3)} h{" "}
                    {rect.h.toFixed(3)}
                  </span>
                </div>
                <button
                  type="button"
                  onClick={() => handleClear(r.id)}
                  className="cursor-pointer rounded border border-ink-200 bg-white px-2 py-1 font-jp text-[10px] text-ink-700 transition-colors hover:bg-ink-50"
                >
                  クリア
                </button>
              </div>
            );
          })}
        </div>
      </div>

      {/* フッター */}
      <div className="flex gap-2 border-t border-ink-200 bg-white p-3">
        <button
          type="button"
          onClick={handleResetAll}
          className="cursor-pointer rounded-lg border border-ink-200 bg-white px-3 py-3 font-jp text-sm font-semibold text-ink-900 transition-colors hover:bg-ink-50"
        >
          全クリア
        </button>
        <button
          type="button"
          onClick={onBack}
          className="flex-1 cursor-pointer rounded-lg border border-ink-200 bg-white px-3 py-3 font-jp text-sm font-semibold text-ink-900 transition-colors hover:bg-ink-50"
        >
          キャンセル
        </button>
        <button
          type="button"
          onClick={handleSave}
          className="flex-1 cursor-pointer rounded-lg border-0 px-3 py-3 font-jp text-sm font-bold text-white transition-opacity hover:opacity-95"
          style={{ background: "var(--gradient-acial)" }}
        >
          保存
        </button>
      </div>
    </div>
  );
}

function regionLabel(region: RoiRegionId): string {
  const def = REGION_DEFS.find((r) => r.id === region);
  return def ? def.label : region;
}

function clamp01(v: number): number {
  if (Number.isNaN(v)) return 0;
  return Math.max(0, Math.min(1, v));
}

/**
 * canvas 上 (CSS px) の 2 点から比率矩形を組み立てる。
 *
 * 矩形が極端に小さい (= ユーザがクリックしただけ) ケースは無視して、
 * 従来の指定が消えないようにする (= `null` を返してキャンセル扱い)。
 */
function rectToRatio(
  start: { x: number; y: number },
  end: { x: number; y: number },
  width: number,
  height: number,
): RoiRect | null {
  if (width <= 0 || height <= 0) return null;
  const x = Math.min(start.x, end.x) / width;
  const y = Math.min(start.y, end.y) / height;
  const w = Math.abs(end.x - start.x) / width;
  const h = Math.abs(end.y - start.y) / height;
  if (w < 0.005 || h < 0.005) return null;
  return {
    x: clamp01(x),
    y: clamp01(y),
    w: Math.min(1 - clamp01(x), w),
    h: Math.min(1 - clamp01(y), h),
  };
}
