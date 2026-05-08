import { useEffect, useState } from "react";
import { ChevronDown, ChevronLeft } from "lucide-react";
import {
  type AppSettings,
  type CaptureWindow,
  type InferenceBackend,
} from "@/types";
import { listCaptureWindows, saveSettings } from "@/lib/tauriCommands";
import { cn } from "@/lib/utils";

interface SettingsScreenProps {
  initialSettings: AppSettings;
  onBack: () => void;
  onSaved: (next: AppSettings) => void;
}

export function SettingsScreen({
  initialSettings,
  onBack,
  onSaved,
}: SettingsScreenProps) {
  const [settings, setSettings] = useState<AppSettings>(initialSettings);
  const [windows, setWindows] = useState<CaptureWindow[]>([]);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [windowsError, setWindowsError] = useState<string | null>(null);
  const [saveError, setSaveError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    listCaptureWindows()
      .then((list) => {
        if (!cancelled) setWindows(list);
      })
      .catch(() => {
        if (!cancelled) {
          setWindows([]);
          setWindowsError("ウィンドウ一覧の取得に失敗しました");
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const selectedWindow = windows.find(
    (w) => w.id === settings.capture_target_window_id,
  );

  const handlePickWindow = () => {
    // 画面側で簡易的にローテートさせる (デザイン版はネイティブのドロップダウン代替)
    if (windows.length === 0) return;
    const currentIndex = windows.findIndex(
      (w) => w.id === settings.capture_target_window_id,
    );
    // 未選択 (-1) の場合は先頭に。選択済みなら次の要素にローテート。
    const nextIndex =
      currentIndex === -1 ? 0 : (currentIndex + 1) % windows.length;
    const next = windows[nextIndex];
    setSettings((s) => ({
      ...s,
      capture_target_window_id: next.id,
      capture_target_window_title: next.title,
    }));
  };

  const handlePickModelFile = async () => {
    try {
      const { open } = await import("@tauri-apps/plugin-dialog");
      const path = await open({
        multiple: false,
        directory: false,
        filters: [{ name: "Mortal Model", extensions: ["pth", "pt"] }],
      });
      if (typeof path === "string") {
        setSettings((s) => ({ ...s, mortal_model_path: path }));
      }
    } catch {
      const path = window.prompt("Mortal モデルファイルのパスを入力してください");
      if (path) setSettings((s) => ({ ...s, mortal_model_path: path }));
    }
  };

  const handleSave = async () => {
    setSaving(true);
    setSaveError(null);
    // 既存ストア由来や stale な負数/0 を温存しないよう、保存直前に再正規化する。
    const normalized: AppSettings = {
      ...settings,
      data_retention_days: {
        inference_log: clampDay(settings.data_retention_days.inference_log),
        tile_image: clampDay(settings.data_retention_days.tile_image),
        error_log: clampDay(settings.data_retention_days.error_log),
      },
    };
    try {
      await saveSettings(normalized);
      onSaved(normalized);
    } catch {
      setSaveError("設定の保存に失敗しました");
    } finally {
      setSaving(false);
    }
  };

  const modelFileName = filenameOf(settings.mortal_model_path);
  const modelFileDir = dirOf(settings.mortal_model_path);

  return (
    <div
      className="mx-auto flex h-full w-full max-w-[480px] flex-col overflow-hidden border border-ink-200 bg-white font-jp"
      style={{
        boxShadow:
          "0 24px 60px rgba(15,15,30,0.14), 0 4px 12px rgba(15,15,30,0.06)",
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
          設定
        </h2>
      </div>

      {/* 中央 */}
      <div className="flex flex-1 flex-col gap-[18px] overflow-y-auto p-4">
        <SettingGroup label="キャプチャ対象">
          <SelectField
            value={
              loading
                ? "ウィンドウを取得中..."
                : selectedWindow
                  ? selectedWindow.title
                  : "未選択"
            }
            onClick={handlePickWindow}
          />
          {windowsError ? (
            <Hint tone="danger">{windowsError}</Hint>
          ) : (
            <Hint>起動中のウィンドウから選択</Hint>
          )}
        </SettingGroup>

        <SettingGroup label="Mortal モデル">
          <FileField
            value={modelFileName ?? "未選択"}
            path={modelFileDir ?? ""}
            onPick={handlePickModelFile}
          />
        </SettingGroup>

        <SettingGroup label="推論バックエンド">
          <SegmentedField
            options={["ROCm", "CPU フォールバック"]}
            active={settings.inference_backend === "rocm" ? 0 : 1}
            onChange={(i) =>
              setSettings((s) => ({
                ...s,
                inference_backend: (i === 0 ? "rocm" : "cpu") as InferenceBackend,
              }))
            }
          />
          <Hint>ROCm 7.2.1 (Radeon 860M) ・ public preview</Hint>
        </SettingGroup>

        <SettingGroup label="LLM 推奨理由">
          <ToggleField
            on={settings.show_llm_reason}
            ariaLabel="LLM 推奨理由"
            onChange={(v) => setSettings((s) => ({ ...s, show_llm_reason: v }))}
          />
          <Hint>S-01 — Claude Haiku で打牌理由を生成</Hint>
        </SettingGroup>

        <SettingGroup label="危険牌・安全牌の表示">
          <ToggleField
            on={settings.show_danger_safe}
            ariaLabel="危険牌・安全牌の表示"
            onChange={(v) =>
              setSettings((s) => ({ ...s, show_danger_safe: v }))
            }
          />
          <Hint>S-02 — 他家リーチ時に常時表示</Hint>
        </SettingGroup>

        <SettingGroup label="データ保存期間">
          <RetentionRow
            label="推論履歴"
            days={settings.data_retention_days.inference_log}
            onChange={(v) =>
              setSettings((s) => ({
                ...s,
                data_retention_days: { ...s.data_retention_days, inference_log: v },
              }))
            }
          />
          <RetentionRow
            label="牌画像サンプル"
            days={settings.data_retention_days.tile_image}
            onChange={(v) =>
              setSettings((s) => ({
                ...s,
                data_retention_days: { ...s.data_retention_days, tile_image: v },
              }))
            }
          />
          <RetentionRow
            label="エラーログ"
            days={settings.data_retention_days.error_log}
            onChange={(v) =>
              setSettings((s) => ({
                ...s,
                data_retention_days: { ...s.data_retention_days, error_log: v },
              }))
            }
          />
        </SettingGroup>
      </div>

      {/* フッター */}
      <div className="border-t border-ink-200 bg-white p-3">
        {saveError && (
          <div
            role="alert"
            className="mb-2 rounded-md px-3 py-2 font-jp text-[12px] font-semibold text-danger"
            style={{
              background: "var(--color-danger-bg)",
              border: "1px solid rgba(199,27,0,0.18)",
            }}
          >
            {saveError}
          </div>
        )}
        <div className="flex gap-2">
          <button
            type="button"
            onClick={onBack}
            disabled={saving}
            className="flex-1 cursor-pointer rounded-lg border border-ink-200 bg-white px-3 py-3 font-jp text-sm font-semibold text-ink-900 transition-colors hover:bg-ink-50 disabled:cursor-not-allowed disabled:opacity-50"
          >
            キャンセル
          </button>
          <button
            type="button"
            onClick={handleSave}
            disabled={saving}
            className="flex-1 cursor-pointer rounded-lg border-0 px-3 py-3 font-jp text-sm font-bold text-white transition-opacity hover:opacity-95 disabled:cursor-not-allowed disabled:opacity-50"
            style={{ background: "var(--gradient-acial)" }}
          >
            {saving ? "保存中..." : "保存"}
          </button>
        </div>
      </div>
    </div>
  );
}

function SettingGroup({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <div>
      <div className="mb-1.5 font-sans text-[12px] font-bold uppercase tracking-[0.18em] text-ink-500">
        {label}
      </div>
      <div className="flex flex-col gap-1.5">{children}</div>
    </div>
  );
}

function SelectField({
  value,
  onClick,
}: {
  value: string;
  onClick?: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="flex w-full cursor-pointer items-center justify-between rounded-md border border-ink-200 bg-white px-3 py-2.5 text-left font-jp text-[13px] text-ink-900 transition-colors hover:bg-ink-50"
    >
      <span className="truncate">{value}</span>
      <ChevronDown className="h-3.5 w-3.5 shrink-0 text-ink-500" strokeWidth={2} />
    </button>
  );
}

function FileField({
  value,
  path,
  onPick,
}: {
  value: string;
  path: string;
  onPick: () => void;
}) {
  return (
    <div className="flex items-center justify-between gap-3 rounded-md border border-ink-200 bg-white px-3 py-2.5">
      <div className="min-w-0 flex-1">
        <div className="truncate font-mono text-xs text-ink-900">{value}</div>
        {path && (
          <div className="truncate font-mono text-[10px] text-ink-500">{path}</div>
        )}
      </div>
      <button
        type="button"
        onClick={onPick}
        className="shrink-0 cursor-pointer rounded border border-ink-200 bg-white px-2.5 py-1 font-jp text-[11px] text-ink-900 transition-colors hover:bg-ink-50"
      >
        変更
      </button>
    </div>
  );
}

function SegmentedField({
  options,
  active,
  onChange,
}: {
  options: string[];
  active: number;
  onChange: (index: number) => void;
}) {
  return (
    <div className="inline-flex gap-0.5 rounded-md bg-ink-100 p-[3px]">
      {options.map((o, i) => (
        <button
          key={o}
          type="button"
          onClick={() => onChange(i)}
          className={cn(
            "cursor-pointer rounded border-0 px-3 py-1.5 font-jp text-[12px] font-semibold transition-colors",
            i === active
              ? "bg-white text-ink-900"
              : "bg-transparent text-ink-500",
          )}
          style={i === active ? { boxShadow: "var(--shadow-1)" } : undefined}
        >
          {o}
        </button>
      ))}
    </div>
  );
}

function ToggleField({
  on,
  ariaLabel,
  onChange,
}: {
  on: boolean;
  ariaLabel: string;
  onChange: (next: boolean) => void;
}) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={on}
      aria-label={ariaLabel}
      onClick={() => onChange(!on)}
      className={cn(
        "inline-flex h-[22px] w-[38px] cursor-pointer items-center rounded-full border-0 p-0.5 transition-colors",
        on ? "bg-ink-900" : "bg-ink-200",
      )}
    >
      <span
        className="block h-[18px] w-[18px] rounded-full bg-white transition-transform"
        style={{
          transform: on ? "translateX(16px)" : "translateX(0)",
          boxShadow: "0 1px 2px rgba(0,0,0,0.2)",
        }}
      />
    </button>
  );
}

function Hint({
  children,
  tone = "muted",
}: {
  children: React.ReactNode;
  tone?: "muted" | "danger";
}) {
  return (
    <div
      className={cn(
        "font-jp text-[11px]",
        tone === "danger" ? "text-danger" : "text-ink-500",
      )}
    >
      {children}
    </div>
  );
}

function clampDay(value: number): number {
  return Math.max(1, Number.isFinite(value) ? value : 1);
}

function RetentionRow({
  label,
  days,
  onChange,
}: {
  label: string;
  days: number;
  onChange: (next: number) => void;
}) {
  return (
    <div className="flex items-center justify-between rounded-md border border-ink-200 bg-white px-3 py-2">
      <span className="font-jp text-[13px] text-ink-900">{label}</span>
      <div className="flex items-center gap-2">
        <input
          type="number"
          min={1}
          value={days}
          onChange={(e) =>
            onChange(Math.max(1, Number(e.target.value) || 1))
          }
          className="w-16 rounded border border-ink-200 bg-white px-2 py-0.5 text-right font-mono text-xs text-ink-700"
        />
        <span className="font-mono text-xs text-ink-500">日</span>
      </div>
    </div>
  );
}

function filenameOf(path: string | null): string | null {
  if (!path) return null;
  const m = path.match(/[^\\/]+$/);
  return m ? m[0] : path;
}

function dirOf(path: string | null): string | null {
  if (!path) return null;
  const m = path.match(/^(.*[\\/])/);
  return m ? m[0] : "";
}
