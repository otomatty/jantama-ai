import { useEffect, useState } from "react";
import { ArrowLeft, FolderOpen } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Select } from "@/components/ui/select";
import type { AppSettings, CaptureWindow } from "@/types";
import { listCaptureWindows, saveSettings } from "@/lib/tauriCommands";

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

  useEffect(() => {
    let cancelled = false;
    listCaptureWindows()
      .then((list) => {
        if (!cancelled) setWindows(list);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const handleRefreshWindows = async () => {
    setLoading(true);
    try {
      setWindows(await listCaptureWindows());
    } finally {
      setLoading(false);
    }
  };

  const handlePickModelFile = async () => {
    // Tauri 環境では tauri-plugin-dialog を呼ぶ
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
      // ブラウザ実行時は手入力フォールバック
      const path = window.prompt("Mortal モデルファイルのパスを入力してください");
      if (path) setSettings((s) => ({ ...s, mortal_model_path: path }));
    }
  };

  const handleSave = async () => {
    setSaving(true);
    try {
      await saveSettings(settings);
      onSaved(settings);
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="flex h-full flex-col">
      <header className="flex items-center justify-between border-b border-[var(--color-border)] px-4 py-3">
        <div className="flex items-center gap-2">
          <Button variant="ghost" size="sm" onClick={onBack}>
            <ArrowLeft />
            戻る
          </Button>
          <h1 className="text-lg font-semibold">設定</h1>
        </div>
        <Button size="sm" onClick={handleSave} disabled={saving}>
          {saving ? "保存中..." : "保存"}
        </Button>
      </header>

      <main className="flex-1 overflow-auto p-6">
        <div className="mx-auto flex max-w-2xl flex-col gap-6">
          {/* キャプチャ対象ウィンドウ */}
          <Card>
            <CardHeader>
              <CardTitle>キャプチャ対象ウィンドウ</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <Label htmlFor="capture-window">
                雀魂のウィンドウを選択してください (Steam版 / ブラウザ版)
              </Label>
              <div className="flex gap-2">
                <Select
                  id="capture-window"
                  value={settings.capture_target_window_id ?? ""}
                  onChange={(e) => {
                    const id = e.target.value || null;
                    const w = windows.find((w) => w.id === id);
                    setSettings((s) => ({
                      ...s,
                      capture_target_window_id: id,
                      capture_target_window_title: w?.title ?? null,
                    }));
                  }}
                  disabled={loading}
                >
                  <option value="">-- 選択してください --</option>
                  {windows.map((w) => (
                    <option key={w.id} value={w.id}>
                      {w.title} {w.app_name ? `[${w.app_name}]` : ""}
                    </option>
                  ))}
                </Select>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handleRefreshWindows}
                  disabled={loading}
                >
                  再取得
                </Button>
              </div>
            </CardContent>
          </Card>

          {/* Mortal モデルパス */}
          <Card>
            <CardHeader>
              <CardTitle>Mortal モデルパス</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <Label htmlFor="mortal-path">
                Mortal の学習済みモデル (.pth / .pt) を選択してください
              </Label>
              <div className="flex gap-2">
                <Input
                  id="mortal-path"
                  value={settings.mortal_model_path ?? ""}
                  placeholder="C:\path\to\model.pth"
                  onChange={(e) =>
                    setSettings((s) => ({
                      ...s,
                      mortal_model_path: e.target.value || null,
                    }))
                  }
                />
                <Button variant="outline" size="sm" onClick={handlePickModelFile}>
                  <FolderOpen />
                  選択
                </Button>
              </div>
            </CardContent>
          </Card>

          {/* データ保存期間 (PRD §4.2 Should、UI のみ MVP で表示) */}
          <Card>
            <CardHeader>
              <CardTitle>データ保存期間</CardTitle>
            </CardHeader>
            <CardContent className="grid grid-cols-3 gap-4">
              <div>
                <Label>推論履歴 (日)</Label>
                <Input
                  type="number"
                  min={1}
                  value={settings.data_retention_days.inference_log}
                  onChange={(e) =>
                    setSettings((s) => ({
                      ...s,
                      data_retention_days: {
                        ...s.data_retention_days,
                        inference_log: Number(e.target.value),
                      },
                    }))
                  }
                />
              </div>
              <div>
                <Label>牌画像 (日)</Label>
                <Input
                  type="number"
                  min={1}
                  value={settings.data_retention_days.tile_image}
                  onChange={(e) =>
                    setSettings((s) => ({
                      ...s,
                      data_retention_days: {
                        ...s.data_retention_days,
                        tile_image: Number(e.target.value),
                      },
                    }))
                  }
                />
              </div>
              <div>
                <Label>エラーログ (日)</Label>
                <Input
                  type="number"
                  min={1}
                  value={settings.data_retention_days.error_log}
                  onChange={(e) =>
                    setSettings((s) => ({
                      ...s,
                      data_retention_days: {
                        ...s.data_retention_days,
                        error_log: Number(e.target.value),
                      },
                    }))
                  }
                />
              </div>
            </CardContent>
          </Card>
        </div>
      </main>
    </div>
  );
}
