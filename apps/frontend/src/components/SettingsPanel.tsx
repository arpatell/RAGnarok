import { useMemo, type RefObject } from "react";
import {
  THEME_PRESETS,
  type ReaderMode,
  type ReaderSettings,
  type ThemePresetId
} from "../types";

interface SettingsPanelProps {
  isOpen: boolean;
  mode: ReaderMode;
  settings: ReaderSettings;
  panelRef?: RefObject<HTMLElement>;
  onClose: () => void;
  onChangeSettings: (next: ReaderSettings) => void;
}

function isHexColor(value: string): boolean {
  return /^#[0-9A-Fa-f]{6}$/.test(value);
}

function normalizeHexColor(value: string, fallback: string): string {
  if (isHexColor(value)) {
    return value;
  }

  const cleaned = value.trim();
  if (/^#[0-9A-Fa-f]{3}$/.test(cleaned)) {
    const expanded = cleaned
      .slice(1)
      .split("")
      .map((part) => `${part}${part}`)
      .join("");
    return `#${expanded}`;
  }

  return fallback;
}

function detectPresetId(settings: ReaderSettings): ThemePresetId {
  const matched = THEME_PRESETS.find(
    (preset) =>
      preset.interfaceColor.toLowerCase() === settings.interfaceColor.toLowerCase() &&
      preset.textColor.toLowerCase() === settings.textColor.toLowerCase() &&
      preset.accentColor.toLowerCase() === settings.accentColor.toLowerCase() &&
      preset.readerBackground.toLowerCase() === settings.backgroundColor.toLowerCase()
  );

  return matched?.id ?? settings.readerThemePreset;
}

export function SettingsPanel({ isOpen, mode, settings, panelRef, onClose, onChangeSettings }: SettingsPanelProps) {

  const activePreset = useMemo(() => detectPresetId(settings), [settings]);

  function update<K extends keyof ReaderSettings>(key: K, value: ReaderSettings[K]) {
    onChangeSettings({
      ...settings,
      [key]: value
    });
  }

  function applyThemePreset(presetId: ThemePresetId) {
    const preset = THEME_PRESETS.find((item) => item.id === presetId);
    if (!preset) {
      return;
    }

    onChangeSettings({
      ...settings,
      readerThemePreset: preset.id,
      interfaceColor: preset.interfaceColor,
      textColor: preset.textColor,
      accentColor: preset.accentColor,
      backgroundColor: preset.readerBackground
    });
  }

  function updateThemeColor(
    key: "interfaceColor" | "textColor" | "accentColor" | "backgroundColor",
    value: string
  ) {
    const fallback = settings[key];
    const normalized = normalizeHexColor(value, fallback);

    onChangeSettings({
      ...settings,
      [key]: normalized,
      readerThemePreset: detectPresetId({
        ...settings,
        [key]: normalized
      })
    });
  }

  return (
    <aside ref={panelRef} className={`settings-panel ${isOpen ? "open" : ""}`} aria-hidden={!isOpen}>
      <div className="settings-header">
        <h3>Reader Settings</h3>
        <button type="button" className="ghost" onClick={onClose}>
          Close
        </button>
      </div>

      <label>
        Reader theme
        <select
          value={activePreset}
          onChange={(event) => applyThemePreset(event.target.value as ThemePresetId)}
        >
          {THEME_PRESETS.map((preset) => (
            <option key={preset.id} value={preset.id}>
              {preset.label}
            </option>
          ))}
        </select>
      </label>

      <div className="theme-editor-grid" role="group" aria-label="Theme colors">
        <label>
          Interface color
          <input
            type="color"
            value={normalizeHexColor(settings.interfaceColor, "#323840")}
            onChange={(event) => updateThemeColor("interfaceColor", event.target.value)}
          />
        </label>

        <label>
          Text color
          <input
            type="color"
            value={normalizeHexColor(settings.textColor, "#e6ebf0")}
            onChange={(event) => updateThemeColor("textColor", event.target.value)}
          />
        </label>

        <label>
          Accent color
          <input
            type="color"
            value={normalizeHexColor(settings.accentColor, "#607387")}
            onChange={(event) => updateThemeColor("accentColor", event.target.value)}
          />
        </label>

        <label>
          Reader background
          <input
            type="color"
            value={normalizeHexColor(settings.backgroundColor, "#2a3037")}
            onChange={(event) => updateThemeColor("backgroundColor", event.target.value)}
          />
        </label>
      </div>

      <label>
        Fit mode
        <select
          value={settings.fitMode}
          onChange={(event) => update("fitMode", event.target.value as ReaderSettings["fitMode"])}
        >
          <option value="width-fit">Width fit</option>
          <option value="height-fit">Height fit</option>
          <option value="original">Original size</option>
          <option value="custom">Custom zoom</option>
        </select>
      </label>

      <label>
        Zoom percentage
        <input
          type="range"
          min={50}
          max={200}
          step={5}
          value={settings.zoomPercent}
          onChange={(event) => update("zoomPercent", Number(event.target.value))}
        />
        <span>{settings.zoomPercent}%</span>
      </label>

      {mode === "paginated" ? (
        <>
          <label>
            Reading direction
            <select
              value={settings.readingDirection}
              onChange={(event) =>
                update("readingDirection", event.target.value as ReaderSettings["readingDirection"])
              }
            >
              <option value="rtl">Right to left</option>
              <option value="ltr">Left to right</option>
            </select>
          </label>

          <label>
            Spread mode
            <select
              value={settings.spreadMode}
              onChange={(event) => update("spreadMode", event.target.value as ReaderSettings["spreadMode"])}
            >
              <option value="single">Single page</option>
              <option value="double">Double spread</option>
            </select>
          </label>
        </>
      ) : null}

      <label>
        Transition style
        <select
          value={settings.transitionStyle}
          onChange={(event) =>
            update("transitionStyle", event.target.value as ReaderSettings["transitionStyle"])
          }
        >
          <option value="none">None</option>
          <option value="fade">Fade</option>
          <option value="slide">Slide</option>
        </select>
      </label>

      <label>
        Page preload
        <input
          type="range"
          min={1}
          max={9}
          step={1}
          value={settings.preloadDepth}
          onChange={(event) => update("preloadDepth", Number(event.target.value))}
        />
        <span>{settings.preloadDepth}</span>
      </label>

      <label>
        Vertical scroll speed using keyboard arrows
        <input
          type="range"
          min={5}
          max={50}
          step={5}
          value={settings.verticalArrowStepPx}
          onChange={(event) => update("verticalArrowStepPx", Number(event.target.value))}
        />
        <span>{settings.verticalArrowStepPx}px</span>
      </label>

      <label>
        Panel gap
        <input
          type="range"
          min={0}
          max={32}
          step={1}
          value={settings.panelGap}
          onChange={(event) => update("panelGap", Number(event.target.value))}
        />
        <span>{settings.panelGap}px</span>
      </label>

      <label className="checkbox-row">
        <input
          type="checkbox"
          checked={settings.resetPageScrollAfterFlip}
          onChange={(event) => update("resetPageScrollAfterFlip", event.target.checked)}
        />
        Reset page scroll after page flip
      </label>

      <label className="checkbox-row">
        <input
          type="checkbox"
          checked={settings.turnPagesByClicking}
          onChange={(event) => update("turnPagesByClicking", event.target.checked)}
        />
        Turn pages by clicking
      </label>

      <label className="checkbox-row">
        <input
          type="checkbox"
          checked={settings.turnPagesWithArrowsInVerticalView}
          onChange={(event) => update("turnPagesWithArrowsInVerticalView", event.target.checked)}
        />
        Turn pages with arrow keys in vertical view
      </label>

      <label className="checkbox-row">
        <input
          type="checkbox"
          checked={settings.enableSwipeGestures}
          onChange={(event) => update("enableSwipeGestures", event.target.checked)}
        />
        Enable swipe gestures
      </label>

      <label className="checkbox-row">
        <input
          type="checkbox"
          checked={settings.pinVerticalPageSelector}
          onChange={(event) => update("pinVerticalPageSelector", event.target.checked)}
        />
        Pin vertical page selector on left
      </label>

      <label>
        Browser history/back button behavior
        <select
          value={settings.browserHistoryBehavior}
          onChange={(event) =>
            update("browserHistoryBehavior", event.target.value as ReaderSettings["browserHistoryBehavior"])
          }
        >
          <option value="dont-touch">Don&apos;t touch browser history</option>
          <option value="title-only">Just change page title</option>
          <option value="chapter-history">Add every chapter to history</option>
          <option value="chapter-and-pages">Add every chapter and page skips</option>
          <option value="all-moves">Add every move to history</option>
        </select>
      </label>

      <label className="checkbox-row">
        <input
          type="checkbox"
          checked={settings.saveProgress}
          onChange={(event) => update("saveProgress", event.target.checked)}
        />
        Save reading progress
      </label>

      <label className="checkbox-row">
        <input
          type="checkbox"
          checked={settings.highContrast}
          onChange={(event) => update("highContrast", event.target.checked)}
        />
        High contrast mode
      </label>
    </aside>
  );
}
