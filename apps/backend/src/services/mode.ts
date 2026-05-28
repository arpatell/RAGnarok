import type { PanelCandidate, ReaderMode } from "../types.js";

const SCROLL_RATIO_THRESHOLD = 1.7;

export function detectReadingMode(
  defaultMode: ReaderMode,
  panelCandidates: PanelCandidate[]
): ReaderMode {
  const dimensioned = panelCandidates.filter(
    (panel) => typeof panel.width === "number" && typeof panel.height === "number" && panel.width > 0
  );

  if (dimensioned.length === 0) {
    return defaultMode;
  }

  const tallCount = dimensioned.filter((panel) => (panel.height as number) / (panel.width as number) >= SCROLL_RATIO_THRESHOLD).length;
  const tallRatio = tallCount / dimensioned.length;

  if (tallRatio >= 0.6) {
    return "scroll";
  }

  if (tallRatio <= 0.3) {
    return "paginated";
  }

  return defaultMode;
}
