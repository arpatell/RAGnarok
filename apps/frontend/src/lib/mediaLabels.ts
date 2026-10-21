import type { RagSearchResult } from "../types";

const TYPE_LABELS: Record<string, string> = {
  tv: "TV",
  tv_special: "TV Special",
  ova: "OVA",
  ona: "ONA",
  pv: "PV",
  cm: "CM",
  one_shot: "One-Shot",
  oneshot: "One-Shot",
  lightnovel: "Light Novel"
};

function formatType(value: string): string {
  const key = value.trim().toLowerCase().replace(/[\s-]+/g, "_");
  return TYPE_LABELS[key] ?? key.split("_").filter(Boolean)
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1)).join(" ");
}

export function toDisplayTypeLabel(result: Pick<RagSearchResult, "media_type" | "display_type">): string {
  const mediaType = result.media_type.trim().toLowerCase();
  const displayType = result.display_type?.trim() ?? "";

  if (mediaType === "anime") {
    const subtype = displayType.replace(/^anime\s*\((.*)\)$/i, "$1");
    return `Anime (${!subtype || subtype.toLowerCase() === "anime" ? "TV" : formatType(subtype)})`;
  }

  return formatType(displayType || mediaType || "manga");
}
