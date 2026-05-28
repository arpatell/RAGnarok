import type { ChapterListItem } from "../types.js";

const CHAPTER_PREFIX = /^(chapter|ch\.?|episode|ep\.?)\s*/i;
const NUMERIC_PATTERN = /-?\d+(?:\.\d+)?/;
const SPECIAL_PATTERN = /(prologue|bonus|extra|special|oneshot|one-shot|pilot|omake|side story)/i;

function parseNumericValue(input: string): number | null {
  const cleaned = input.replace(CHAPTER_PREFIX, "").trim();
  const match = cleaned.match(NUMERIC_PATTERN);
  if (!match) {
    return null;
  }

  const value = Number.parseFloat(match[0]);
  return Number.isFinite(value) ? value : null;
}

function inferSortLabel(item: ChapterListItem): string {
  return (item.number || item.title || "").trim();
}

export function sortChapterList(chapters: ChapterListItem[]): ChapterListItem[] {
  const uniqueByUrl = new Map<string, ChapterListItem>();
  for (const chapter of chapters) {
    if (!chapter.url) {
      continue;
    }
    uniqueByUrl.set(chapter.url, chapter);
  }

  const decorated = Array.from(uniqueByUrl.values()).map((chapter) => {
    const label = inferSortLabel(chapter);
    const parsedNumber = parseNumericValue(label);
    const special = parsedNumber === null || SPECIAL_PATTERN.test(label);

    return {
      chapter: {
        ...chapter,
        special
      },
      label,
      parsedNumber,
      special
    };
  });

  const specials = decorated
    .filter((entry) => entry.special)
    .sort((a, b) => a.label.localeCompare(b.label, undefined, { sensitivity: "base" }))
    .map((entry) => entry.chapter);

  const numerics = decorated
    .filter((entry) => !entry.special)
    .sort((a, b) => {
      if (a.parsedNumber === b.parsedNumber) {
        return a.label.localeCompare(b.label, undefined, { sensitivity: "base" });
      }

      return (a.parsedNumber ?? 0) - (b.parsedNumber ?? 0);
    })
    .map((entry) => entry.chapter);

  return [...numerics, ...specials];
}
