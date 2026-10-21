import assert from "node:assert/strict";
import { test } from "node:test";
import { toDisplayTypeLabel } from "../src/lib/mediaLabels";

const cases: Array<[string, string | null, string]> = [
  ["manga", "manga", "Manga"],
  ["manga", "manhwa", "Manhwa"],
  ["manga", "MANHUA", "Manhua"],
  ["manga", "one_shot", "One-Shot"],
  ["manga", "one shot", "One-Shot"],
  ["manga", "oneshot", "One-Shot"],
  ["manga", "light_novel", "Light Novel"],
  ["manga", "novel", "Novel"],
  ["manga", null, "Manga"],
  ["", null, "Manga"],
  ["anime", "tv", "Anime (TV)"],
  ["anime", "TV Special", "Anime (TV Special)"],
  ["anime", "movie", "Anime (Movie)"],
  ["anime", "ova", "Anime (OVA)"],
  ["anime", "ona", "Anime (ONA)"],
  ["anime", "special", "Anime (Special)"],
  ["anime", "music", "Anime (Music)"],
  ["anime", "Anime (tv)", "Anime (TV)"],
  [" Anime ", " anime ", "Anime (TV)"],
  ["anime", null, "Anime (TV)"]
];

for (const [media_type, display_type, expected] of cases) {
  test(`${media_type}/${display_type} displays as ${expected}`, () => {
    assert.equal(toDisplayTypeLabel({ media_type, display_type }), expected);
  });
}
