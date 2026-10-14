/**
 * Searching conversations (ADR-0017): their names and messages are indexed as
 * Passages are (../documents/keywords), and searched as ../wordSearch searches.
 */
import { keywordText } from "../documents/keywords";

/**
 * A message's text as plain text, for the index and for excerpts: its Citation
 * markers and the Markdown around words taken out, its lines joined by spaces.
 */
export function plainText(markdown: string): string {
  return markdown
    .replace(/[ \t]?\[\^\d{1,4}\]/g, "")
    .split(/\r?\n/)
    .map((line) =>
      line
        .replace(/^\s*(?:#{1,6}\s+|>\s?|[-*+]\s+|\d+[.)]\s+|```\S*|\$\$)/, "")
        .replace(/!?\[([^\]]*)\]\([^)]*\)/g, "$1")
        .replace(/\*\*|__|~~|`/g, ""),
    )
    .join(" ")
    .replace(/\s+/g, " ")
    .trim();
}

/** What the index stores for a piece of text: its words (see `keywordText`). */
export const indexText = (text: string) => keywordText(text);
