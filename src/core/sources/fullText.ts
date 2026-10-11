/**
 * An open copy, fetched and kept as the Document's file: a PDF as it came,
 * or text a provider gives by section as Markdown, each section under its
 * heading, so a Citation names the section its quote is in ("§ 3.1 Cell-free
 * methods"; ADR-0011's Locations for Markdown) and the check reads that
 * section's text.
 */
import { BROWSER_USER_AGENT, SourceError, type SourceHttp } from "./http";
import { checkPdf } from "./providers/common";
import type { FullText, TextSection } from "./types";

/** The largest PDF downloaded: far more than a paper, less than a scanned book. */
export const MAX_PDF_BYTES = 100 * 1024 * 1024;

/**
 * An open-access PDF anywhere on the web (a publisher's or a repository's),
 * downloaded once, as adding a web page fetches it (#261): without cookies,
 * within a time and size limit. What comes back must be a PDF: a sign-in
 * page or a paywall is "not-open", and nothing is kept of it.
 */
export async function downloadOpenPdf(
  http: SourceHttp,
  url: string,
  signal: AbortSignal,
): Promise<FullText> {
  const host = URL.parse(url)?.host ?? url;
  const { bytes, url: finalUrl } = await http.get(url, {
    // One download at a time from each host.
    lane: `host:${host}`,
    intervalMs: 1000,
    accept: "application/pdf",
    signal,
    timeoutMs: 60_000,
    userAgent: BROWSER_USER_AGENT,
    maxBytes: MAX_PDF_BYTES,
  });
  checkPdf(bytes, url);
  return { format: "pdf", bytes, url: finalUrl };
}

/** A line Markdown would read as structure (a heading, a fence, a quote) kept as text. */
const escapeLine = (line: string) => line.replace(/^(\s*)(#|>|```|~~~)/, "$1\\$2");

/** A text given by section, as Markdown: the title, then each section under a heading of its level. */
export function sectionsToMarkdown(title: string, sections: readonly TextSection[]): string {
  const lines = [`# ${title.replace(/\s+/g, " ").trim() || "Untitled"}`];
  for (const section of sections) {
    const level = Math.min(6, Math.max(1, section.level) + 1);
    lines.push("", `${"#".repeat(level)} ${section.heading.replace(/\s+/g, " ").trim()}`);
    for (const paragraph of section.paragraphs) {
      lines.push("", escapeLine(paragraph.replace(/\s+/g, " ").trim()));
    }
  }
  return `${lines.join("\n")}\n`;
}

/** What a fetched copy is kept as: its bytes, and the file's extension. */
export function fileOf(text: FullText): { data: Uint8Array; extension: ".pdf" | ".md" } {
  if (text.format === "pdf") return { data: text.bytes, extension: ".pdf" };
  if (text.sections.every((section) => section.paragraphs.length === 0)) {
    throw new SourceError("not-open", "The text has no paragraphs.");
  }
  return {
    data: new TextEncoder().encode(sectionsToMarkdown(text.title, text.sections)),
    extension: ".md",
  };
}
