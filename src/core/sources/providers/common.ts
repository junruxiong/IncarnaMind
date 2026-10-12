/** Small readers the providers share for what their services answer. */
import { decodeEntities } from "../../documents/formats/xml";
import { SourceError } from "../http";
import type { TextSection } from "../types";

/** A value as text: "" for anything that isn't text. */
export const text = (value: unknown): string => (typeof value === "string" ? value.trim() : "");

/** A whole number of 0 or more, else null. */
export const optionalNumber = (value: unknown): number | null =>
  typeof value === "number" && Number.isInteger(value) && value >= 0 ? value : null;

/**
 * A title as a citation shows it: without the markup some services keep in
 * it (<i>, <sup>), on one line, and without the full stop PubMed ends each with.
 */
export function plainTitle(title: string): string {
  const plain = decodeEntities(title.replace(/<[^>]+>/g, ""))
    .replace(/\s+/g, " ")
    .trim();
  return plain.replace(/(?<![.!?]\w?)\.$/, "").trim();
}

/** A PDF file starts with "%PDF-"; a sign-in or paywall page sent instead doesn't. */
export function checkPdf(bytes: Uint8Array, url: string): void {
  const head = new TextDecoder("latin1").decode(bytes.subarray(0, 1024));
  if (!head.includes("%PDF-")) {
    throw new SourceError("not-open", `${url} didn't give a PDF.`);
  }
}

/** The JSON a service answered, as an object; throws `SourceError` for anything else. */
export function jsonObject(body: string, service: string): Record<string, unknown> {
  let answer: unknown;
  try {
    answer = JSON.parse(body);
  } catch {
    throw new SourceError("service-error", `${service}'s answer isn't JSON.`);
  }
  if (typeof answer !== "object" || answer === null || Array.isArray(answer)) {
    throw new SourceError("service-error", `${service}'s answer is empty.`);
  }
  return answer as Record<string, unknown>;
}

/** A name some services keep in capitals ("SITAGLIPTIN AND METFORMIN") as a sentence; mixed case is left as written. */
export function sentenceCase(name: string): string {
  const plain = name.replace(/\s+/g, " ").trim();
  if (plain === "" || /[a-z]/.test(plain)) return plain;
  const lower = plain.toLowerCase();
  return `${lower.slice(0, 1).toUpperCase()}${lower.slice(1)}`;
}

/** A company's name some services keep in capitals ("ZYDUS LIFESCIENCES LIMITED") with each word capitalised; short words (LLC, INC) stay. */
export function companyName(name: string): string {
  const plain = name.replace(/\s+/g, " ").trim();
  if (/[a-z]/.test(plain)) return plain;
  return plain
    .split(" ")
    .map((word) =>
      word.replace(/[A-Z]{4,}/g, (long) => `${long.slice(0, 1)}${long.slice(1).toLowerCase()}`),
    )
    .join(" ");
}

/** "20260812" as "2026-08-12"; null for anything else. */
export function compactDate(value: string): string | null {
  const match = /^(\d{4})(\d{2})(\d{2})$/.exec(value);
  return match ? `${match[1]}-${match[2]}-${match[3]}` : null;
}

/** Runs of spaces and line breaks as single spaces. */
export const tidy = (value: string): string => value.replace(/\s+/g, " ").trim();

/** What a text says of itself (where it is from, which version), as its first section. */
export const aboutText = (paragraphs: string[]): TextSection => ({
  heading: "About this text",
  level: 1,
  paragraphs,
});

/**
 * Keeps what the last few loads gave, by key, so the same text isn't fetched
 * twice in a session; a load that fails is forgotten, so it is tried again.
 */
export function remembered<V>(max = 20): (key: string, load: () => Promise<V>) => Promise<V> {
  const kept = new Map<string, Promise<V>>();
  return (key, load) => {
    const known = kept.get(key);
    if (known) return known;
    const loading = load();
    kept.set(key, loading);
    loading.catch(() => kept.delete(key));
    while (kept.size > max) {
      const oldest = kept.keys().next().value;
      if (oldest === undefined) break;
      kept.delete(oldest);
    }
    return loading;
  };
}

/** What a label says of itself, as its first section, with the source's own notice when it has one. */
export function aboutLabel(info: {
  source: string;
  setId: string;
  version: string;
  effective: string | null;
  notice: string | null;
}): TextSection {
  const effective = info.effective ? `, effective ${info.effective}` : "";
  const lines = [`${info.source}. Set ID ${info.setId}, version ${info.version}${effective}.`];
  if (info.notice) lines.push(info.notice);
  return { heading: "About this label", level: 1, paragraphs: lines };
}
