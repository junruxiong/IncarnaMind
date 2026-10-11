/** Small readers the providers share for what their services answer. */
import { decodeEntities } from "../../documents/formats/xml";
import { SourceError } from "../http";

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
