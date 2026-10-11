/**
 * The licence an open copy is under, from how providers write it: OpenAlex's
 * ids ("cc-by-nc-nd", "public-domain"), or a licence's address in a JATS
 * article ("https://creativecommons.org/licenses/by-nc/4.0/"). Only whether a
 * licence rules out commercial use is read from it; the result says so.
 */
import type { SourceLicence } from "../api";

const CC_PARTS = new Set(["by", "nc", "nd", "sa"]);

/** "cc-by-nc-nd" from a Creative Commons address, or null for any other. */
function ccFromAddress(text: string): string | null {
  const match =
    /creativecommons\.org\/(?:licenses|licences)\/([a-z-]+)\//i.exec(text) ??
    /creativecommons\.org\/publicdomain\/(zero|mark)\//i.exec(text);
  if (!match?.[1]) return null;
  const kind = match[1].toLowerCase();
  if (kind === "zero") return "cc0";
  if (kind === "mark") return "public-domain";
  return kind.split("-").every((part) => CC_PARTS.has(part)) ? `cc-${kind}` : null;
}

/** A licence as a provider writes it (an id or an address); null for none. */
export function licenceOf(written: string | null | undefined): SourceLicence | null {
  const text = written?.trim() ?? "";
  if (text === "") return null;
  const address = ccFromAddress(text);
  // Anything but an id or a licence's address (a sentence about reuse) names no licence.
  if (!address && /\s/.test(text)) return null;
  const id = (address ?? text).toLowerCase();
  if (id === "cc0" || id === "public-domain") {
    return { id, label: id === "cc0" ? "CC0" : id, commercial: true };
  }
  const cc = /^cc-((?:by|nc|nd|sa)(?:-(?:by|nc|nd|sa))*)$/.exec(id);
  if (cc?.[1]) {
    const parts = cc[1].split("-");
    return {
      id,
      label: `CC ${parts.map((part) => part.toUpperCase()).join("-")}`,
      commercial: !parts.includes("nc"),
    };
  }
  // "other-oa", "publisher-specific-oa" and the like: open to read, terms unknown.
  return { id, label: id, commercial: null };
}
