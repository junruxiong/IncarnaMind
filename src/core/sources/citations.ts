/**
 * US federal citations the User types, looked up rather than searched for:
 * a section or a part of the Code of Federal Regulations ("17 CFR
 * 240.10b-5", "17 C.F.R. § 240.10b-5(b)", "17 CFR Part 243"), with a date
 * for the version in force then ("… as of 2019-06-30"), and a section of the
 * United States Code ("42 U.S.C. § 1983"). A paragraph ("(b)") names the
 * section it is in: the section is added, and a Citation points at the
 * paragraph inside it.
 */

export interface CfrCitation {
  title: number;
  part: string;
  /** "240.10b-5"; null for the whole part. */
  section: string | null;
  /** ISO 8601: the version in force on that day; null for the current one. */
  date: string | null;
}

export interface UscCitation {
  title: number;
  /** "1983", "78u-4". */
  section: string;
}

/** "… (2019-06-30)", "… as of 2019-06-30", "… on 2019-06-30", and `cfrId`'s "…@2019-06-30". */
const DATE =
  /^(.*?)[\s,]*(?:\(\s*(\d{4}-\d{2}-\d{2})\s*\)|(?:as\s+of|on|at)\s+(\d{4}-\d{2}-\d{2})|@(\d{4}-\d{2}-\d{2}))$/i;

const CFR =
  /^(?:title\s+)?(\d{1,2})\s*c\.?\s*f\.?\s*r\.?\s*(?:(?:§+|sec(?:tion|s|\.)?)\s*)?(?:(part|pt\.?)\s*)?(\d{1,4}[a-z]?)(?:\.(\d[0-9a-z-]*(?:\.[0-9a-z-]+)*))?(?:\([0-9a-z]{1,6}\))*$/i;

const USC =
  /^(?:title\s+)?(\d{1,2})\s*u\.?\s*s\.?\s*(?:c\.?|code)\s*(?:(?:§+|sec(?:tion|\.)?)\s*)?(\d{1,5}[a-z]*(?:[-–]\d+[a-z]*)*)(?:\([0-9a-z]{1,6}\))*$/i;

/** A real day: "2019-02-30" isn't one. */
function validDate(value: string): boolean {
  const time = Date.parse(`${value}T00:00:00Z`);
  return !Number.isNaN(time) && new Date(time).toISOString().slice(0, 10) === value;
}

export function parseCfr(query: string): CfrCitation | null {
  const text = query.replace(/\s+/g, " ").trim();
  const dated = DATE.exec(text);
  const date = dated?.[2] ?? dated?.[3] ?? dated?.[4] ?? null;
  if (date && !validDate(date)) return null;
  const match = CFR.exec((dated?.[1] ?? text).trim());
  if (!match?.[1] || !match[3]) return null;
  const title = Number(match[1]);
  if (title < 1 || title > 50) return null;
  // "17 CFR Part 240.10" isn't a citation.
  if (match[2] && match[4]) return null;
  return {
    title,
    part: match[3],
    section: match[4] ? `${match[3]}.${match[4]}` : null,
    date,
  };
}

export function parseUsc(query: string): UscCitation | null {
  const match = USC.exec(query.replace(/\s+/g, " ").trim());
  if (!match?.[1] || !match[2]) return null;
  const title = Number(match[1]);
  if (title < 1 || title > 54) return null;
  return { title, section: match[2].replace(/–/g, "-") };
}

/** How a CFR citation is kept: as an identifier's id and a work's key. */
export function cfrId(citation: CfrCitation): string {
  const where = citation.section ?? `Part ${citation.part}`;
  return `${citation.title} CFR ${where}${citation.date ? `@${citation.date}` : ""}`;
}

export const uscId = (citation: UscCitation) => `${citation.title} U.S.C. ${citation.section}`;
