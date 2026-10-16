/**
 * Court decisions a sentence cites: a case name with "v." and a reporter
 * citation ("Brown v. Board of Education, 347 U.S. 483 (1954)"), a neutral
 * citation of the UK and the Commonwealth ("[2024] UKSC 5") or a law-report
 * citation ("[2019] 1 WLR 1234"). A "v." with no citation, or a number that
 * looks like none, doesn't count.
 */

export interface CaseCitation {
  /** "Brown v. Board of Education"; null when the citation stands without one. */
  name: string | null;
  /** "347 U.S. 483", "[2024] UKSC 5". */
  cite: string;
}

const REPORTERS = [
  "U.S.",
  "S. Ct.",
  "L. Ed.",
  "L. Ed. 2d",
  "F.",
  "F.2d",
  "F.3d",
  "F.4th",
  "F. Supp.",
  "F. Supp. 2d",
  "F. Supp. 3d",
  "F. App'x",
  "F.R.D.",
  "B.R.",
  "Fed. Cl.",
  "A.",
  "A.2d",
  "A.3d",
  "N.E.",
  "N.E.2d",
  "N.E.3d",
  "N.W.",
  "N.W.2d",
  "N.Y.",
  "N.Y.2d",
  "N.Y.3d",
  "N.Y.S.",
  "N.Y.S.2d",
  "N.Y.S.3d",
  "P.",
  "P.2d",
  "P.3d",
  "S.E.",
  "S.E.2d",
  "S.W.",
  "S.W.2d",
  "S.W.3d",
  "So.",
  "So. 2d",
  "So. 3d",
  "Cal.",
  "Cal. 2d",
  "Cal. 3d",
  "Cal. 4th",
  "Cal. 5th",
  "Cal. App.",
  "Cal. App. 2d",
  "Cal. App. 3d",
  "Cal. App. 4th",
  "Cal. App. 5th",
  "Cal. Rptr.",
  "Cal. Rptr. 2d",
  "Cal. Rptr. 3d",
  "Ill.",
  "Ill. 2d",
  "Ill. App. 3d",
  "Mass.",
  "Wash. 2d",
];

const NEUTRAL_COURTS = [
  "UKSC",
  "UKPC",
  "UKHL",
  "EWCA Civ",
  "EWCA Crim",
  "EWHC",
  "UKUT",
  "UKFTT",
  "UKEAT",
  "CSIH",
  "CSOH",
  "NICA",
  "NIQB",
  "IESC",
  "IECA",
  "IEHC",
  "HCA",
  "FCA",
  "FCAFC",
  "NSWSC",
  "NSWCA",
  "NSWCCA",
  "VSC",
  "VSCA",
  "QSC",
  "QCA",
  "WASC",
  "SASC",
  "NZSC",
  "NZCA",
  "NZHC",
  "SGCA",
  "SGHC",
  "HKCFA",
  "HKCA",
  "HKCFI",
];

const LAW_REPORTS = [
  "WLR",
  "AC",
  "QB",
  "KB",
  "Ch",
  "Fam",
  "All ER",
  "Lloyd's Rep",
  "Cr App R",
  "BCLC",
  "ICR",
  "IRLR",
  "FSR",
  "RPC",
  "SLT",
  "NZLR",
];

const escaped = (text: string) =>
  text
    .replace(/[.*+?^${}()|[\]\\]/g, "\\$&")
    .replace(/ /g, "\\s+")
    .replace(/'/g, "['’]");

const alternatives = (names: readonly string[]) =>
  [...names]
    .sort((a, b) => b.length - a.length)
    .map(escaped)
    .join("|");

const CAPITALISED = String.raw`[A-Z][\p{L}\p{N}.'’&-]*`;
const COMPANY_SUFFIX = String.raw`(?:,\s+(?:Inc|Ltd|LLC|LLP|Co|Corp|Jr|Sr)\.?)?`;
const PARTY = `${CAPITALISED}(?:\\s+(?:${CAPITALISED}|of|the|and|for|ex|rel\\.|de|la|&)){0,6}${COMPANY_SUFFIX}`;
const NAME = `${PARTY}\\s+v\\.?\\s+${PARTY}`;
const SIGNAL = /^(?:(?:see|also|but|cf|compare|accord|contra|e\.g)\b[.,]?\s+)+/i;

const US_CITE = String.raw`\d{1,4}\s+(?:${alternatives(REPORTERS)})\s+\d{1,5}`;
const NEUTRAL_CITE = String.raw`\[(?:1[89]|20)\d{2}\]\s+(?:${alternatives(NEUTRAL_COURTS)})\s+\d{1,5}(?:\s*\([A-Za-z]{1,8}\))?`;
const REPORT_CITE = String.raw`\[(?:1[89]|20)\d{2}\]\s+(?:\d{1,2}\s+)?(?:${alternatives(LAW_REPORTS)})\s+\d{1,5}`;

const AMERICAN = new RegExp(`(${NAME}),?\\s+(${US_CITE})(?![\\p{L}\\p{N}])`, "gu");
const COMMONWEALTH = new RegExp(`${NEUTRAL_CITE}|${REPORT_CITE}`, "gu");
const NAME_BEFORE = new RegExp(`(${NAME}),?\\s*$`, "u");

const spaced = (text: string) => text.replace(/\s+/g, " ").trim();
const bare = (name: string) => name.replace(SIGNAL, "");

/** The court decisions a sentence cites, in the order it cites them. */
export function caseCitations(sentence: string): CaseCitation[] {
  const text = spaced(sentence);
  const found: (CaseCitation & { at: number })[] = [];
  for (const match of text.matchAll(AMERICAN)) {
    const [, name, cite] = match;
    if (name !== undefined && cite !== undefined) {
      found.push({ at: match.index, name: bare(name), cite });
    }
  }
  for (const match of text.matchAll(COMMONWEALTH)) {
    const before = NAME_BEFORE.exec(text.slice(0, match.index))?.[1];
    found.push({ at: match.index, name: before === undefined ? null : bare(before), cite: match[0] });
  }
  return found.sort((a, b) => a.at - b.at).map(({ name, cite }) => ({ name, cite }));
}

/** Words only, lower case: "Hallmark Builders, Inc. v. Rio Grande" reads "hallmark builders inc v rio grande". */
const plain = (text: string) =>
  text
    .toLowerCase()
    .replace(/[^\p{L}\p{N}]+/gu, " ")
    .trim();

/** Whether a text, such as a Document's name or a Passage of it, names the decision: its citation, or both of its parties. */
export function namesDecision(text: string, decision: CaseCitation): boolean {
  const words = ` ${plain(text)} `;
  if (words.includes(` ${plain(decision.cite)} `)) return true;
  const parties = decision.name?.split(/\s+v\.?\s+/) ?? [];
  return parties.length === 2 && parties.every((party) => words.includes(` ${plain(party)} `));
}
