/**
 * The cases `npm run eval:scope` checks (README.md): a Mind that summarises
 * the employment agreement of ./cases, in a Library of the agreement and
 * other invented Documents, either filed into the Folder that holds the
 * agreement or in none. The placement and the number of other Documents are a
 * case's parameters; the Documents and the sentences never change, so a case
 * can be run again unchanged.
 */
import { writeFile } from "node:fs/promises";
import { join } from "node:path";
import { buildPdf } from "../../tests/helpers/pdf";
import type { EvalDocument } from "../lib/evaluationSet";
import { DOCUMENTS, type RepeatDocument, type RepeatSentence, SETS } from "./cases";
import { OTHER_DOCUMENTS } from "./scopeDocuments";

export type Placement = "unfiled" | "filed";

export interface ScopeCase {
  id: string;
  placement: Placement;
  /** How many Documents besides the agreement the Library holds, the first of OTHER_DOCUMENTS. */
  others: number;
}

export interface ScopeSentence extends RepeatSentence {
  /** What the agreement says that settles the sentence; null for a sentence it says nothing of. */
  clause: string | null;
}

export const OTHER_COUNTS = [5, 20] as const;

export const SCOPE_CASES: readonly ScopeCase[] = OTHER_COUNTS.flatMap((others) =>
  (["unfiled", "filed"] as const).map((placement) => ({
    id: `${placement}-${others}`,
    placement,
    others,
  })),
);

export const MIND = "Summary of the Okoro agreement";

export const FOLDER = "Okoro agreement";

/** The agreement's clauses as they read in its Passage, by the key of the sentence they settle. */
const CLAUSES: Readonly<Record<string, string | null>> = {
  notice: "either side must give three months' written notice",
  "own-words": "may work from home two days a week",
  car: null,
  holiday: "entitled to 27 working days of paid holiday",
};

function agreement(): RepeatDocument {
  const found = DOCUMENTS.find((each) => each.language === "en");
  if (!found) throw new Error("The English agreement is missing from the repeats' Documents.");
  return found;
}

export const AGREEMENT = agreement();

/** Four sentences of the `en` set, worded as there: two the agreement states, two it doesn't. */
export const SCOPE_SENTENCES: readonly ScopeSentence[] = Object.entries(CLAUSES).map(
  ([key, clause]) => {
    const sentence = SETS.find((each) => each.id === "en")?.sentences.find(
      (each) => each.key === key,
    );
    if (!sentence) throw new Error(`The \`en\` set has no sentence "${key}".`);
    return { ...sentence, clause };
  },
);

/** Writes a case's Documents into `folder`, the agreement first, as the files a User adds. */
export async function writeScopeDocuments(
  folder: string,
  scopeCase: ScopeCase,
): Promise<EvalDocument[]> {
  const documents = [AGREEMENT, ...OTHER_DOCUMENTS.slice(0, scopeCase.others)];
  const written: EvalDocument[] = [];
  for (const document of documents) {
    const path = join(folder, document.file);
    await writeFile(path, buildPdf(document.pages));
    written.push({ key: document.file, path });
  }
  return written;
}
