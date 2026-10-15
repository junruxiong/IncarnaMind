/**
 * The guards an everyday request in the assistant must pass (#893, #894),
 * read from what the core stored, as a User would see it:
 *
 * - **Asked to write, the Mind changed.** A request for text in a Mind
 *   ("write a summary page", "make it longer") changed or made one.
 * - **Every claimed change is real.** What the reply says it did to a Mind,
 *   one of its Tool calls did (../../src/core/answers/claimedChanges).
 * - **Citations are there.** With Documents in scope, the reply cites them,
 *   or, for text written into a Mind, the Mind does.
 * - **Every Document is covered.** Asked about all the Documents in scope,
 *   each is cited, or the reply names it as one it didn't cover.
 *
 * The everyday-requests evaluation asserts these for every request, and the
 * unit tests hold them to the failures that were found (tests/eval).
 */
import { changeLogOf, unbackedClaims } from "../../src/core/answers/claimedChanges";
import type { ClaimedChange, ConversationAnswer, MarkedCitation } from "../../src/core/api";

/** What a request expects of its Answer. */
export interface Expected {
  /** It asks for text in a Mind: writing, adding, changing, making one. */
  write: boolean;
  /** Its Answer must cite the Documents in scope. */
  cite: boolean;
  /** It asks about every Document in scope ("summarise these papers"). */
  coverAll: boolean;
}

/** What one request came to. */
export interface RequestOutcome {
  expected: Expected;
  answer: Pick<ConversationAnswer, "text" | "toolCalls" | "steps"> & {
    citations: readonly Pick<MarkedCitation, "documentId">[];
  };
  /** The Documents the Citations it wrote into Minds quote, by id. */
  citedInMinds: readonly string[];
  /** The Documents in the Question's Search scope. */
  scoped: readonly { id: string; name: string }[];
  /** The names of the User's Minds, so a quoted name counts as a Mind's only when it is one. */
  minds: readonly string[];
}

export type RequestFailure =
  /** Asked to write, it changed no Mind. */
  | { kind: "no-change" }
  /** Its reply claims changes no call made; `flagged`: the app said so under it. */
  | { kind: "unbacked"; claims: ClaimedChange[]; flagged: boolean }
  /** Documents were in scope and nothing it wrote cites them. */
  | { kind: "no-citations" }
  /** Asked about every Document in scope, these it neither cited nor named. */
  | { kind: "uncovered"; documents: string[] };

/** A Document's name as a reply may write it: any case, spaces made alike. */
const key = (text: string) => text.replace(/\s+/g, " ").trim().toLocaleLowerCase();

/** The ways one request failed its guards; none when it passed. */
export function requestFailures(outcome: RequestOutcome): RequestFailure[] {
  const { expected, answer } = outcome;
  const failures: RequestFailure[] = [];
  const log = changeLogOf(answer.toolCalls);
  if (expected.write && !log.changed && !log.made) failures.push({ kind: "no-change" });

  const claimed = unbackedClaims(answer.text, answer.toolCalls, { minds: outcome.minds });
  const flagged = answer.steps?.unbackedClaims ?? [];
  const claims = [...new Set([...claimed, ...flagged])];
  if (claims.length > 0) failures.push({ kind: "unbacked", claims, flagged: flagged.length > 0 });

  const cited = new Set([
    ...answer.citations.map((citation) => citation.documentId),
    ...outcome.citedInMinds,
  ]);
  const citedHere = expected.write ? outcome.citedInMinds.length : answer.citations.length;
  if (expected.cite && outcome.scoped.length > 0 && citedHere === 0) {
    failures.push({ kind: "no-citations" });
  }

  if (expected.coverAll) {
    const reply = key(answer.text);
    const uncovered = outcome.scoped
      .filter((document) => !cited.has(document.id) && !reply.includes(key(document.name)))
      .map((document) => document.name);
    if (uncovered.length > 0) failures.push({ kind: "uncovered", documents: uncovered });
  }
  return failures;
}

const CLAIMED: Record<ClaimedChange, string> = {
  made: "made a Mind",
  added: "added text to one",
  changed: "changed one",
  removed: "removed or replaced text",
};

/** A failure as the evaluation's report lists it. */
export function failureLine(failure: RequestFailure): string {
  switch (failure.kind) {
    case "no-change":
      return "asked to write, it changed no Mind";
    case "unbacked":
      return `claims it ${failure.claims.map((claim) => CLAIMED[claim]).join(" and ")}, which no call did${failure.flagged ? " (flagged under the reply)" : ""}`;
    case "no-citations":
      return "Documents in scope, and nothing it wrote cites them";
    case "uncovered":
      return `neither cites nor names ${failure.documents.map((name) => `“${name}”`).join(", ")}`;
  }
}
