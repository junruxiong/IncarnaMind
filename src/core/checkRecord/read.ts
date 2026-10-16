/**
 * What a Mind's checks left on it, read for the check record (#738): each
 * checked sentence and each Citation with its result, the Document version it
 * rests on, the model, the time, and whose text it is. Read from the Mind's
 * own marks and Citations, the AI use record's models and the Documents as
 * they are now; what none of them holds is left empty, never guessed.
 */
import * as Y from "yjs";
import {
  citationState,
  citedLocation,
  liveDocumentOf,
  shownDocumentName,
} from "../../shared/citations";
import { claimCheckShown, type SentenceResult, sentenceResultOf } from "../../shared/claimChecks";
import type { Translate } from "../../shared/locations";
import { comparableSentence, INLINE_NODE, sentenceSpans } from "../../shared/sentences";
import { textOriginOf } from "../../shared/textOrigins";
import { groupsOf, isProposed } from "../aiUseStatement/figures";
import { textAttribute } from "../answers/blocks";
import { checkedSentences } from "../answers/checks/sentences";
import {
  type AiUseEntry,
  ANSWER_BLOCK,
  BLOCK_ID_ATTRIBUTE,
  type CitationAttributes,
  type CitationCheck,
  type Document,
  GENERATED_MARK,
  type KeptCitationText,
  QUESTION_BLOCK,
} from "../api";
import { type Footnote, type Inline, readBlock, reviewReading } from "../mindText";
import { layout } from "../proposals/write";

/** Whose words a checked sentence is: the User's, written by an AI model, or someone else's (#78, #95). */
export type TextOrigin = "user" | "ai" | "other";

/** The model that checked a sentence or wrote a Citation's Answer, and the service it ran on. */
export interface RecordModel {
  name: string;
  /** The service the model's company runs it on; null for a model on this computer. */
  service: string | null;
}

/** The Document version a check rests on, as its file was when it was checked. */
export interface RecordDocument {
  name: string;
  /** The file's name where it is kept, or a web page's address; null once the Document is gone. */
  file: string | null;
  /** In bytes, when this version is the file as it is now. */
  size: number | null;
  /** The SHA-256 of the file's bytes, in hex. */
  sha256: string | null;
  location: string | null;
  quote: string | null;
}

export interface RecordItem {
  kind: "sentence" | "citation";
  /** The sentence checked, or the one a Citation stands in. */
  text: string;
  result: SentenceResult | CitationCheck;
  origin: TextOrigin;
  /** What the evidence says, in the line the check kept. */
  says: string | null;
  /** Null when no Document backs it: an unsupported sentence. */
  evidence: RecordDocument | null;
  /** The check's model for a sentence; the model that wrote the Answer for a Citation, whose quote IncarnaMind checks without one. */
  model: RecordModel | null;
  /** When a sentence was checked, ISO 8601 UTC. A Citation's quote check isn't dated. */
  checkedAt: string | null;
}

export interface RecordSources {
  fragment: Y.XmlFragment;
  entries: readonly AiUseEntry[];
  documents: readonly Document[];
  kept: readonly KeptCitationText[];
  folderId: string | null;
  t: Translate;
  /** A model as the record names it, or null when the Mind doesn't say. */
  model(providerId: string | null, modelId: string | null): RecordModel | null;
  /** A Citation's quote as its Document's text reads it (#905): a PDF's split words whole again. */
  readQuote(attributes: Partial<CitationAttributes>): string | null;
}

interface Run {
  insert: unknown;
  attributes?: Record<string, unknown>;
}

const fileNameOf = (path: string) => path.split(/[\\/]/).at(-1) ?? path;

function documentOf(
  attributes: Partial<CitationAttributes>,
  { documents, t, readQuote }: Pick<RecordSources, "documents" | "t" | "readQuote">,
): RecordDocument {
  const found = liveDocumentOf(attributes, documents);
  const live = found && documents.find((each) => each.id === found.id);
  const sha256 = attributes.contentHash ? attributes.contentHash : null;
  return {
    name: shownDocumentName(attributes, documents) || t("export.unnamedDocument"),
    file: live ? (live.web?.address ?? fileNameOf(live.path)) : null,
    size: live && sha256 !== null && live.contentHash === sha256 ? live.size : null,
    sha256,
    location: citedLocation(attributes, t),
    quote: readQuote(attributes),
  };
}

/** The top-level Block an element is in. */
function topOf(fragment: Y.XmlFragment, element: Y.XmlElement): Y.XmlElement {
  let top = element;
  while (top.parent instanceof Y.XmlElement) top = top.parent;
  return top.parent === fragment ? top : element;
}

const idOf = (element: Y.XmlElement): string | null => {
  const id = element.getAttribute(BLOCK_ID_ATTRIBUTE);
  return typeof id === "string" && id !== "" ? id : null;
};

/** Whose words a Block holds, apart from any AI text marks inside it. */
function blockOrigin(fragment: Y.XmlFragment, top: Y.XmlElement): TextOrigin | null {
  const id = idOf(top);
  if (fragment.doc && id !== null && textOriginOf(fragment.doc, id) !== null) return "other";
  return top.nodeName === ANSWER_BLOCK ? "ai" : null;
}

/** Whether any of `from`..`to` of a paragraph's text is AI text. */
function holdsAiText(element: Y.XmlElement, from: number, to: number): boolean {
  for (const { node, start } of layout(element)) {
    if (!(node instanceof Y.XmlText)) continue;
    let offset = start;
    for (const run of node.toDelta() as Run[]) {
      const length = typeof run.insert === "string" ? run.insert.length : 1;
      if (run.attributes?.[GENERATED_MARK] && offset < to && offset + length > from) return true;
      offset += length;
    }
  }
  return false;
}

/** Every checked sentence and every Citation of the Mind, each in reading order. */
export function readRecord(sources: RecordSources): {
  sentences: RecordItem[];
  citations: RecordItem[];
} {
  const { fragment, entries, documents, folderId, model } = sources;
  const versions = new Map(documents.map((document) => [document.id, document.contentHash]));

  const sentences = checkedSentences(fragment).flatMap(
    ({ element, from, to, check, sentence }): RecordItem[] => {
      const text = comparableSentence(sentence);
      const result = sentenceResultOf(
        claimCheckShown(check, { text: sentence, versions, folderId }),
      );
      if (!result) return [];
      return [
        {
          kind: "sentence",
          text,
          result,
          origin:
            blockOrigin(fragment, topOf(fragment, element)) ??
            (holdsAiText(element, from, to) ? "ai" : "user"),
          says: check.says,
          evidence: check.evidence ? documentOf(check.evidence, sources) : null,
          model: model(check.providerId, check.modelId),
          checkedAt: check.checkedAt === "" ? null : check.checkedAt,
        },
      ];
    },
  );

  const attributesOf = new Map<Footnote, Partial<CitationAttributes>>();
  const readOptions = {
    includeQuestions: false,
    footnote: (attributes: Record<string, unknown>): Footnote => {
      const note: Footnote = { source: "", unverified: false };
      attributesOf.set(note, attributes as Partial<CitationAttributes>);
      return note;
    },
    trimBeforeCitation: false,
    review: reviewReading(fragment),
  };
  const citations: RecordItem[] = [];
  for (const child of fragment.toArray()) {
    if (!(child instanceof Y.XmlElement) || child.nodeName === QUESTION_BLOCK) continue;
    if (isProposed(child)) continue;
    const id = idOf(child);
    const entry =
      child.nodeName === ANSWER_BLOCK || id === null
        ? undefined
        : entries.find((each) => each.modelId !== null && each.blockIds.includes(id));
    const wroteIt =
      child.nodeName === ANSWER_BLOCK
        ? model(textAttribute(child, "providerId"), textAttribute(child, "modelId"))
        : entry
          ? model(entry.providerId, entry.modelId)
          : null;
    const origin = blockOrigin(fragment, child);
    for (const group of groupsOf(readBlock(child, readOptions))) {
      citations.push(...citationsIn(group, { attributesOf, origin, wroteIt, sources }));
    }
  }
  return { sentences, citations };
}

/** The Citations in one run of inline text, each with the sentence it follows. */
function citationsIn(
  group: readonly Inline[],
  context: {
    attributesOf: ReadonlyMap<Footnote, Partial<CitationAttributes>>;
    origin: TextOrigin | null;
    wroteIt: RecordModel | null;
    sources: RecordSources;
  },
): RecordItem[] {
  const { attributesOf, origin, wroteIt, sources } = context;
  let plain = "";
  const aiAt: boolean[] = [];
  const placed: { offset: number; attributes: Partial<CitationAttributes> }[] = [];
  const add = (text: string, ai: boolean) => {
    for (const character of text) {
      plain += character;
      aiAt.push(ai);
    }
  };
  for (const inline of group) {
    if (inline.inserted) continue;
    if (inline.kind === "text") {
      add(inline.text, inline.ai === true);
    } else if (inline.kind === "citation") {
      const attributes = attributesOf.get(inline.footnote);
      if (attributes) placed.push({ offset: aiAt.length, attributes });
      add(INLINE_NODE, false);
    } else if (inline.kind === "break") {
      add("\n", false);
    }
  }
  const spans = sentenceSpans(plain);
  return placed.map(({ offset, attributes }): RecordItem => {
    const span = spans.find((each) => each.from <= offset && offset < each.to) ??
      spans.filter((each) => each.from <= offset).at(-1) ?? { from: 0, to: aiAt.length };
    return {
      kind: "citation",
      text: comparableSentence(plain.slice(span.from, span.to)),
      result: citationState(attributes, sources.documents, sources.kept).check,
      origin: origin ?? (aiAt.slice(span.from, span.to).some(Boolean) ? "ai" : "user"),
      says: attributes.verdict ?? null,
      evidence: documentOf(attributes, sources),
      model: wroteIt,
      checkedAt: null,
    };
  });
}
