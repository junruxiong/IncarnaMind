/**
 * The example Minds (onboarding): "Where tea comes from", the first-run
 * default, and one each for papers, reports, contracts and meetings, which
 * the default's banner offers, each in English and in Chinese with Documents of its
 * own. Each is a Mind written in advance that works before any chat model is
 * set up. Its Documents ship with the app (`Paths.examples`), each with its
 * source, licence and attribution in LICENSE. Made, they are copied into the
 * data folder and linked as a Linked folder; the Mind holds a Question and an
 * Answer whose Citations quote real sentences of them (src/core/exampleSets.ts).
 * Its Citations are checked like any others, against the stored text, once all
 * its Documents have been read.
 *
 * Which Mind and Linked folder are an example's is a device value, so no
 * table changes: the renderer marks them "Example", and `remove` deletes both.
 */
import { randomUUID } from "node:crypto";
import { cp, mkdir, rm } from "node:fs/promises";
import { join, relative, resolve } from "node:path";
import type * as Y from "yjs";
import { withCitations } from "../shared/blockJson";
import {
  contentHash,
  createElement,
  findBlock,
  type NodeJSON,
  syncContent,
  toMarkdown,
  topLevelBlocks,
} from "./answers/blocks";
import { markdownToBlocks } from "./answers/markdown";
import {
  ANSWER_BLOCK,
  BLOCK_ID_ATTRIBUTE,
  CITATION_NODE,
  type CitationAttributes,
  type CitationRecheck,
  type Document,
  type ExampleGroup,
  type Examples,
  type LinkedFolder,
  type Mind,
  QUESTION_BLOCK,
  type RecheckCitationInput,
} from "./api";
import { EXAMPLE_SETS, type ExampleQuote, type ExampleSet } from "./exampleSets";
import type { Language } from "./language";

/** The device value naming the tea example, once made; the groups' are `examples.<group>`. */
const EXAMPLES_VALUE = "examples";
/** Set once the examples were offered on a first run, so they aren't made again unasked. */
const OFFERED_VALUE = "examplesOffered";
const ORDER_VALUE = "examplesOrder";
const DEFAULT_GROUP: ExampleGroup = "tea";
const OFFERED_GROUPS: readonly ExampleGroup[] = ["papers", "contracts", "meetings"];

const GROUPS = Object.keys(EXAMPLE_SETS) as ExampleGroup[];
const valueKey = (group: ExampleGroup) =>
  group === "tea" ? EXAMPLES_VALUE : `${EXAMPLES_VALUE}.${group}`;

interface StoredExamples {
  mindId: string;
  linkedFolderId: string;
  answerId: string;
  /** The copies of the files, in the data folder. */
  folder: string;
  /** The interface language they were made in. */
  language: Language;
  /** Once both Citations have been checked against the Documents' text. */
  checked: boolean;
}

export interface ExamplesOptions {
  /** The shipped examples folder; undefined when this copy ships none (most tests). */
  source: string | undefined;
  dataDir: string;
  language(): Language;
  readValue(key: string): unknown;
  writeValue(key: string, value: unknown): void;
  createMind(title: string): Mind;
  /** Renames a Mind. */
  renameMind(mindId: string, title: string): void;
  /** A Mind's title; "" for one that doesn't exist. */
  mindTitle(mindId: string): string;
  mindExists(mindId: string): boolean;
  deleteMind(mindId: string): void;
  editMind<T>(mindId: string, change: (blocks: Y.XmlFragment) => T): T;
  linkFolder(path: string): Promise<LinkedFolder>;
  unlinkFolder(linkedFolderId: string): void;
  linkedFolderExists(linkedFolderId: string): boolean;
  documentsIn(linkedFolderId: string): Document[];
  /** Whether a check result in some Mind relies on one of these Documents (#819). */
  checksRelyOn(documentIds: readonly string[]): boolean;
  recheck(input: RecheckCitationInput): CitationRecheck;
  /** Pushes "examples.changed". */
  changed(examples: Examples): void;
  random?(): number;
}

/** A Document whose text is stored, so its Citations can be checked. */
const hasText = (document: Document) =>
  !["queued", "extracting", "failed"].includes(document.status);

export function createExamples(options: ExamplesOptions) {
  const stored = (group: ExampleGroup): StoredExamples | null => {
    const value = options.readValue(valueKey(group));
    if (!value || typeof value !== "object") return null;
    const examples = value as Partial<StoredExamples>;
    return typeof examples.mindId === "string" &&
      typeof examples.linkedFolderId === "string" &&
      typeof examples.answerId === "string" &&
      typeof examples.folder === "string"
      ? {
          ...(examples as StoredExamples),
          language: examples.language === "zh-CN" ? "zh-CN" : "en",
        }
      : null;
  };

  const isOffered = (value: unknown): value is ExampleGroup[] =>
    Array.isArray(value) &&
    value.length === OFFERED_GROUPS.length &&
    OFFERED_GROUPS.every((group) => value.includes(group));

  const order = (): ExampleGroup[] => {
    const kept = options.readValue(ORDER_VALUE);
    if (isOffered(kept)) return kept;
    const random = options.random ?? Math.random;
    const shuffled = [...OFFERED_GROUPS];
    for (let index = shuffled.length - 1; index > 0; index--) {
      const other = Math.floor(random() * (index + 1));
      [shuffled[index], shuffled[other]] = [
        shuffled[other] as ExampleGroup,
        shuffled[index] as ExampleGroup,
      ];
    }
    options.writeValue(ORDER_VALUE, shuffled);
    return shuffled;
  };

  /** Each part while it exists: the Mind can be deleted, or the folder unlinked, on its own. */
  const status = (group: ExampleGroup = "tea"): Examples => {
    const examples = stored(group);
    const mind = examples && options.mindExists(examples.mindId) ? examples : null;
    const folder =
      examples && options.linkedFolderExists(examples.linkedFolderId) ? examples : null;
    return {
      group,
      available: options.source !== undefined,
      mindId: mind?.mindId ?? null,
      linkedFolderId: folder?.linkedFolderId ?? null,
      answerId: mind?.answerId ?? null,
      more: [
        ...(group === DEFAULT_GROUP ? [] : [DEFAULT_GROUP]),
        ...order().filter((each) => each !== group),
      ],
    };
  };

  /** A Citation's attributes: what is known before it is checked, then the check's result. */
  const citation = (
    quote: ExampleQuote,
    document: Document | undefined,
    result: CitationRecheck | null,
  ): NodeJSON => {
    const attrs: CitationAttributes = {
      passageId: result?.passageId ?? null,
      documentId: document?.id ?? null,
      documentName: document?.name ?? quote.document,
      contentHash: result?.contentHash ?? null,
      pageFrom: result?.pageFrom ?? null,
      pageTo: result?.pageTo ?? null,
      location: result?.location ?? null,
      quote: quote.quote,
      check: result?.check ?? "checking",
      checkReason: result?.checkReason ?? null,
    };
    return {
      type: CITATION_NODE,
      attrs: Object.fromEntries(Object.entries(attrs).filter(([, value]) => value !== null)),
    };
  };

  const answerContent = (set: ExampleSet, citationFor: (index: number) => NodeJSON): NodeJSON[] =>
    withCitations(markdownToBlocks(set.answer), (marker) => citationFor(marker - 1));

  /**
   * Checks an example's Citations once all its Documents' text is stored, and
   * writes the results into the Answer, unless the User has edited it meanwhile.
   */
  const checkWhenRead = (group: ExampleGroup) => {
    const examples = stored(group);
    if (!examples || examples.checked) return;
    if (!options.linkedFolderExists(examples.linkedFolderId)) return;
    const set = EXAMPLE_SETS[group][examples.language];
    const documents = options.documentsIn(examples.linkedFolderId);
    const found = set.quotes.map((quote) => documents.find((each) => each.name === quote.document));
    if (found.some((document) => !document || !hasText(document))) return;
    const results = set.quotes.map((quote, index) => {
      const document = found[index] as Document;
      return {
        document,
        result: options.recheck({
          documentId: document.id,
          pageFrom: null,
          pageTo: null,
          quote: quote.quote,
        }),
      };
    });
    if (!options.mindExists(examples.mindId)) return;
    options.editMind(examples.mindId, (blocks) => {
      const answer = findBlock(blocks, ANSWER_BLOCK, examples.answerId);
      if (!answer) return;
      // The User changed it: theirs now, and left as it is.
      if (answer.element.getAttribute("generatedHash") !== contentHash(answer.element)) return;
      syncContent(
        answer.element,
        answerContent(set, (index) => {
          const quote = set.quotes[index];
          const checked = results[index];
          return quote && checked
            ? citation(quote, checked.document, checked.result)
            : { type: "paragraph" };
        }),
      );
      answer.element.setAttribute("generatedHash", contentHash(answer.element));
    });
    options.writeValue(valueKey(group), { ...examples, checked: true });
  };

  /** Unlinks an example's Documents and deletes the copies of their files. */
  const discardFiles = async (examples: StoredExamples): Promise<void> => {
    if (options.linkedFolderExists(examples.linkedFolderId)) {
      options.unlinkFolder(examples.linkedFolderId);
    }
    // Only the copies the examples made, in the data folder: never anything else.
    const inside = relative(resolve(options.dataDir), resolve(examples.folder));
    if (inside !== "" && !inside.startsWith("..") && !inside.includes(":")) {
      await rm(examples.folder, { recursive: true, force: true });
    }
  };

  /** Deletes an example Mind and unlinks its Documents, and the copies of their files. */
  const remove = async (group: ExampleGroup = "tea"): Promise<void> => {
    const examples = stored(group);
    if (!examples) return;
    if (options.mindExists(examples.mindId)) options.deleteMind(examples.mindId);
    await discardFiles(examples);
    options.writeValue(valueKey(group), null);
    options.changed(status(group));
  };

  /** Copies an example's Documents (with their licence) into the data folder and links them. */
  const copyAndLink = async (source: string, set: ExampleSet) => {
    // The files, copied into the data folder: the User's own files are never touched.
    // The licence goes with them, so the sources and attribution travel with the files.
    const folder = join(options.dataDir, set.folder);
    await mkdir(folder, { recursive: true });
    await cp(join(source, set.source), folder, {
      recursive: true,
      force: false,
      errorOnExist: false,
    });
    await cp(join(source, "LICENSE"), join(folder, "LICENSE"), {
      force: false,
      errorOnExist: false,
    });
    return { folder, linked: await options.linkFolder(folder) };
  };

  /** Writes an example's Note, Question and Answer into its Mind, replacing what is there. */
  const populate = (
    mindId: string,
    set: ExampleSet,
    folderId: string,
    questionId: string,
    answerId: string,
  ) =>
    options.editMind(mindId, (blocks) => {
      blocks.delete(0, blocks.length);
      const answer = createElement({
        type: ANSWER_BLOCK,
        attrs: { [BLOCK_ID_ATTRIBUTE]: answerId, questionId, status: "done" },
        content: answerContent(set, (index) => {
          const quote = set.quotes[index];
          return quote ? citation(quote, undefined, null) : { type: "paragraph" };
        }),
      });
      blocks.insert(0, [
        createElement({ type: "paragraph", content: [{ type: "text", text: set.intro }] }),
        createElement({
          type: QUESTION_BLOCK,
          attrs: { [BLOCK_ID_ATTRIBUTE]: questionId, scopeFolderIds: [folderId] },
          content: [{ type: "text", text: set.question }],
        }),
        answer,
        createElement({ type: "paragraph" }),
      ]);
      // Unedited: regenerating it with a model of one's own replaces it in place.
      answer.setAttribute("generatedHash", contentHash(answer));
    });

  const create = async (group: ExampleGroup = "tea"): Promise<Examples> => {
    if (!GROUPS.includes(group)) throw new Error(`There is no example for "${group}".`);
    const current = status(group);
    if (current.mindId && current.linkedFolderId) return current;
    const source = options.source;
    if (!source) throw new Error("This copy of IncarnaMind has no examples.");
    // What is left of them (the Mind deleted, or the folder unlinked) goes first.
    if (stored(group)) await remove(group);
    const language = options.language();
    const set = EXAMPLE_SETS[group][language];

    const { folder, linked } = await copyAndLink(source, set);

    const mind = options.createMind(set.title);
    const questionId = randomUUID();
    const answerId = randomUUID();
    populate(mind.id, set, linked.folderId, questionId, answerId);
    options.writeValue(valueKey(group), {
      mindId: mind.id,
      linkedFolderId: linked.id,
      answerId,
      folder,
      language,
      checked: false,
    } satisfies StoredExamples);
    const made = status(group);
    options.changed(made);
    checkWhenRead(group);
    return made;
  };

  /**
   * Whether the Mind is still as the example made it (#709): its title, its
   * Note, its Question and its Answer unchanged, and nothing written after them.
   */
  const untouched = (examples: StoredExamples, set: ExampleSet): boolean => {
    if (options.mindTitle(examples.mindId) !== set.title) return false;
    return options.editMind(examples.mindId, (blocks) => {
      const top = topLevelBlocks(blocks);
      const [note, question, answer, ...rest] = top;
      if (!note || !question || !answer) return false;
      if (note.nodeName !== "paragraph" || toMarkdown(note).trim() !== set.intro) return false;
      if (question.nodeName !== QUESTION_BLOCK || toMarkdown(question).trim() !== set.question) {
        return false;
      }
      if (answer.nodeName !== ANSWER_BLOCK) return false;
      if (answer.getAttribute(BLOCK_ID_ATTRIBUTE) !== examples.answerId) return false;
      if (answer.getAttribute("generatedHash") !== contentHash(answer)) return false;
      return rest.every(
        (block) => block.nodeName === "paragraph" && toMarkdown(block).trim() === "",
      );
    });
  };

  /** Swaps an untouched example, in its Mind, for the same example in `language`. */
  const switchLanguage = async (
    group: ExampleGroup,
    examples: StoredExamples,
    source: string,
    language: Language,
  ): Promise<void> => {
    const set = EXAMPLE_SETS[group][language];
    const sameFolder = resolve(examples.folder) === resolve(join(options.dataDir, set.folder));
    // A folder of the same name would mix the two languages' files: the old ones go first.
    if (sameFolder) await discardFiles(examples);
    const { folder, linked } = await copyAndLink(source, set);
    if (!sameFolder) await discardFiles(examples);
    const questionId = randomUUID();
    const answerId = randomUUID();
    populate(examples.mindId, set, linked.folderId, questionId, answerId);
    options.renameMind(examples.mindId, set.title);
    options.writeValue(valueKey(group), {
      mindId: examples.mindId,
      linkedFolderId: linked.id,
      answerId,
      folder,
      language,
      checked: false,
    } satisfies StoredExamples);
    options.changed(status(group));
    checkWhenRead(group);
  };

  return {
    status,

    create,

    /**
     * The interface language changed (#709): each example the User hasn't
     * touched becomes the same example in the new language, in its Mind, with
     * its Documents swapped. One they changed stays as it is, and so does one
     * whose Documents a check result relies on (#819).
     */
    async followLanguage(): Promise<void> {
      const source = options.source;
      if (source === undefined) return;
      const language = options.language();
      for (const group of GROUPS) {
        const examples = stored(group);
        if (!examples || examples.language === language) continue;
        if (!options.mindExists(examples.mindId)) continue;
        if (!untouched(examples, EXAMPLE_SETS[group][examples.language])) continue;
        // Swapping its Documents would leave the checks that relied on them needing a check
        // again, for no change the User made: it stays as it is.
        if (
          options.checksRelyOn(options.documentsIn(examples.linkedFolderId).map((each) => each.id))
        ) {
          continue;
        }
        await switchLanguage(group, examples, source, language);
      }
    },

    /**
     * On a first run (no Mind yet, examples never offered): makes the tea
     * example, once. Returns it, or null when nothing was made.
     */
    async offer(hasMinds: boolean): Promise<Examples | null> {
      if (options.source === undefined || options.readValue(OFFERED_VALUE) === true) return null;
      options.writeValue(OFFERED_VALUE, true);
      if (hasMinds) return null;
      return create("tea");
    },

    remove,

    /** A Document's status changed: the examples' Citations may now be checkable. */
    documentChanged() {
      for (const group of GROUPS) checkWhenRead(group);
    },
  };
}
