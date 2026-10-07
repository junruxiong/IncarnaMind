/**
 * The example Mind (onboarding): "Where tea comes from", a Mind written in
 * advance that works before any chat model is set up. Two excerpts of
 * Wikipedia articles on tea, one English and one Chinese (CC BY-SA, with
 * their attribution in each file and in LICENSE), ship with the app
 * (`Paths.examples`). Made, they are copied into the data folder and linked
 * as a Linked folder; the Mind holds a Question and an Answer whose two
 * Citations quote real sentences of them. Its Citations are checked like any
 * others, against the stored text, once both Documents have been read.
 *
 * Which Mind and Linked folder are the examples is a device value, so no
 * table changes: the renderer marks them "Example", and `remove` deletes both.
 */
import { randomUUID } from "node:crypto";
import { cp, mkdir, rm } from "node:fs/promises";
import { join, relative, resolve } from "node:path";
import type * as Y from "yjs";
import {
  contentHash,
  createElement,
  findBlock,
  type NodeJSON,
  syncContent,
} from "./answers/blocks";
import { withCitations } from "./answers/citations";
import { markdownToBlocks } from "./answers/markdown";
import {
  ANSWER_BLOCK,
  BLOCK_ID_ATTRIBUTE,
  CITATION_NODE,
  type CitationAttributes,
  type CitationRecheck,
  type Document,
  type Examples,
  type LinkedFolder,
  type Mind,
  QUESTION_BLOCK,
  type RecheckCitationInput,
} from "./api";
import type { Language } from "./language";

/** The device value naming the examples, once made. */
const EXAMPLES_VALUE = "examples";
/** Set once the examples were offered on a first run, so they aren't made again unasked. */
const OFFERED_VALUE = "examplesOffered";

/** The shipped files the Citations quote, by file name without extension. */
const ENGLISH_DOCUMENT = "Tea · Wikipedia";
const CHINESE_DOCUMENT = "茶 · 维基百科";

/** The quotes, word for word from the shipped files. */
const QUOTES = [
  {
    document: ENGLISH_DOCUMENT,
    quote:
      "Tea drinking may have begun in the region of Yunnan, where it was used for medicinal purposes.",
  },
  { document: CHINESE_DOCUMENT, quote: "18世纪，英国已成为欧洲最大的茶叶消费国。" },
] as const;

/** The Mind's text, in the interface language it is made in. `[^n]` is the nth quote's Citation. */
const TEXT: Record<
  Language,
  { folder: string; title: string; intro: string; question: string; answer: string }
> = {
  en: {
    folder: "Examples",
    title: "Where tea comes from",
    intro:
      "A short reading note on the history of tea. Two articles are linked as Documents: one in English, one in Chinese.",
    question: "Where does tea come from, and how did it spread around the world?",
    answer: [
      "Tea was probably first drunk in Yunnan, in southwest China, where it was used as a medicine [^1].",
      "",
      "It spread with trade, across Asia and later by sea to Europe, and by the 18th century Britain drank more tea than any other country in Europe [^2].",
    ].join("\n"),
  },
  "zh-CN": {
    folder: "示例",
    title: "茶从哪里来",
    intro: "一篇关于茶的历史的简短阅读笔记。两篇文章作为文档链接在这里：一篇英文，一篇中文。",
    question: "茶起源于哪里？又是如何传遍世界的？",
    answer: [
      "人们最早饮茶可能是在中国西南的云南，当时茶被用作药物 [^1]。",
      "",
      "茶随着贸易传遍亚洲，后来经海路传入欧洲；到 18 世纪，英国已成为欧洲饮茶最多的国家 [^2]。",
    ].join("\n"),
  },
};

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
  mindExists(mindId: string): boolean;
  deleteMind(mindId: string): void;
  editMind<T>(mindId: string, change: (blocks: Y.XmlFragment) => T): T;
  linkFolder(path: string): Promise<LinkedFolder>;
  unlinkFolder(linkedFolderId: string): void;
  linkedFolderExists(linkedFolderId: string): boolean;
  documentsIn(linkedFolderId: string): Document[];
  recheck(input: RecheckCitationInput): CitationRecheck;
  /** Pushes "examples.changed". */
  changed(examples: Examples): void;
}

/** A Document whose text is stored, so its Citations can be checked. */
const hasText = (document: Document) =>
  !["queued", "extracting", "failed"].includes(document.status);

export function createExamples(options: ExamplesOptions) {
  const stored = (): StoredExamples | null => {
    const value = options.readValue(EXAMPLES_VALUE);
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

  const status = (): Examples => {
    const examples = stored();
    const live =
      examples &&
      options.mindExists(examples.mindId) &&
      options.linkedFolderExists(examples.linkedFolderId)
        ? examples
        : null;
    return {
      available: options.source !== undefined,
      mindId: live?.mindId ?? null,
      linkedFolderId: live?.linkedFolderId ?? null,
    };
  };

  /** A Citation's attributes: what is known before it is checked, then the check's result. */
  const citation = (
    quote: (typeof QUOTES)[number],
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

  const answerContent = (
    language: Language,
    citationFor: (index: number) => NodeJSON,
  ): NodeJSON[] =>
    withCitations(markdownToBlocks(TEXT[language].answer), (marker) => citationFor(marker - 1));

  /**
   * Checks the Citations once both Documents' text is stored, and writes the
   * results into the Answer, unless the User has edited it meanwhile.
   */
  const checkWhenRead = () => {
    const examples = stored();
    if (!examples || examples.checked) return;
    if (!options.linkedFolderExists(examples.linkedFolderId)) return;
    const documents = options.documentsIn(examples.linkedFolderId);
    const found = QUOTES.map((quote) => documents.find((each) => each.name === quote.document));
    if (found.some((document) => !document || !hasText(document))) return;
    const results = QUOTES.map((quote, index) => {
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
        answerContent(examples.language, (index) => {
          const quote = QUOTES[index];
          const checked = results[index];
          return quote && checked
            ? citation(quote, checked.document, checked.result)
            : { type: "paragraph" };
        }),
      );
      answer.element.setAttribute("generatedHash", contentHash(answer.element));
    });
    options.writeValue(EXAMPLES_VALUE, { ...examples, checked: true });
  };

  const create = async (): Promise<Examples> => {
    const current = status();
    if (current.mindId) return current;
    const source = options.source;
    if (!source) throw new Error("This copy of IncarnaMind has no examples.");
    const language = options.language();
    const text = TEXT[language];

    // The files, copied into the data folder: the User's own files are never touched.
    const folder = join(options.dataDir, text.folder);
    await mkdir(folder, { recursive: true });
    await cp(source, folder, { recursive: true, force: false, errorOnExist: false });
    const linked = await options.linkFolder(folder);

    const mind = options.createMind(text.title);
    const questionId = randomUUID();
    const answerId = randomUUID();
    options.editMind(mind.id, (blocks) => {
      blocks.delete(0, blocks.length);
      const answer = createElement({
        type: ANSWER_BLOCK,
        attrs: { [BLOCK_ID_ATTRIBUTE]: answerId, questionId, status: "done" },
        content: answerContent(language, (index) => {
          const quote = QUOTES[index];
          return quote ? citation(quote, undefined, null) : { type: "paragraph" };
        }),
      });
      blocks.insert(0, [
        createElement({ type: "paragraph", content: [{ type: "text", text: text.intro }] }),
        createElement({
          type: QUESTION_BLOCK,
          attrs: { [BLOCK_ID_ATTRIBUTE]: questionId, scopeFolderIds: [linked.folderId] },
          content: [{ type: "text", text: text.question }],
        }),
        answer,
        createElement({ type: "paragraph" }),
      ]);
      // Unedited: regenerating it with a model of one's own replaces it without asking.
      answer.setAttribute("generatedHash", contentHash(answer));
    });
    options.writeValue(EXAMPLES_VALUE, {
      mindId: mind.id,
      linkedFolderId: linked.id,
      answerId,
      folder,
      language,
      checked: false,
    } satisfies StoredExamples);
    const made = status();
    options.changed(made);
    checkWhenRead();
    return made;
  };

  return {
    status,

    create,

    /**
     * On a first run (no Mind yet, examples never offered): makes them, once.
     * Returns the examples, or null when nothing was made.
     */
    async offer(hasMinds: boolean): Promise<Examples | null> {
      if (options.source === undefined || options.readValue(OFFERED_VALUE) === true) return null;
      options.writeValue(OFFERED_VALUE, true);
      if (hasMinds) return null;
      return create();
    },

    /** Deletes the example Mind and unlinks its Documents, and the copies of their files. */
    async remove(): Promise<void> {
      const examples = stored();
      if (!examples) return;
      if (options.mindExists(examples.mindId)) options.deleteMind(examples.mindId);
      if (options.linkedFolderExists(examples.linkedFolderId)) {
        options.unlinkFolder(examples.linkedFolderId);
      }
      // Only the copies the examples made, in the data folder: never anything else.
      const inside = relative(resolve(options.dataDir), resolve(examples.folder));
      if (inside !== "" && !inside.startsWith("..") && !inside.includes(":")) {
        await rm(examples.folder, { recursive: true, force: true });
      }
      options.writeValue(EXAMPLES_VALUE, null);
      options.changed(status());
    },

    /** A Document's status changed: the examples' Citations may now be checkable. */
    documentChanged: checkWhenRead,
  };
}
