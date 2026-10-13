/**
 * Exporting a Mind to Markdown, .docx, PDF or HTML, or its cited pages to a source pack (see
 * `CoreApi.exportMind`). The core makes the file's bytes; the host decides where they go (the
 * desktop app asks with the system save dialog), so nothing here touches the file system. A PDF
 * is printed by the host's `Printer`, from the HTML made here; an HTML file is that same
 * HTML, whole.
 */
import * as Y from "yjs";
import { layoutChart } from "../../shared/chartLayout";
import { chartSvg } from "../../shared/chartSvg";
import { seriesName } from "../../shared/charts";
import {
  type CitationStyle,
  type CitationStyleId,
  citationStyle,
} from "../../shared/citationStyles";
import { citationState, citedLocation, shownDocumentName } from "../../shared/citations";
import type { DrawnPage } from "../../shared/citedPage";
import { titleWithoutSite } from "../../shared/documentNames";
import { translate } from "../../shared/i18n";
import type { Translate } from "../../shared/locations";
import { mindName } from "../../shared/naming";
import type { PrintPaper } from "../../shared/printDocument";
import { pendingProposals } from "../../shared/proposals";
import type { SourceDocument } from "../../shared/sources";
import { wordFileOf, wordImportProgressOf } from "../../shared/wordFiles";
import type { Logger, Pictures, Printer } from "../adapters";
import {
  type CitationAttributes,
  type CitationFormatRequest,
  EXPORT_FORMATS,
  type ExportFormat,
  type ExportMindOptions,
  type FormattedCitations,
  type KeptCitationText,
  MIND_CONTENT_FIELD,
  type Mind,
  type MindExport,
  type MindExportPreview,
  WORD_CHANGE_ID,
} from "../api";
import { InvalidInputError, isRecord } from "../errors";
import { safeFileName } from "../fileNames";
import type { Language } from "../language";
import {
  type Block,
  countQuestions,
  type FailedCheck,
  type Footnote,
  footnotesIn,
  type ReadOptions,
  readBlocks,
  reviewReading,
  theirChangesReading,
} from "../mindText";
import { withChangesAccepted } from "../shownChanges";
import { addSources, styleCitations } from "./citations";
import { renderDocx } from "./docx";
import { renderHtml } from "./html";
import { exportLabels } from "./labels";
import { renderMarkdown } from "./markdown";
import { unifyIdeographs } from "./pdfUnicode";
import { renderPrintHtml } from "./printHtml";
import { isUnverified, numberCitations, packPages, renderSourcePack } from "./sourcePack";
import { exportWordPatch } from "./wordPatch";
import type { ReviewChoices } from "./wordReview";

/** What a footnote needs of a live Document: a web page's site, address and fetch dates. */
type LiveDocument = SourceDocument;

export interface ExportsDependencies {
  /** The Mind; throws if it doesn't exist or was deleted. */
  mind(mindId: unknown): Mind;
  /** Reads the Mind's Blocks (see `MindContent.read`). */
  read<T>(mindId: string, look: (blocks: Y.XmlFragment) => T): T;
  /** The live Documents: a Citation whose Document was deleted can't be checked. */
  liveDocuments(): SourceDocument[];
  /** The text kept of Documents unlinked with their Linked folder: Citations quoting it still check. */
  keptCitationTexts(): KeptCitationText[];
  /** The interface language, which the export's own words are in. */
  language(): Language;
  /** The Mind's citation style (#289). */
  citationStyle(mindId: string): CitationStyleId;
  /** Whether the Mind shows its Bibliography without a citation style (#102). */
  bibliography(mindId: string): boolean;
  /** Formats Citations in a style, as the editor does (see `CoreApi.formatCitations`). */
  formatCitations(request: CitationFormatRequest): Promise<FormattedCitations>;
  /** A Citation's saved page (#133); null if none is saved. */
  citedPage(citation: Readonly<Record<string, unknown>>): Promise<DrawnPage | null>;
  /** Prints a PDF; null where this copy can't. */
  printer: Printer | null;
  /** The paper a PDF is printed on, which goes by the computer's region (`paperFor`). */
  paper(): PrintPaper;
  /** Draws charts' pictures for Word; without it, a chart goes into Word as its table. */
  pictures?: Pictures;
  /** The package a Mind was imported from (#104); throws if it is gone. */
  wordPackage(mindId: string): Promise<Uint8Array>;
  /** The name the User's edits go under as tracked changes in a Word file (`Settings.author`). */
  author(): string;
  log: Logger;
  now(): string;
}

/** The text's width on A4 and on Letter, in CSS pixels, so Word never scales the picture. */
const PICTURE_WIDTH = 600;
/** Sharp when printed or zoomed. */
const PICTURE_SCALE = 2;

const EXTENSIONS: Readonly<Record<ExportFormat, string>> = {
  markdown: "md",
  docx: "docx",
  pdf: "pdf",
  "source-pack": "pdf",
  html: "html",
};

/** What an export holds, read once for both its preview and its file. */
interface Prepared {
  format: ExportFormat;
  options: Required<ExportMindOptions>;
  mindId: string;
  title: string;
  readOptions: ReadOptions;
  /** For a Mind imported from a Word file: whether its import is still going. */
  word: { importing: boolean } | null;
  blocks: Block[];
  preview: MindExportPreview;
  language: Language;
  style: CitationStyle;
  documents: readonly SourceDocument[];
  bibliography: boolean;
  /** Each Citation's footnote with the attributes it was made from, for the source pack. */
  attributesOf: ReadonlyMap<Footnote, Partial<CitationAttributes>>;
}

export function createExports(deps: ExportsDependencies) {
  const prepare = (mindId: unknown, input: unknown): Prepared => {
    const options = parseExportOptions(input);
    const { format, includeQuestions } = options;
    const mind = deps.mind(mindId);
    const language = deps.language();
    const style = citationStyle(deps.citationStyle(mind.id));
    const documents = deps.liveDocuments();
    const kept = deps.keptCitationTexts();
    const attributesOf = new Map<Footnote, Partial<CitationAttributes>>();
    const footnote = (attributes: Record<string, unknown>): Footnote => {
      const note = footnoteOf(attributes as Partial<CitationAttributes>, documents, kept, language);
      attributesOf.set(note, attributes as Partial<CitationAttributes>);
      return note;
    };
    // Footnotes go right after the word, and so do Numbered's Citations; a style decides for
    // its own (./citations).
    const trimBeforeCitation = style.form === "numbered";

    const live = new Set(documents.map((document) => document.id));
    const readOptions: ReadOptions = {
      includeQuestions,
      footnote,
      trimBeforeCitation,
      trimBeforeFootnote: true,
      comparisonText: (key, params) => translate(language, key, params),
      documentRemoved: (documentId) => !live.has(documentId),
    };
    const { read, questions, word, theirChanges } = deps.read(mind.id, (fragment) => {
      const { doc } = fragment;
      const fromWord = doc !== null && wordFileOf(doc) !== null;
      // A Word file shows what is waiting as tracked changes, and can mark AI text (#107); the
      // others show someone's changes from the Word file as changes, or accepted (#734).
      if (format === "docx") readOptions.review = reviewReading(fragment);
      else if (doc && fromWord && format !== "source-pack")
        readOptions.review = theirChangesReading(doc);
      return {
        read: readBlocks(fragment, readOptions),
        questions: countQuestions(fragment),
        word: doc && fromWord ? { importing: wordImportProgressOf(doc) !== null } : null,
        theirChanges: fromWord
          ? pendingProposals(fragment).filter((each) => each.id.startsWith(WORD_CHANGE_ID)).length
          : 0,
      };
    });
    const blocks =
      format !== "docx" && options.theirChanges === "accepted" ? withChangesAccepted(read) : read;
    // A source pack holds the Citations the Mind numbers; the others, every footnote.
    const footnotes =
      format === "source-pack"
        ? numberCitations(blocks, style.form, attributesOf).map((each) => each.footnote)
        : footnotesIn(blocks);
    const name = safeFileName(mindName(mind) || translate(language, "mind.untitled"), "Mind");
    const preview: MindExportPreview = {
      fileName: `${format === "source-pack" ? translate(language, "export.pack.fileName", { name }) : name}.${EXTENSIONS[format]}`,
      citations: footnotes.length,
      unverifiedCitations: footnotes.filter((each) => each.unverified).length,
      questions,
      fromWord: word !== null,
      theirChanges,
    };
    return {
      format,
      options,
      mindId: mind.id,
      title: mind.title,
      readOptions,
      word,
      blocks,
      preview,
      language,
      style,
      documents,
      bibliography: deps.bibliography(mind.id),
      attributesOf,
    };
  };

  /** The source pack's PDF: the Mind's cited pages, as saved, with their quotes marked. */
  const sourcePack = async (prepared: Prepared): Promise<MindExport> => {
    const { title, blocks, style, language, attributesOf, preview } = prepared;
    const citations = numberCitations(blocks, style.form, attributesOf);
    if (citations.length === 0) throw new Error(translate(language, "export.pack.empty"));
    const pages = await packPages(citations, deps.citedPage);
    const printed = await (deps.printer as Printer).print({
      language,
      paper: deps.paper(),
      title: preview.fileName.slice(0, -".pdf".length),
      body: renderSourcePack(title, pages, language),
      notes: [],
    });
    return {
      ...preview,
      unverifiedCitations: pages.flatMap((page) => page.entries).filter(isUnverified).length,
      data: unifyIdeographs(printed),
      word: null,
    };
  };

  return {
    preview(mindId: unknown, options: unknown): MindExportPreview {
      return prepare(mindId, options).preview;
    },

    async export(mindId: unknown, options: unknown): Promise<MindExport> {
      const prepared = prepare(mindId, options);
      const { title, preview, language, format, style, documents } = prepared;
      if ((format === "pdf" || format === "source-pack") && !deps.printer) {
        throw new Error("This copy of IncarnaMind can't make PDFs.");
      }
      if (format === "source-pack") return sourcePack(prepared);
      if (format === "docx" && prepared.word) return exportIntoWord(prepared, prepared.word);
      const blocks =
        style.form !== "numbered"
          ? await styleCitations(prepared.blocks, style, deps.formatCitations)
          : prepared.bibliography
            ? addSources(
                prepared.blocks,
                { language, documents },
                translate(language, "bibliography.title"),
              )
            : prepared.blocks;
      const labels = exportLabels(language);
      let data: Uint8Array;
      if (format === "markdown") {
        data = new TextEncoder().encode(renderMarkdown(title, blocks, labels));
      } else if (format === "docx") {
        data = renderDocx(await withPictures(blocks, language, deps), {
          title,
          labels,
          language: language === "zh-CN" ? "zh-CN" : "en-US",
          paper: language === "en" ? "letter" : "a4",
          created: deps.now(),
          review: reviewChoices(prepared, false),
        });
      } else if (format === "html") {
        const drawn = await withPictures(blocks, language, deps);
        data = new TextEncoder().encode(
          renderHtml(title, drawn, labels, {
            language,
            name: preview.fileName.slice(0, -".html".length),
          }),
        );
      } else {
        // Charts as Word has them: the picture, drawn at twice its size, else the table of values.
        const drawn = await withPictures(blocks, language, deps);
        const { body, notes } = renderPrintHtml(title, drawn, labels);
        const printed = await (deps.printer as Printer).print({
          language,
          paper: deps.paper(),
          title: preview.fileName.slice(0, -".pdf".length),
          body,
          notes,
        });
        data = unifyIdeographs(printed);
      }
      return { ...preview, data, word: null };
    },
  };

  /** What a Word file says for review (#107): the User's edits are tracked only in a file of theirs. */
  function reviewChoices(prepared: Prepared, imported: boolean): ReviewChoices {
    const { options } = prepared;
    return {
      trackAs: imported && options.trackChanges ? deps.author() : null,
      markAiText: options.markAiText,
      commentFailedChecks: options.commentFailedChecks,
      now: deps.now(),
    };
  }

  /** A Mind imported from a Word file, written back into that file (#105). */
  async function exportIntoWord(
    prepared: Prepared,
    word: NonNullable<Prepared["word"]>,
  ): Promise<MindExport> {
    if (word.importing) throw new Error("This Mind is still being imported from its Word file.");
    const { language, style, documents } = prepared;
    // A copy of the Mind as it is now: typing while the file is written doesn't change it.
    const state = deps.read(prepared.mindId, (blocks) =>
      blocks.doc ? Y.encodeStateAsUpdate(blocks.doc) : null,
    );
    const doc = new Y.Doc();
    try {
      if (state) Y.applyUpdate(doc, state);
      const { data, report } = await exportWordPatch({
        bytes: await deps.wordPackage(prepared.mindId),
        blocks: doc.getXmlFragment(MIND_CONTENT_FIELD),
        readOptions: prepared.readOptions,
        labels: exportLabels(language),
        style,
        bibliography: prepared.bibliography,
        bibliographyHeading: translate(language, "bibliography.title"),
        sources: { language, documents },
        formatCitations: deps.formatCitations,
        withPictures: (blocks) => withPictures(blocks, language, deps),
        review: reviewChoices(prepared, true),
      });
      deps.log.info("export.word", {
        notKept: report.notKept.length,
        unknownParts: report.unknownParts.length,
      });
      return { ...prepared.preview, data, word: report };
    } finally {
      doc.destroy();
    }
  }
}

/** A picture that can't be drawn leaves the chart's table, and the log says why. */
async function withPictures(
  blocks: readonly Block[],
  language: Language,
  deps: Pick<ExportsDependencies, "pictures" | "log">,
): Promise<Block[]> {
  const { pictures } = deps;
  if (!pictures) return [...blocks];
  const draw = async (block: Extract<Block, { kind: "chart" }>): Promise<Block> => {
    const { chart } = block;
    const layout = layoutChart(chart, {
      width: PICTURE_WIDTH,
      language,
      legendInside: true,
      seriesName: (index) =>
        seriesName(chart, index, (key, params) => translate(language, key, params)),
    });
    if (layout.points.length === 0) return block;
    try {
      const png = await pictures.png(chartSvg(layout, PICTURE_SCALE));
      const kind = translate(language, `chart.kindName.${chart.kind}`);
      return {
        ...block,
        picture: {
          src: `data:image/png;base64,${Buffer.from(png).toString("base64")}`,
          alt: chart.title
            ? translate(language, "export.chart.picture", { kind, title: chart.title })
            : kind,
        },
      };
    } catch (error) {
      deps.log.warn("export.chartPicture", {
        kind: chart.kind,
        points: layout.points.length,
        message: error instanceof Error ? error.message : String(error),
      });
      return block;
    }
  };
  // One at a time: each picture opens a window of its own.
  const inOrder = async <T, U>(items: readonly T[], each: (item: T) => Promise<U>) => {
    const done: U[] = [];
    for (const item of items) done.push(await each(item));
    return done;
  };
  const visit = async (block: Block): Promise<Block> => {
    switch (block.kind) {
      case "chart":
        return draw(block);
      case "list":
        return { ...block, items: await inOrder(block.items, (item) => inOrder(item, visit)) };
      case "quote":
        return { ...block, content: await inOrder(block.content, visit) };
      default:
        return block;
    }
  };
  return inOrder(blocks, visit);
}

function parseExportOptions(input: unknown): Required<ExportMindOptions> {
  if (!isRecord(input)) throw new InvalidInputError("Export options must be an object.");
  const { format, includeQuestions, accept, theirChanges } = input;
  const choice = (name: "trackChanges" | "markAiText" | "commentFailedChecks") => {
    const value = input[name];
    if (value !== undefined && typeof value !== "boolean") {
      throw new InvalidInputError({ key: "error.export.choice", values: { name } });
    }
    return value === true;
  };
  if (!EXPORT_FORMATS.some((each) => each === format)) {
    throw new InvalidInputError(
      'The export format must be "markdown", "docx", "pdf", "source-pack" or "html".',
    );
  }
  if (includeQuestions !== undefined && typeof includeQuestions !== "boolean") {
    throw new InvalidInputError("includeQuestions must be true or false.");
  }
  if (theirChanges !== undefined && theirChanges !== "shown" && theirChanges !== "accepted") {
    throw new InvalidInputError({ key: "error.export.theirChanges" });
  }
  if (
    accept !== undefined &&
    !(Array.isArray(accept) && accept.every((key) => typeof key === "string"))
  ) {
    throw new InvalidInputError("accept must be a list of what can't be kept.");
  }
  // Markdown is an archive, so it keeps Questions; the others are deliverables, so they don't.
  return {
    format: format as ExportFormat,
    includeQuestions: includeQuestions ?? format === "markdown",
    accept: accept ?? [],
    trackChanges: choice("trackChanges"),
    markAiText: choice("markAiText"),
    commentFailedChecks: choice("commentFailedChecks"),
    theirChanges: theirChanges ?? "shown",
  };
}

/** The day a web page's version was fetched, as the interface language writes a date. */
function dayOf(at: string, language: Language): string {
  return new Intl.DateTimeFormat(language === "zh-CN" ? "zh-CN" : "en-GB", {
    day: "numeric",
    month: "long",
    year: "numeric",
  }).format(new Date(at));
}

/**
 * A Citation's footnote: its Document's name and its Location's short label,
 * e.g. "Tides, p. 12–13", "Deck, slide 4" or "Model, Revenue, rows 12–14",
 * unverified unless its badge says "Quote found". A live web page's also names
 * its site, its address and the day the quoted version was fetched, as the
 * "accessed" date citation styles read: "Tides, Tide Wiki, § History, ¶ 3,
 * https://tides.example/history, accessed 3 October 2026".
 */
function footnoteOf(
  attributes: Partial<CitationAttributes>,
  documents: readonly LiveDocument[],
  kept: readonly KeptCitationText[],
  language: Language,
): Footnote {
  const t: Translate = (key, params) => translate(language, key, params);
  const document = shownDocumentName(attributes, documents) || t("export.unnamedDocument");
  const location = citedLocation(attributes, t);
  const state = citationState(attributes, documents, kept);
  const unverified = state.check !== "found";
  const failed: FailedCheck | null =
    state.check === "not-found" || state.check === "unsupported" || state.check === "contradicted"
      ? { check: state.check, quote: attributes.quote ?? null, verdict: attributes.verdict ?? null }
      : null;
  const web = documents.find((each) => each.id === state.documentId)?.web;
  if (!web) {
    return {
      source: location
        ? t("export.citation.location", { document, location })
        : t("export.citation.document", { document }),
      unverified,
      ...(failed && { failed }),
    };
  }
  const version = web.versions.find((each) => each.contentHash === attributes.contentHash);
  // The site is named after the title, so a title that ends with it leaves it off.
  const parts = [titleWithoutSite(document, web.siteName)];
  if (web.siteName && web.siteName !== document) parts.push(web.siteName);
  if (location) parts.push(location);
  parts.push(web.address);
  if (version)
    parts.push(t("export.citation.accessed", { date: dayOf(version.fetchedAt, language) }));
  return {
    source: parts.join(t("export.citation.separator")),
    unverified,
    link: web.address,
    ...(failed && { failed }),
  };
}
