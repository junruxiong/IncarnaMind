/**
 * Exporting a Mind to Markdown or .docx (see `CoreApi.exportMind`). The core
 * makes the file's bytes; the host decides where they go (the desktop app asks
 * with the system save dialog), so nothing here touches the file system.
 */
import type * as Y from "yjs";
import { layoutChart } from "../../shared/chartLayout";
import { chartSvg } from "../../shared/chartSvg";
import { seriesName } from "../../shared/charts";
import {
  type CitationStyle,
  type CitationStyleId,
  citationStyle,
} from "../../shared/citationStyles";
import { citationState, citedLocation } from "../../shared/citations";
import { translate } from "../../shared/i18n";
import { mindName } from "../../shared/naming";
import type { Logger, Pictures } from "../adapters";
import type {
  CitationAttributes,
  CitationFormatRequest,
  Document,
  ExportFormat,
  ExportMindOptions,
  FormattedCitations,
  KeptCitationText,
  Mind,
  MindExport,
  MindExportPreview,
} from "../api";
import { InvalidInputError, isRecord } from "../errors";
import { safeFileName } from "../fileNames";
import type { Language } from "../language";
import { type Block, countQuestions, type Footnote, footnotesIn, readBlocks } from "../mindText";
import { styleCitations } from "./citations";
import { renderDocx } from "./docx";
import { exportLabels } from "./labels";
import { renderMarkdown } from "./markdown";

export interface ExportsDependencies {
  /** The Mind; throws if it doesn't exist or was deleted. */
  mind(mindId: unknown): Mind;
  /** Reads the Mind's Blocks (see `MindContent.read`). */
  read<T>(mindId: string, look: (blocks: Y.XmlFragment) => T): T;
  /** The live Documents: a Citation whose Document was deleted can't be checked. */
  liveDocuments(): Pick<Document, "id" | "contentHash">[];
  /** The text kept of Documents unlinked with their Linked folder: Citations quoting it still check. */
  keptCitationTexts(): KeptCitationText[];
  /** The interface language, which the export's own words are in. */
  language(): Language;
  /** The Mind's citation style (#289). */
  citationStyle(mindId: string): CitationStyleId;
  /** Formats Citations in a style, as the editor does (see `CoreApi.formatCitations`). */
  formatCitations(request: CitationFormatRequest): Promise<FormattedCitations>;
  /** Draws charts' pictures for Word; without it, a chart goes into Word as its table. */
  pictures?: Pictures;
  log: Logger;
  now(): string;
}

/** The text's width on A4 and on Letter, in CSS pixels, so Word never scales the picture. */
const PICTURE_WIDTH = 600;
/** Sharp when printed or zoomed. */
const PICTURE_SCALE = 2;

const EXTENSIONS: Readonly<Record<ExportFormat, string>> = { markdown: "md", docx: "docx" };

/** What an export holds, read once for both its preview and its file. */
interface Prepared {
  format: ExportFormat;
  title: string;
  blocks: Block[];
  preview: MindExportPreview;
  language: Language;
  style: CitationStyle;
}

export function createExports(deps: ExportsDependencies) {
  const prepare = (mindId: unknown, input: unknown): Prepared => {
    const { format, includeQuestions } = parseExportOptions(input);
    const mind = deps.mind(mindId);
    const language = deps.language();
    const style = citationStyle(deps.citationStyle(mind.id));
    const documents = deps.liveDocuments();
    const kept = deps.keptCitationTexts();
    const footnote = (attributes: Record<string, unknown>): Footnote =>
      footnoteOf(attributes as Partial<CitationAttributes>, documents, kept, language);
    // Footnotes go right after the word, and so do Numbered's Citations; a style decides for
    // its own (./citations).
    const trimBeforeCitation = style.form === "numbered";

    const live = new Set(documents.map((document) => document.id));
    const { blocks, questions } = deps.read(mind.id, (fragment) => ({
      blocks: readBlocks(fragment, {
        includeQuestions,
        footnote,
        trimBeforeCitation,
        trimBeforeFootnote: true,
        comparisonText: (key, params) => translate(language, key, params),
        documentRemoved: (documentId) => !live.has(documentId),
      }),
      questions: countQuestions(fragment),
    }));
    const footnotes = footnotesIn(blocks);
    const preview: MindExportPreview = {
      fileName: `${safeFileName(mindName(mind) || translate(language, "mind.untitled"), "Mind")}.${EXTENSIONS[format]}`,
      citations: footnotes.length,
      unverifiedCitations: footnotes.filter((each) => each.unverified).length,
      questions,
    };
    return { format, title: mind.title, blocks, preview, language, style };
  };

  return {
    preview(mindId: unknown, options: unknown): MindExportPreview {
      return prepare(mindId, options).preview;
    },

    async export(mindId: unknown, options: unknown): Promise<MindExport> {
      const prepared = prepare(mindId, options);
      const { title, preview, language, format, style } = prepared;
      const blocks =
        style.form === "numbered"
          ? prepared.blocks
          : await styleCitations(prepared.blocks, style, deps.formatCitations);
      const labels = exportLabels(language);
      const data =
        format === "markdown"
          ? new TextEncoder().encode(renderMarkdown(title, blocks, labels))
          : renderDocx(await withPictures(blocks, language, deps), {
              title,
              labels,
              language: language === "zh-CN" ? "zh-CN" : "en-US",
              paper: language === "en" ? "letter" : "a4",
              created: deps.now(),
            });
      return { ...preview, data };
    },
  };
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
  const { format, includeQuestions } = input;
  if (format !== "markdown" && format !== "docx") {
    throw new InvalidInputError('The export format must be "markdown" or "docx".');
  }
  if (includeQuestions !== undefined && typeof includeQuestions !== "boolean") {
    throw new InvalidInputError("includeQuestions must be true or false.");
  }
  // Markdown is an archive, so it keeps Questions; the .docx is the deliverable, so it doesn't.
  return { format, includeQuestions: includeQuestions ?? format === "markdown" };
}

/**
 * A Citation's footnote: its Document's name and its Location's short label,
 * e.g. "Tides, p. 12–13", "Deck, slide 4" or "Model, Revenue, rows 12–14",
 * unverified unless its badge says "Quote found".
 */
function footnoteOf(
  attributes: Partial<CitationAttributes>,
  documents: readonly Pick<Document, "id" | "contentHash">[],
  kept: readonly KeptCitationText[],
  language: Language,
): Footnote {
  const document = attributes.documentName || translate(language, "export.unnamedDocument");
  const location = citedLocation(attributes, (key, params) => translate(language, key, params));
  return {
    source: location
      ? translate(language, "export.citation.location", { document, location })
      : translate(language, "export.citation.document", { document }),
    unverified: citationState(attributes, documents, kept).check !== "found",
  };
}
