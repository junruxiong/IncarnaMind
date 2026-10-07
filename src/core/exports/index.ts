/**
 * Exporting a Mind to Markdown or .docx (see `CoreApi.exportMind`). The core
 * makes the file's bytes; the host decides where they go (the desktop app asks
 * with the system save dialog), so nothing here touches the file system.
 */
import type * as Y from "yjs";
import { citationState, citedLocation } from "../../shared/citations";
import { translate } from "../../shared/i18n";
import type {
  CitationAttributes,
  Document,
  ExportFormat,
  ExportMindOptions,
  Mind,
  MindExport,
  MindExportPreview,
} from "../api";
import { InvalidInputError, isRecord } from "../errors";
import { safeFileName } from "../fileNames";
import type { Language } from "../language";
import { renderDocx } from "./docx";
import { renderMarkdown } from "./markdown";
import { type Block, countQuestions, type Footnote, footnotesIn, readBlocks } from "./model";

export interface ExportsDependencies {
  /** The Mind; throws if it doesn't exist or was deleted. */
  mind(mindId: unknown): Mind;
  /** Reads the Mind's Blocks (see `MindContent.read`). */
  read<T>(mindId: string, look: (blocks: Y.XmlFragment) => T): T;
  /** The live Documents: a Citation whose Document was deleted can't be checked. */
  liveDocuments(): Pick<Document, "id" | "contentHash">[];
  /** The interface language, which the export's own words are in. */
  language(): Language;
  now(): string;
}

const EXTENSIONS: Readonly<Record<ExportFormat, string>> = { markdown: "md", docx: "docx" };

/** What an export holds, read once for both its preview and its file. */
interface Prepared {
  format: ExportFormat;
  title: string;
  blocks: Block[];
  preview: MindExportPreview;
  language: Language;
}

export function createExports(deps: ExportsDependencies) {
  const prepare = (mindId: unknown, input: unknown): Prepared => {
    const { format, includeQuestions } = parseExportOptions(input);
    const mind = deps.mind(mindId);
    const language = deps.language();
    const documents = deps.liveDocuments();
    const footnote = (attributes: Record<string, unknown>): Footnote =>
      footnoteOf(attributes as Partial<CitationAttributes>, documents, language);

    const { blocks, questions } = deps.read(mind.id, (fragment) => ({
      blocks: readBlocks(fragment, { includeQuestions, footnote }),
      questions: countQuestions(fragment),
    }));
    const footnotes = footnotesIn(blocks);
    const preview: MindExportPreview = {
      fileName: `${safeFileName(mind.title || translate(language, "mind.untitled"), "Mind")}.${EXTENSIONS[format]}`,
      citations: footnotes.length,
      unverifiedCitations: footnotes.filter((each) => each.unverified).length,
      questions,
    };
    return { format, title: mind.title, blocks, preview, language };
  };

  return {
    preview(mindId: unknown, options: unknown): MindExportPreview {
      return prepare(mindId, options).preview;
    },

    export(mindId: unknown, options: unknown): MindExport {
      const { title, blocks, preview, language, format } = prepare(mindId, options);
      const labels = {
        question: translate(language, "export.question"),
        unverified: translate(language, "export.unverified"),
      };
      const data =
        format === "markdown"
          ? new TextEncoder().encode(renderMarkdown(title, blocks, labels))
          : renderDocx(blocks, {
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
  language: Language,
): Footnote {
  const document = attributes.documentName || translate(language, "export.unnamedDocument");
  const location = citedLocation(attributes, (key, params) => translate(language, key, params));
  return {
    source: location
      ? translate(language, "export.citation.location", { document, location })
      : translate(language, "export.citation.document", { document }),
    unverified: citationState(attributes, documents).check !== "found",
  };
}
