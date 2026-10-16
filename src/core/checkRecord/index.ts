/**
 * The check record of a Mind (#738): what IncarnaMind checked in it and how, as a PDF to file with
 * the work or a CSV for the User's own records. It is made from what the Mind keeps of its
 * checks (see ./read), on this computer; the host decides where the bytes go, and nothing is sent.
 */
import type * as Y from "yjs";
import { translate } from "../../shared/i18n";
import { mindName } from "../../shared/naming";
import type { PrintPaper } from "../../shared/printDocument";
import type { Printer } from "../adapters";
import {
  type AiUseEntry,
  CHECK_RECORD_FORMATS,
  type ChatProvider,
  type CheckRecordFormat,
  type Document,
  type KeptCitationText,
  type Mind,
} from "../api";
import { describeModel } from "../chatModelNames";
import { InvalidInputError } from "../errors";
import { unifyIdeographs } from "../exports/pdfUnicode";
import { safeFileName } from "../fileNames";
import type { Language } from "../language";
import { renderCsv } from "./csv";
import { renderRecordHtml } from "./html";
import { type RecordItem, readRecord } from "./read";

export interface CheckRecordDependencies {
  /** Throws `NotFoundError` unless the Mind exists. */
  mind(mindId: unknown): Mind;
  /** Reads a Mind's Blocks; the Mind is left as it is. */
  read<T>(mindId: string, look: (blocks: Y.XmlFragment) => T): T;
  /** The Mind's entries in the record of AI use. */
  entries(mindId: string): AiUseEntry[];
  providers(): Promise<ChatProvider[]>;
  documents(): readonly Document[];
  keptCitationTexts(): readonly KeptCitationText[];
  /** A Folder's name; null for one that isn't there. */
  folderName(folderId: string): string | null;
  /** Whether Local mode is on. */
  localMode(): boolean;
  language(): Language;
  /** Prints a PDF; null where this copy can't. */
  printer: Printer | null;
  paper(): PrintPaper;
  now(): string;
}

/** What a record says: the summary and each checked sentence and Citation. */
export interface CheckRecord {
  mind: string;
  /** When it was made, ISO 8601 UTC. */
  exportedAt: string;
  /** The Folder the Mind's checks search; null for every Document. */
  folder: string | null;
  localMode: boolean;
  /** The models in it, each as "Name (Service)", or "Name" alone for one on this computer. */
  models: string[];
  sentences: RecordItem[];
  citations: RecordItem[];
  language: Language;
}

/** A record's file: its name and its bytes. */
export interface CheckRecordFile {
  fileName: string;
  data: Uint8Array;
}

function parseFormat(format: unknown): CheckRecordFormat {
  const known = CHECK_RECORD_FORMATS.find((each) => each === format);
  if (!known) throw new InvalidInputError('The check record format must be "pdf" or "csv".');
  return known;
}

export function createCheckRecord(deps: CheckRecordDependencies) {
  async function build(mindId: unknown): Promise<CheckRecord> {
    const mind = deps.mind(mindId);
    const language = deps.language();
    const providers = await deps.providers();
    const t = (key: Parameters<typeof translate>[1], params?: Parameters<typeof translate>[2]) =>
      translate(language, key, params);
    const read = deps.read(mind.id, (fragment) =>
      readRecord({
        fragment,
        entries: deps.entries(mind.id),
        documents: deps.documents(),
        kept: deps.keptCitationTexts(),
        folderId: mind.folderId,
        t,
        model: (providerId, modelId) => describeModel(providers, providerId, modelId),
      }),
    );
    const named = new Set<string>();
    for (const { model } of [...read.sentences, ...read.citations]) {
      if (model) named.add(model.service ? `${model.name} (${model.service})` : model.name);
    }
    return {
      mind: mindName(mind) || t("mind.untitled"),
      exportedAt: deps.now(),
      folder: mind.folderId === null ? null : deps.folderName(mind.folderId),
      localMode: deps.localMode(),
      models: [...named].sort((a, b) => a.localeCompare(b)),
      ...read,
      language,
    };
  }

  return {
    build,

    async file(mindId: unknown, format: unknown): Promise<CheckRecordFile> {
      const chosen = parseFormat(format);
      const record = await build(mindId);
      const { language } = record;
      const name = safeFileName(record.mind, "Mind");
      const base = translate(language, "checkRecord.fileName", { name });
      if (chosen === "csv") {
        return {
          fileName: `${base}.csv`,
          data: new TextEncoder().encode(renderCsv(record)),
        };
      }
      if (!deps.printer) throw new Error("This copy of IncarnaMind can't make PDFs.");
      const printed = await deps.printer.print({
        language,
        paper: deps.paper(),
        title: base,
        body: renderRecordHtml(record),
        notes: [],
        footer: translate(language, "checkRecord.note"),
      });
      return { fileName: `${base}.pdf`, data: unifyIdeographs(printed) };
    },
  };
}
