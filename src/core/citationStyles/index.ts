/**
 * Citation styles (#289, ADR-0016): a Mind's Citations and Bibliography in a style, from the
 * cited Documents' reference records (#286). The editor asks for them through
 * `CoreApi.formatCitations`, and the exports format with the same function, so the two agree.
 */
import { type CitationStyle, citationStyle } from "../../shared/citationStyles";
import { displayName } from "../../shared/documentNames";
import { missingDetails } from "../../shared/references";
import type {
  BibliographyEntry,
  CitationFormatRequest,
  CitedWork,
  Document,
  FormattedCitations,
  StyledRun,
} from "../api";
import { type CslItem, cslItemOf, locatorOf, nameOnlyItem } from "./csl";
import { CitationFormatError, type CitationStyleEngine, createWorkerEngine } from "./engine";
import type { FormatJob } from "./format";
import { parseRequest } from "./request";

export { CitationFormatError } from "./engine";

export interface CitationStylesOptions {
  /** resources/citation-styles: the engine, the styles and the locales; none: Numbered only. */
  folder: string | undefined;
  /** The live Document, or undefined for one deleted since it was cited. */
  document(id: string): Document | undefined;
}

/** Answers to the last requests, which a Mind asks for again as it is edited. */
const CACHE_SIZE = 24;

/** A cited work's reference: its record, else its name, and whether that is too thin to cite. */
interface Reference {
  item: CslItem;
  documentId: string | null;
  needsDetails: boolean;
}

/**
 * Whether a record is too thin to cite well: none at all, a work missing what its type needs
 * (`missingDetails`), or one with neither authors nor a date, whatever its type.
 */
function thin(document: Document | undefined): boolean {
  const record = document?.reference;
  if (!record) return true;
  if (missingDetails(record).length > 0) return true;
  return record.fields.authors.length === 0 && record.fields.issued === "";
}

function referenceOf(work: CitedWork, document: Document | undefined): Reference {
  const name = (document && displayName(document)) || work.documentName || "";
  if (!document) {
    return {
      item: nameOnlyItem(`name:${name}`, name),
      documentId: null,
      needsDetails: true,
    };
  }
  const fields = document.reference?.fields;
  const item = fields ? cslItemOf(document.id, fields) : nameOnlyItem(document.id, name);
  // A work with no title is cited by its Document's name rather than by nothing.
  if (!item.title) item.title = name;
  return { item, documentId: document.id, needsDetails: thin(document) };
}

/**
 * Numbered, IncarnaMind's own form: the editor numbers its Citations and the exports give
 * each a footnote naming its Document and Location, so only the Bibliography is made here:
 * each cited Document once, by name, in alphabetical order.
 */
function numbered(request: CitationFormatRequest, references: Map<string, Reference>) {
  const collator = new Intl.Collator(["en", "zh-CN"], { sensitivity: "base", numeric: true });
  const bibliography: BibliographyEntry[] = [...references.values()]
    .map((reference) => ({
      documentId: reference.documentId,
      label: null,
      runs: [{ text: reference.item.title ?? "" }],
      needsDetails: reference.needsDetails,
    }))
    .sort((a, b) => collator.compare(a.runs[0]?.text ?? "", b.runs[0]?.text ?? ""));
  return {
    style: request.style,
    form: "numbered" as const,
    clusters: request.clusters.map(() => ({ runs: [] })),
    bibliography,
    hangingIndent: false,
  };
}

export function createCitationStyles(options: CitationStylesOptions) {
  let engine: CitationStyleEngine | undefined;
  const cache = new Map<string, FormattedCitations>();

  const remember = (key: string, value: FormattedCitations) => {
    cache.delete(key);
    cache.set(key, value);
    for (const oldest of cache.keys()) {
      if (cache.size <= CACHE_SIZE) break;
      cache.delete(oldest);
    }
  };

  async function formatWith(
    style: CitationStyle & { file: string },
    request: CitationFormatRequest,
    references: Map<string, Reference>,
    keyOf: (work: CitedWork) => string,
  ): Promise<FormattedCitations> {
    const job: FormatJob = {
      items: [...references.values()].map((reference) => reference.item),
      citations: request.clusters.map((cluster, index) => ({
        items: cluster.works.map((work) => ({
          id: (references.get(keyOf(work)) as Reference).item.id,
          ...locatorOf(work.location, style.language),
        })),
        // Each cluster is one note, numbered through the Mind.
        noteIndex: index + 1,
      })),
      englishTermsForOthers: style.language === "zh-CN",
    };
    const key = JSON.stringify([style.id, job]);
    const known = cache.get(key);
    if (known) {
      remember(key, known);
      return known;
    }
    if (options.folder === undefined) {
      throw new CitationFormatError("The citation styles aren't installed.", true);
    }
    engine ??= createWorkerEngine(options.folder);
    const { description, output } = await engine.format(style.file, job);
    const byId = new Map(
      [...references.values()].map((reference) => [reference.item.id, reference]),
    );
    const formatted: FormattedCitations = {
      style: style.id,
      form: description.class === "note" ? "note" : "in-text",
      clusters: output.citations.map((runs) => ({
        runs: runs.map(({ item, ...run }) => (item === undefined ? run : { ...run, work: item })),
      })),
      bibliography: output.bibliography.map((entry): BibliographyEntry => {
        const reference = byId.get(entry.id);
        return {
          documentId: reference?.documentId ?? null,
          label: entry.firstField as StyledRun[] | null,
          runs: entry.runs,
          needsDetails: reference?.needsDetails ?? true,
        };
      }),
      hangingIndent: output.hangingIndent,
    };
    remember(key, formatted);
    return formatted;
  }

  return {
    /** Formats a request (see `CoreApi.formatCitations`); `input` is checked first. */
    async format(input: unknown): Promise<FormattedCitations> {
      const request = parseRequest(input);
      const style = citationStyle(request.style);
      const keyOf = (work: CitedWork) =>
        work.documentId !== null && options.document(work.documentId)
          ? work.documentId
          : `name:${work.documentName ?? ""}`;
      const references = new Map<string, Reference>();
      for (const cluster of request.clusters) {
        for (const work of cluster.works) {
          const key = keyOf(work);
          if (references.has(key)) continue;
          const document = work.documentId !== null ? options.document(work.documentId) : undefined;
          references.set(key, referenceOf(work, document));
        }
      }
      if (style.file === null) return numbered(request, references);
      if (references.size === 0) {
        return {
          style: style.id,
          form: style.form,
          clusters: [],
          bibliography: [],
          hangingIndent: false,
        };
      }
      return formatWith({ ...style, file: style.file }, request, references, keyOf);
    },

    close(): void {
      engine?.close();
      engine = undefined;
      cache.clear();
    },
  };
}

export type CitationStyles = ReturnType<typeof createCitationStyles>;
