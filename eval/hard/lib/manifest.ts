/**
 * The hard tier's library (eval/hard/README.md): public Documents fetched at
 * evaluation time, never committed, described by eval/hard/library.json.
 *
 * Each Document comes from a source whose terms allow downloading it for an
 * evaluation, recorded with the terms' address and their own words. Each is
 * fetched from its URL, taken out of an archive that is (CUAD's contracts), or
 * read from the repository (the gating set's Documents), and checked against
 * its SHA-256 before it is added. Nothing here touches the network: see
 * ./fetch.
 */
import { readFileSync } from "node:fs";
import { extname, join } from "node:path";

/** The library's domains, as the report groups them. */
export const DOMAINS = ["contracts", "filings", "papers", "reports", "manuals", "medical"] as const;
export type Domain = (typeof DOMAINS)[number];

export const DOMAIN_LABELS: Record<Domain, string> = {
  contracts: "Contracts",
  filings: "Company reports and filings",
  papers: "Academic papers",
  reports: "Government and international reports",
  manuals: "Technical manuals",
  medical: "Medical and regulatory",
};

export type HardLanguage = "en" | "zh";
export const LANGUAGES: readonly HardLanguage[] = ["en", "zh"];

/** File formats the app reads now (ADR-0011), by extension. */
export const FORMATS = ["pdf", "docx", "pptx", "xlsx", "csv", "md", "txt"] as const;
export type HardFormat = (typeof FORMATS)[number];

/**
 * The licences and terms the library's sources may have: each lets the User
 * download the Documents and read them for an evaluation on their own
 * computer, which is all the run does. Several allow non-commercial use only:
 * IncarnaMind is free and open source (ADR-0002), so its evaluation is.
 * Nothing is redistributed: the files stay in the evaluation's cache.
 */
export const LICENCES = {
  "CC-BY-4.0": { nonCommercial: false, name: "Creative Commons Attribution 4.0" },
  "CC-BY-SA-4.0": { nonCommercial: false, name: "Creative Commons Attribution-ShareAlike 4.0" },
  "CC-BY-3.0-IGO": { nonCommercial: false, name: "Creative Commons Attribution 3.0 IGO" },
  "CC-BY-NC-4.0": { nonCommercial: true, name: "Creative Commons Attribution-NonCommercial 4.0" },
  "CC-BY-NC-3.0-IGO": {
    nonCommercial: true,
    name: "Creative Commons Attribution-NonCommercial 3.0 IGO",
  },
  "CC-BY-NC-SA-3.0-IGO": {
    nonCommercial: true,
    name: "Creative Commons Attribution-NonCommercial-ShareAlike 3.0 IGO",
  },
  "public-domain-us": {
    nonCommercial: false,
    name: "Work of the US federal government, public domain (17 U.S.C. 105)",
  },
  "OGL-UK-3.0": { nonCommercial: false, name: "Open Government Licence v3.0" },
  "PRC-copyright-law-art-5": {
    nonCommercial: false,
    name: "Official document of a Chinese state organ, outside copyright (Copyright Law of the PRC, art. 5)",
  },
  "terms-non-commercial": {
    nonCommercial: true,
    name: "The publisher's terms allow downloading for personal or non-commercial use",
  },
  "arxiv-personal-research": {
    nonCommercial: false,
    name: "arXiv's terms allow retrieving e-prints for personal use or research",
  },
  repository: {
    nonCommercial: false,
    name: "Already in this repository, under the terms recorded there",
  },
} as const;
export type LicenceId = keyof typeof LICENCES;

export interface HardSource {
  name: string;
  licence: LicenceId;
  /** The terms or licence page, checked by hand; for "repository", the file in it that records them. */
  terms: string;
  /** What the terms say about downloading, in their own words. */
  permission: string;
  /** When the terms were read, YYYY-MM-DD. */
  checked: string;
}

/** An archive Documents are taken out of: fetched once, kept in the cache. */
export interface HardArchive {
  key: string;
  source: string;
  url: string;
  sha256: string;
  bytes: number;
}

export interface HardDocument {
  key: string;
  /** Its file's name, without the extension: what the app shows as the Document's name. */
  title: string;
  domain: Domain;
  language: HardLanguage;
  format: HardFormat;
  source: string;
  /** Fetched from here... */
  url?: string;
  /** ...or taken out of an archive, by its path in it... */
  archive?: string;
  member?: string;
  /** ...or read from the repository, relative to its root. */
  path?: string;
  sha256: string;
  bytes: number;
  /** A PDF's page count, for planning; null for other formats. */
  pages?: number | null;
  /** Shared by versions or translations of one Document (near-duplicate and cross-lingual Questions). */
  group?: string;
  /** Which version it is in its group, e.g. "2024" or "2024Q2". */
  edition?: string;
  /** The key of the Document it translates, in the other language. */
  translationOf?: string;
}

export interface Manifest {
  version: number;
  description: string;
  sources: Record<string, HardSource>;
  archives: HardArchive[];
  documents: HardDocument[];
}

/** The manifest, relative to the repository root. */
export const MANIFEST = "eval/hard/library.json";

/** The most the library may download, archives included. */
export const MAX_DOWNLOAD_BYTES = 3_000_000_000;

const HEX64 = /^[0-9a-f]{64}$/;
const KEY = /^[a-z0-9][a-z0-9-]*$/;

/** Characters a file name can't hold on macOS, Windows or Linux. */
const UNSAFE_NAME = /[/\\:*?"<>|]/g;

/** The Document's file name in the cache: its title, made safe, and its format's extension. */
export const fileNameOf = (document: Pick<HardDocument, "title" | "format">): string =>
  `${document.title.replace(UNSAFE_NAME, "-").trim()}.${document.format}`;

/** Where a fetched Document is kept: by domain, under its own title. */
export const cachePathOf = (cacheDir: string, document: HardDocument): string =>
  join(cacheDir, "hard", "files", document.domain, fileNameOf(document));

/** What a manifest downloads: its archives, and the Documents with a URL. */
export function downloadBytes(manifest: Pick<Manifest, "archives" | "documents">): number {
  return (
    manifest.archives.reduce((sum, archive) => sum + archive.bytes, 0) +
    manifest.documents
      .filter((document) => document.url)
      .reduce((sum, document) => sum + document.bytes, 0)
  );
}

/**
 * Every problem with a manifest: what a source, an archive or a Document
 * lacks, or breaks. Empty when it is sound.
 */
export function manifestProblems(manifest: Manifest): string[] {
  const problems: string[] = [];
  const problem = (message: string) => problems.push(message);
  if (manifest.version !== 1) problem(`version must be 1, not ${manifest.version}.`);

  for (const [id, source] of Object.entries(manifest.sources ?? {})) {
    if (!(source.licence in LICENCES))
      problem(`source ${id}: unknown licence "${source.licence}".`);
    const termsAt =
      source.licence === "repository"
        ? !/^[a-z]+:/.test(source.terms ?? "") && Boolean(source.terms?.trim())
        : /^https:\/\//.test(source.terms ?? "");
    if (!termsAt) {
      problem(
        `source ${id}: terms must be an https URL (a file in the repository for "repository").`,
      );
    }
    if (!source.permission?.trim()) problem(`source ${id}: no permission quoted from its terms.`);
    if (!/^\d{4}-\d{2}-\d{2}$/.test(source.checked ?? "")) {
      problem(`source ${id}: checked must be a date (YYYY-MM-DD).`);
    }
    if (!source.name?.trim()) problem(`source ${id}: no name.`);
  }
  const used = new Set<string>();
  const sourceOf = (owner: string, id: string) => {
    if (!manifest.sources?.[id]) problem(`${owner}: unknown source "${id}".`);
    used.add(id);
  };

  const archives = new Set<string>();
  for (const archive of manifest.archives ?? []) {
    if (!KEY.test(archive.key)) problem(`archive ${archive.key}: the key must be lowercase kebab.`);
    if (archives.has(archive.key)) problem(`archive ${archive.key}: listed twice.`);
    archives.add(archive.key);
    sourceOf(`archive ${archive.key}`, archive.source);
    if (!/^https:\/\//.test(archive.url)) problem(`archive ${archive.key}: url must be https.`);
    if (!HEX64.test(archive.sha256))
      problem(`archive ${archive.key}: sha256 must be 64 hex digits.`);
    if (!(archive.bytes > 0)) problem(`archive ${archive.key}: bytes must be positive.`);
  }

  const keys = new Set<string>();
  const files = new Set<string>();
  const hashes = new Map<string, string>();
  for (const document of manifest.documents ?? []) {
    const at = `document ${document.key}`;
    if (!KEY.test(document.key ?? "")) problem(`${at}: the key must be lowercase kebab.`);
    if (keys.has(document.key)) problem(`${at}: listed twice.`);
    keys.add(document.key);
    if (!document.title?.trim()) problem(`${at}: no title.`);
    if (!DOMAINS.includes(document.domain)) problem(`${at}: unknown domain "${document.domain}".`);
    if (!LANGUAGES.includes(document.language)) problem(`${at}: unknown language.`);
    if (!FORMATS.includes(document.format)) problem(`${at}: unknown format "${document.format}".`);
    sourceOf(at, document.source);
    const where = [document.url, document.archive, document.path].filter(Boolean).length;
    if (where !== 1) problem(`${at}: needs exactly one of url, archive (with member) or path.`);
    if (document.url !== undefined && !/^https:\/\//.test(document.url)) {
      problem(`${at}: url must be https.`);
    }
    if (document.archive !== undefined) {
      if (!archives.has(document.archive)) problem(`${at}: unknown archive "${document.archive}".`);
      if (!document.member?.trim()) problem(`${at}: an archived Document needs its member path.`);
    }
    const file = document.member ?? document.path;
    if (file !== undefined && extname(file).toLowerCase() !== `.${document.format}`) {
      problem(`${at}: its file isn't a .${document.format}.`);
    }
    if (!HEX64.test(document.sha256 ?? "")) problem(`${at}: sha256 must be 64 hex digits.`);
    const twin = hashes.get(document.sha256);
    if (twin) problem(`${at}: the same file as ${twin}.`);
    hashes.set(document.sha256, document.key);
    if (!(document.bytes > 0)) problem(`${at}: bytes must be positive.`);
    if (document.domain && document.title) {
      // Unique across the library, not only in its domain's folder: Answers name Documents by it.
      const name = fileNameOf(document).toLowerCase();
      if (files.has(name)) problem(`${at}: another Document has the file name ${name}.`);
      files.add(name);
    }
  }

  const byKey = new Map((manifest.documents ?? []).map((document) => [document.key, document]));
  const groups = new Map<string, HardDocument[]>();
  for (const document of manifest.documents ?? []) {
    if (document.group)
      groups.set(document.group, [...(groups.get(document.group) ?? []), document]);
    if (document.translationOf !== undefined) {
      const original = byKey.get(document.translationOf);
      if (!original) problem(`document ${document.key}: translates an unknown Document.`);
      else if (original.language === document.language) {
        problem(`document ${document.key}: a translation must be in the other language.`);
      }
    }
  }
  for (const [group, members] of groups) {
    if (members.length < 2) problem(`group ${group}: has only one Document.`);
  }
  for (const id of Object.keys(manifest.sources ?? {})) {
    if (!used.has(id)) problem(`source ${id}: no Document or archive uses it.`);
  }
  const bytes = downloadBytes(manifest);
  if (bytes > MAX_DOWNLOAD_BYTES) {
    problem(`the library downloads ${bytes} bytes, more than ${MAX_DOWNLOAD_BYTES}.`);
  }
  return problems;
}

/** Reads the manifest and fails, saying why, if it isn't sound. */
export function loadManifest(root: string, source = MANIFEST): Manifest {
  const manifest = JSON.parse(readFileSync(join(root, source), "utf8")) as Manifest;
  const problems = manifestProblems(manifest);
  if (problems.length > 0) throw new Error(`${source}:\n- ${problems.join("\n- ")}`);
  return manifest;
}

/** The other Documents of a Document's group: its other versions or translations. */
export function siblingsOf(manifest: Pick<Manifest, "documents">, key: string): HardDocument[] {
  const document = manifest.documents.find((each) => each.key === key);
  if (!document?.group) return [];
  return manifest.documents.filter(
    (each) => each.group === document.group && each.key !== document.key,
  );
}
