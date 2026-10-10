/**
 * The hard tier's library (eval/hard/library.json, eval/hard/README.md):
 * the manifest's checks of sources, licences and files, the committed
 * manifest itself, and fetching into the cache with checksums, without the
 * network.
 */
import { createHash } from "node:crypto";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { afterEach, describe, expect, test } from "vitest";
import { fetchLibrary, pauseFor, USER_AGENT } from "../../eval/hard/lib/fetch";
import {
  cachePathOf,
  downloadBytes,
  fileNameOf,
  type HardDocument,
  LICENCES,
  loadManifest,
  MAX_DOWNLOAD_BYTES,
  type Manifest,
  manifestProblems,
  siblingsOf,
} from "../../eval/hard/lib/manifest";
import { buildZip } from "../helpers/zip";

const ROOT = fileURLToPath(new URL("../..", import.meta.url));
const sha256 = (bytes: Uint8Array | string) => createHash("sha256").update(bytes).digest("hex");

const document = (key: string, extra: Partial<HardDocument> = {}): HardDocument => ({
  key,
  title: key,
  domain: "reports",
  language: "en",
  format: "pdf",
  source: "open",
  url: `https://example.org/${key}.pdf`,
  sha256: sha256(key),
  bytes: 10,
  ...extra,
});

const manifest = (documents: HardDocument[], extra: Partial<Manifest> = {}): Manifest => ({
  version: 1,
  description: "",
  sources: {
    open: {
      name: "An open publisher",
      licence: "CC-BY-4.0",
      terms: "https://example.org/terms",
      permission: "You may share and adapt.",
      checked: "2026-10-10",
    },
  },
  archives: [],
  documents,
  ...extra,
});

describe("The hard tier's manifest", () => {
  test("the committed library is sound: known licences, checksums, under 3 GB, no source that wants contact details", () => {
    const library = loadManifest(ROOT);
    expect(library.documents.length).toBeGreaterThanOrEqual(300);
    expect(library.documents.length).toBeLessThanOrEqual(500);
    expect(downloadBytes(library)).toBeLessThanOrEqual(MAX_DOWNLOAD_BYTES);
    for (const source of Object.values(library.sources))
      expect(LICENCES).toHaveProperty(source.licence);
    const urls = [
      ...library.documents.flatMap((each) => (each.url ? [each.url] : [])),
      ...library.archives.map((each) => each.url),
    ];
    for (const url of urls) {
      const host = new URL(url).hostname;
      // SEC EDGAR's fair-access policy and Wikimedia's API ask for contact details in requests.
      expect(host, url).not.toMatch(/(^|\.)sec\.gov$|(^|\.)wikipedia\.org$|(^|\.)wikimedia\.org$/);
    }
  });

  test("the committed library covers every domain, both languages, and Office formats beside PDFs", () => {
    const library = loadManifest(ROOT);
    const domains = new Set(library.documents.map((each) => each.domain));
    expect([...domains].sort()).toEqual(
      ["contracts", "filings", "manuals", "medical", "papers", "reports"].sort(),
    );
    expect(library.documents.filter((each) => each.language === "zh").length).toBeGreaterThan(30);
    const formats = new Set(library.documents.map((each) => each.format));
    for (const format of ["pdf", "docx", "xlsx", "pptx"]) expect(formats).toContain(format);
  });

  test("a sound manifest has no problems", () => {
    expect(manifestProblems(manifest([document("a"), document("b")]))).toEqual([]);
  });

  test("a source needs a known licence, its terms' address and their words", () => {
    const problems = manifestProblems(
      manifest([document("a")], {
        sources: {
          open: {
            name: "Somewhere",
            licence: "all-rights-reserved" as never,
            terms: "http://example.org/terms",
            permission: " ",
            checked: "yesterday",
          },
        },
      }),
    );
    expect(problems).toEqual([
      'source open: unknown licence "all-rights-reserved".',
      'source open: terms must be an https URL (a file in the repository for "repository").',
      "source open: no permission quoted from its terms.",
      "source open: checked must be a date (YYYY-MM-DD).",
    ]);
  });

  test("a Document in the repository names the file that records its terms", () => {
    const local = manifest(
      [document("a", { url: undefined, path: "data/a.pdf", source: "repo" })],
      {
        sources: {
          repo: {
            name: "This repository",
            licence: "repository",
            terms: "eval/README.md",
            permission: "Already here.",
            checked: "2026-10-10",
          },
        },
      },
    );
    expect(manifestProblems(local)).toEqual([]);
  });

  test("every Document is fetched, taken from an archive or read from the repository, with its checksum", () => {
    const problems = manifestProblems(
      manifest([
        document("both", { path: "data/both.pdf" }),
        document("plain", { url: "http://example.org/plain.pdf" }),
        document("archived", { url: undefined, archive: "missing" }),
        document("hashless", { sha256: "abc" }),
        document("twin", { sha256: sha256("plain") }),
      ]),
    );
    expect(problems).toEqual([
      "document both: needs exactly one of url, archive (with member) or path.",
      "document plain: url must be https.",
      'document archived: unknown archive "missing".',
      "document archived: an archived Document needs its member path.",
      "document hashless: sha256 must be 64 hex digits.",
      "document twin: the same file as plain.",
    ]);
  });

  test("keys and file names are unique, groups have two or more versions, translations are in the other language", () => {
    const problems = manifestProblems(
      manifest([
        document("a"),
        document("a"),
        document("b", { title: "a" }),
        document("c", { group: "lonely" }),
        document("d", { translationOf: "a" }),
        document("e", { translationOf: "nowhere", language: "zh" }),
      ]),
    );
    expect(problems).toContain("document a: listed twice.");
    expect(problems).toContain("document b: another Document has the file name a.pdf.");
    expect(problems).toContain("group lonely: has only one Document.");
    expect(problems).toContain("document d: a translation must be in the other language.");
    expect(problems).toContain("document e: translates an unknown Document.");
  });

  test("an unused source, and a library over 3 GB, are problems", () => {
    const problems = manifestProblems(
      manifest([document("big", { bytes: MAX_DOWNLOAD_BYTES + 1 })], {
        sources: {
          ...manifest([]).sources,
          unused: {
            name: "Unused",
            licence: "OGL-UK-3.0",
            terms: "https://example.org/ogl",
            permission: "Copy and publish.",
            checked: "2026-10-10",
          },
        },
      }),
    );
    expect(problems).toContain("source unused: no Document or archive uses it.");
    expect(problems.some((each) => each.startsWith("the library downloads"))).toBe(true);
  });

  test("file names are the Documents' titles, made safe on every platform", () => {
    expect(fileNameOf({ title: "Report: 2024/25 <draft>?", format: "pdf" })).toBe(
      "Report- 2024-25 -draft--.pdf",
    );
    expect(cachePathOf("/cache", document("x", { domain: "medical" }))).toBe(
      join("/cache", "hard", "files", "medical", "x.pdf"),
    );
  });

  test("a Document's siblings are the other versions in its group", () => {
    const library = manifest([
      document("v1", { group: "g" }),
      document("v2", { group: "g" }),
      document("other"),
    ]);
    expect(siblingsOf(library, "v1").map((each) => each.key)).toEqual(["v2"]);
    expect(siblingsOf(library, "other")).toEqual([]);
  });
});

describe("Fetching the hard tier's library", () => {
  let cache: string;
  afterEach(async () => {
    if (cache) await rm(cache, { recursive: true, force: true });
  });

  /** A fake network serving `files` by URL, recording each request's headers. */
  function network(files: Record<string, Uint8Array | string>) {
    const requests: { url: string; headers: Record<string, string> }[] = [];
    const fetch = (async (url: string | URL, init?: RequestInit) => {
      requests.push({
        url: String(url),
        headers: { ...(init?.headers as Record<string, string>) },
      });
      const body = files[String(url)];
      return body === undefined
        ? new Response("missing", { status: 404 })
        : new Response(typeof body === "string" ? body : Buffer.from(body));
    }) as typeof globalThis.fetch;
    return { fetch, requests };
  }

  test("the User-Agent names the evaluation and no person, and requests send nothing else", async () => {
    expect(USER_AGENT).toBe("IncarnaMind document evaluation");
    expect(USER_AGENT).not.toMatch(/@|https?:|\d/);
    cache = await mkdtemp(join(tmpdir(), "hard-fetch-"));
    const { fetch, requests } = network({ "https://example.org/a.pdf": "a" });
    await fetchLibrary(manifest([document("a")]), {
      root: ROOT,
      cacheDir: cache,
      log: () => {},
      fetch,
      sleep: async () => {},
    });
    expect(requests).toEqual([
      { url: "https://example.org/a.pdf", headers: { "User-Agent": USER_AGENT } },
    ]);
  });

  test("fetches each file once, checks it, and reads it from the cache afterwards", async () => {
    cache = await mkdtemp(join(tmpdir(), "hard-fetch-"));
    const library = manifest([document("a"), document("b")]);
    const { fetch, requests } = network({
      "https://example.org/a.pdf": "a",
      "https://example.org/b.pdf": "b",
    });
    const options = { root: ROOT, cacheDir: cache, log: () => {}, fetch, sleep: async () => {} };
    const first = await fetchLibrary(library, options);
    expect(first.documents.map((each) => each.status)).toEqual(["ready", "ready"]);
    expect(first.fetchedBytes).toBe(2);
    expect(await readFile(first.documents[0]?.path as string, "utf8")).toBe("a");
    const again = await fetchLibrary(library, options);
    expect(again.fetchedBytes).toBe(0);
    expect(requests).toHaveLength(2);
  });

  test("a file that changed at its source, or can't be fetched, is left out with the reason", async () => {
    cache = await mkdtemp(join(tmpdir(), "hard-fetch-"));
    const { fetch, requests } = network({ "https://example.org/changed.pdf": "something else" });
    const result = await fetchLibrary(manifest([document("changed"), document("gone")]), {
      root: ROOT,
      cacheDir: cache,
      log: () => {},
      fetch,
      sleep: async () => {},
    });
    expect(result.documents.map(({ key, status, path }) => ({ key, status, path }))).toEqual([
      { key: "changed", status: "changed", path: null },
      { key: "gone", status: "unavailable", path: null },
    ]);
    expect(result.documents[1]?.reason).toMatch(/HTTP 404/);
    // A refused file isn't asked for again.
    expect(requests.filter((each) => each.url.endsWith("gone.pdf"))).toHaveLength(1);
  });

  test("takes Documents out of an archive, fetched once and checked, and checks each member", async () => {
    cache = await mkdtemp(join(tmpdir(), "hard-fetch-"));
    const zip = buildZip([
      { name: "set/one.pdf", data: "one" },
      { name: "set/two.pdf", data: "two" },
    ]);
    const library = manifest(
      [
        document("one", {
          url: undefined,
          archive: "set",
          member: "set/one.pdf",
          sha256: sha256("one"),
        }),
        document("two", {
          url: undefined,
          archive: "set",
          member: "set/two.pdf",
          sha256: sha256("2"),
        }),
      ],
      {
        archives: [
          {
            key: "set",
            source: "open",
            url: "https://example.org/set.zip",
            sha256: sha256(zip),
            bytes: zip.length,
          },
        ],
      },
    );
    const { fetch } = network({ "https://example.org/set.zip": zip });
    const result = await fetchLibrary(library, {
      root: ROOT,
      cacheDir: cache,
      log: () => {},
      fetch,
      sleep: async () => {},
    });
    expect(result.documents.map((each) => each.status)).toEqual(["ready", "changed"]);
    expect(await readFile(result.documents[0]?.path as string, "utf8")).toBe("one");
  });

  test("a Document in the repository is read where it is, if it is the version recorded", async () => {
    cache = await mkdtemp(join(tmpdir(), "hard-fetch-"));
    await writeFile(join(cache, "kept.pdf"), "kept");
    const local = (hash: string) =>
      manifest([document("kept", { url: undefined, path: "kept.pdf", sha256: hash })]);
    const options = { root: cache, cacheDir: cache, log: () => {}, sleep: async () => {} };
    expect((await fetchLibrary(local(sha256("kept")), options)).documents[0]).toMatchObject({
      status: "ready",
      path: join(cache, "kept.pdf"),
    });
    expect((await fetchLibrary(local(sha256("other")), options)).documents[0]?.status).toBe(
      "changed",
    );
  });

  test("waits 3 seconds between requests to arXiv, as its terms ask, and less elsewhere", async () => {
    expect(pauseFor("export.arxiv.org")).toBe(3000);
    expect(pauseFor("arxiv.org")).toBe(3000);
    expect(pauseFor("documents.un.org")).toBeLessThan(3000);
    cache = await mkdtemp(join(tmpdir(), "hard-fetch-"));
    const waits: number[] = [];
    const { fetch } = network({
      "https://export.arxiv.org/pdf/1": "1",
      "https://export.arxiv.org/pdf/2": "2",
    });
    await fetchLibrary(
      manifest([
        document("p1", { url: "https://export.arxiv.org/pdf/1", sha256: sha256("1") }),
        document("p2", { url: "https://export.arxiv.org/pdf/2", sha256: sha256("2") }),
      ]),
      {
        root: ROOT,
        cacheDir: cache,
        log: () => {},
        fetch,
        sleep: async (ms) => {
          waits.push(ms);
        },
      },
    );
    expect(waits).toHaveLength(1);
    expect(waits[0]).toBeGreaterThan(2900);
  });
});
