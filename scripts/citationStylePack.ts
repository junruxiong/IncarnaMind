/**
 * Writes resources/citation-styles/styles.pack (see src/core/citationStyles/pack.ts): every style
 * of the CSL styles repository and every locale of its locales repository, at the commits pinned
 * below, unmodified. `node --import ./scripts/tsResolve.mjs scripts/citationStylePack.ts`, or
 * with `--check` to build into a temporary file and fail unless it is byte for byte the
 * committed pack. It runs on a maintainer's computer and in CI, never in the app.
 *
 * Each independent style is parsed by the app's own engine (resources/citation-styles/
 * engine.wasm) and formats one reference; a style that fails either is left out, and so are the
 * journal styles that follow it, each named in the log. A journal style is kept as an alias of
 * the style it follows: its id, title, fields and default-locale. The common styles share the
 * first chunk, so the styles most Minds use open with one small read.
 */
import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { constants, brotliCompressSync } from "node:zlib";
import { CslEngine } from "../src/core/citationStyles/wasm.ts";
import {
  type CitationFormat,
  PACK_MAGIC,
  type PackIndex,
  type PackSlot,
  type PackStyle,
} from "../src/core/citationStyles/pack.ts";

const STYLES_COMMIT = "6ba2e9c6aa6e0dab87d3a20b833ed8ff6500f698";
const LOCALES_COMMIT = "a89adece41013402236e2c9020972d7e931fbab8";

/** About this many files to a chunk: small enough to open quickly, large enough to compress well. */
const CHUNK_FILES = 50;

/** The common list's styles, which share the first chunk. */
const COMMON = [
  "apa",
  "modern-language-association",
  "harvard-cite-them-right",
  "chicago-notes-bibliography",
  "chicago-author-date",
  "ieee",
  "nlm-citation-sequence",
  "china-national-standard-gb-t-7714-2015-numeric",
  "china-national-standard-gb-t-7714-2015-author-date",
  "china-national-standard-gb-t-7714-2025-numeric",
  "china-national-standard-gb-t-7714-2025-author-date",
  "nature",
];

const ROOT = resolve(import.meta.dirname, "..");
const OUT = join(ROOT, "resources/citation-styles/styles.pack");
const ENGINE = join(ROOT, "resources/citation-styles/engine.wasm");
const LICENCE = join(ROOT, "resources/notices/cc-by-sa-3.0.txt");

const FORMATS: readonly CitationFormat[] = ["author-date", "author", "numeric", "label", "note"];

/** A repository at a commit, downloaded and unpacked into `into`; the folder it unpacks to. */
function fetchRepository(name: string, commit: string, into: string): string {
  const archive = join(into, `${name}.tar.gz`);
  execFileSync("curl", [
    "--fail",
    "--silent",
    "--location",
    "--output",
    archive,
    `https://codeload.github.com/citation-style-language/${name}/tar.gz/${commit}`,
  ]);
  execFileSync("tar", ["-xzf", archive, "-C", into]);
  return join(into, `${name}-${commit}`);
}

const ENTITIES: Record<string, string> = { amp: "&", lt: "<", gt: ">", quot: '"', apos: "'" };

/** Text from XML, its entities decoded and its white space tidied. */
const tidy = (text: string) =>
  text
    .replace(/&(#x[0-9a-f]+|#\d+|[a-z]+);/gi, (whole, name: string) =>
      name.startsWith("#x")
        ? String.fromCodePoint(Number.parseInt(name.slice(2), 16))
        : name.startsWith("#")
          ? String.fromCodePoint(Number(name.slice(1)))
          : (ENTITIES[name] ?? whole),
    )
    .replace(/\s+/g, " ")
    .trim();

/** An element's attributes, from its start tag's text. */
function attributes(tag: string): Record<string, string> {
  const found: Record<string, string> = {};
  for (const [, name, double, single] of tag.matchAll(/([\w:-]+)\s*=\s*(?:"([^"]*)"|'([^']*)')/g)) {
    found[name as string] = tidy(double ?? single ?? "");
  }
  return found;
}

/** The text of the first element named `name` in `xml`, or "". */
const first = (xml: string, name: string) =>
  tidy(new RegExp(`<${name}>([\\s\\S]*?)</${name}>`).exec(xml)?.[1] ?? "");

/** The names of the people in `xml` in a role: "author", "contributor", "translator". */
const people = (xml: string, role: string) =>
  [...xml.matchAll(new RegExp(`<${role}>([\\s\\S]*?)</${role}>`, "g"))]
    .map(([, person]) => first(person as string, "name"))
    .filter(Boolean);

interface Read {
  style: PackStyle;
  text: string;
  people: { authors: string[]; contributors: string[] };
  self: string;
}

/**
 * A style's file as the pack lists it, from its `<style>` start tag and its `<info>`. The CSL
 * repository's tests keep each file's `<info>` to this shape, so reading it doesn't need a
 * whole XML parser.
 */
function readStyle(id: string, text: string): Read & { parent: string | null } {
  const root = attributes(/<style\b([^>]*)>/.exec(text)?.[1] ?? "");
  const info = /<info>([\s\S]*?)<\/info>/.exec(text)?.[1] ?? "";
  const categories = [...info.matchAll(/<category\b([^>]*?)\/?>/g)].map(([, tag]) =>
    attributes(tag as string),
  );
  const links = [...info.matchAll(/<link\b([^>]*?)\/?>/g)].map(([, tag]) => attributes(tag as string));
  const format = categories.map((each) => each["citation-format"]).find(Boolean);
  const fields = categories.flatMap((each) => (each.field ? [each.field] : []));
  const parentLink = links.find((link) => link.rel === "independent-parent")?.href;
  const title = first(info, "title");
  const short = first(info, "title-short");
  const style: PackStyle = {
    id,
    title: title || id,
    format: FORMATS.find((each) => each === format) ?? "author-date",
  };
  if (short) style.short = short;
  if (root.class === "note") style.note = true;
  if (fields.length > 0) style.fields = fields;
  if (root["default-locale"]) style.locale = root["default-locale"];
  return {
    style,
    text,
    people: { authors: people(info, "author"), contributors: people(info, "contributor") },
    self: links.find((link) => link.rel === "self")?.href ?? "",
    parent: parentLink ? (parentLink.split("/").pop() ?? null) : null,
  };
}

/** Why the engine can't use a style, or null when it parses and formats a reference. */
function engineProblem(engine: CslEngine, text: string, locales: string[]): string | null {
  try {
    const { handle } = engine.parse(text, locales);
    try {
      engine.render(handle, {
        items: [
          {
            id: "a",
            type: "article-journal",
            title: "Tides and the Moon",
            author: [{ family: "Bowler", given: "Diana" }],
            issued: { "date-parts": [[2010]] },
            "container-title": "Ocean Review",
          },
        ],
        citations: [{ items: [{ id: "a", locator: "4", label: "page" }], noteIndex: 1 }],
      });
    } finally {
      engine.drop(handle);
    }
    return null;
  } catch (error) {
    return error instanceof Error ? error.message : String(error);
  }
}

const compress = (json: string) =>
  brotliCompressSync(Buffer.from(json, "utf8"), {
    params: {
      [constants.BROTLI_PARAM_MODE]: constants.BROTLI_MODE_TEXT,
      [constants.BROTLI_PARAM_QUALITY]: 11,
      [constants.BROTLI_PARAM_LGWIN]: 24,
    },
  });

function notice(kept: readonly Read[], locales: readonly { code: string; translators: string[] }[]) {
  const lines = [
    "IncarnaMind formats Citations and Bibliographies with styles and locales from the Citation Style",
    "Language (CSL) project: https://citationstyles.org/",
    "",
    "They are shipped unmodified, in one file of compressed chunks in the app's resources",
    "(citation-styles/styles.pack), taken from https://github.com/citation-style-language/styles at",
    `commit ${STYLES_COMMIT} and https://github.com/citation-style-language/locales at commit`,
    `${LOCALES_COMMIT}. Each is released under the Creative Commons Attribution-ShareAlike 3.0`,
    "Unported licence (https://creativecommons.org/licenses/by-sa/3.0/), whose text follows the",
    "list. Each file's metadata, with its authors, contributors and translators, is kept as it is;",
    "they are named here too. Shipped this way, the files and IncarnaMind form a Collection under",
    "that licence (section 1(b)): IncarnaMind's own code stays under the Apache License 2.0.",
    "",
    "Styles:",
    "",
  ];
  for (const { style, people, self } of kept) {
    lines.push(`- ${style.title} (${style.id}${self ? `, ${self}` : ""})`);
    if (people.authors.length > 0) lines.push(`  Authors: ${people.authors.join("; ")}`);
    if (people.contributors.length > 0) {
      lines.push(`  Contributors: ${people.contributors.join("; ")}`);
    }
  }
  lines.push("", "Locales:", "");
  for (const { code, translators } of locales) {
    lines.push(`- Locale ${code}`);
    if (translators.length > 0) lines.push(`  Translators: ${translators.join("; ")}`);
  }
  lines.push("", "-".repeat(96), "", readFileSync(LICENCE, "utf8").trimEnd(), "");
  return lines.join("\n");
}

function build(out: string) {
  const work = mkdtempSync(join(tmpdir(), "incarnamind-csl-"));
  try {
    const styles = fetchRepository("styles", STYLES_COMMIT, work);
    const localeFolder = fetchRepository("locales", LOCALES_COMMIT, work);

    const localeFiles = readdirSync(localeFolder)
      .filter((name) => /^locales-[a-zA-Z-]+\.xml$/.test(name))
      .sort();
    const localeText = new Map(
      localeFiles.map((name) => [
        name.slice("locales-".length, -".xml".length),
        readFileSync(join(localeFolder, name), "utf8"),
      ]),
    );
    const localesFor = (locale: string | undefined) => {
      const wanted = new Set(["en-US", "zh-CN"]);
      if (locale && localeText.has(locale)) wanted.add(locale);
      return [...wanted].map((code) => localeText.get(code) as string);
    };

    const engine = new CslEngine(new WebAssembly.Module(readFileSync(ENGINE)));
    let live = engine;
    const independent: Read[] = [];
    const left: string[] = [];
    for (const name of readdirSync(styles).filter((file) => file.endsWith(".csl")).sort()) {
      const id = name.slice(0, -".csl".length);
      const read = readStyle(id, readFileSync(join(styles, name), "utf8"));
      const problem = engineProblem(live, read.text, localesFor(read.style.locale));
      if (problem) {
        left.push(`${id}: ${problem}`);
        // A panic leaves the instance unusable.
        live = new CslEngine(new WebAssembly.Module(readFileSync(ENGINE)));
        continue;
      }
      independent.push(read);
    }
    const kept = new Set(independent.map((read) => read.style.id));
    const dependent: Read[] = [];
    const dependentFolder = join(styles, "dependent");
    for (const name of readdirSync(dependentFolder).filter((file) => file.endsWith(".csl")).sort()) {
      const id = name.slice(0, -".csl".length);
      const read = readStyle(id, readFileSync(join(dependentFolder, name), "utf8"));
      if (!read.parent || !kept.has(read.parent)) {
        left.push(`${id}: follows ${read.parent ?? "no style"}, which isn't kept`);
        continue;
      }
      // A journal's format is its own category; with none, the style it follows says.
      dependent.push({ ...read, style: { ...read.style, parent: read.parent } });
    }

    const common = COMMON.map((id) => {
      const found = independent.find((read) => read.style.id === id);
      if (!found) throw new Error(`The common style ${id} isn't kept.`);
      return found;
    });
    const rest = independent.filter((read) => !COMMON.includes(read.style.id));
    const groups: Read[][] = [common];
    for (let at = 0; at < rest.length; at += CHUNK_FILES) groups.push(rest.slice(at, at + CHUNK_FILES));

    const chunks: Buffer[] = [];
    const add = (files: readonly string[]): number => {
      chunks.push(compress(JSON.stringify(files)));
      return chunks.length - 1;
    };
    const entries: PackStyle[] = [];
    for (const group of groups) {
      const chunk = add(group.map((read) => read.text));
      group.forEach((read, slot) => {
        entries.push({ ...read.style, at: [chunk, slot] });
      });
    }
    const parents = new Map(independent.map((read) => [read.style.id, read.style]));
    for (const read of dependent) {
      const parent = parents.get(read.style.parent as string) as PackStyle;
      const own = read.text.includes("citation-format=") ? read.style.format : parent.format;
      const entry: PackStyle = { ...read.style, format: own };
      if (parent.note) entry.note = true;
      entries.push(entry);
    }

    // The locales the common styles use, together, then the rest.
    const firstLocales = ["en-US", "en-GB", "zh-CN"];
    const otherLocales = [...localeText.keys()].filter((code) => !firstLocales.includes(code));
    const locales: Record<string, PackSlot> = {};
    for (const group of [firstLocales, otherLocales]) {
      const chunk = add(group.map((code) => localeText.get(code) as string));
      group.forEach((code, slot) => {
        locales[code] = [chunk, slot];
      });
    }
    const translators = [...localeText].map(([code, text]) => ({
      code,
      translators: people(/<info>([\s\S]*?)<\/info>/.exec(text)?.[1] ?? "", "translator"),
    }));
    const noticeChunk = add([notice([...independent, ...dependent].sort((a, b) => a.style.id.localeCompare(b.style.id)), translators)]);

    let offset = 0;
    const index: PackIndex = {
      version: 1,
      source: { styles: STYLES_COMMIT, locales: LOCALES_COMMIT },
      chunks: chunks.map((bytes) => {
        const where = { offset, length: bytes.length };
        offset += bytes.length;
        return where;
      }),
      styles: entries,
      locales,
      notice: [noticeChunk, 0],
    };
    const indexBytes = compress(JSON.stringify(index));
    const header = Buffer.alloc(PACK_MAGIC.length + 4);
    header.write(PACK_MAGIC, 0, "latin1");
    header.writeUInt32LE(indexBytes.length, PACK_MAGIC.length);
    const pack = Buffer.concat([header, indexBytes, ...chunks]);
    writeFileSync(out, pack);
    return { pack, independent: independent.length, dependent: dependent.length, left };
  } finally {
    rmSync(work, { recursive: true, force: true });
  }
}

const check = process.argv.includes("--check");
const target = check ? join(mkdtempSync(join(tmpdir(), "incarnamind-pack-")), "styles.pack") : OUT;
const { pack, independent, dependent, left } = build(target);
for (const line of left) console.log(`left out: ${line}`);
console.log(
  `${independent} styles and ${dependent} journal styles, ${left.length} left out; ${pack.length} bytes`,
);
const sha = createHash("sha256").update(pack).digest("hex");
if (check) {
  const committed = readFileSync(OUT);
  if (!committed.equals(pack)) {
    console.error("The committed styles.pack isn't what its sources build to.");
    process.exit(1);
  }
} else {
  writeFileSync(`${OUT}.sha256`, `${sha}  styles.pack\n`);
}
