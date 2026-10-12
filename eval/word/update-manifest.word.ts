/**
 * Writes manifest.json (the committed files) and extended.json's feature counts (the extended
 * set, once fetched), and the tables in README.md and EXTENDED.md. Run it after adding or
 * changing a file:
 *
 *   INCARNAMIND_WORD_UPDATE=1 npm run eval:word
 *
 * The committed files come from sources.json (where each came from) and the files under
 * saved/<app>/ (TO-SAVE.md). For the extended set, run `npm run eval:word:fetch` first.
 */
import { createHash } from "node:crypto";
import { existsSync, readdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import {
  type CorpusEntry,
  DOMAINS,
  type ExtendedEntry,
  extendedCache,
  inspectPackage,
  LANGUAGES,
  readExtended,
  WORD_DIR,
} from "./corpus";
import { FEATURES, type Features, featureKinds } from "./features";

type Source = Omit<CorpusEntry, "set" | "sha256" | "features">;

const SAVED_BY = { "word-mac": "Word for Mac", wps: "WPS", "google-docs": "Google Docs" } as const;
const MARKERS = (name: string) => [`<!-- ${name}:start -->`, `<!-- ${name}:end -->`] as const;

/** The files saved by hand (TO-SAVE.md): saved/<app>/<name>.docx, each a re-save of a synthetic file. */
function savedFiles(sources: Source[]): Source[] {
  const root = join(WORD_DIR, "saved");
  if (!existsSync(root)) return [];
  const original = (name: string) => sources.find((each) => each.file.endsWith(`/${name}`));
  return (Object.keys(SAVED_BY) as (keyof typeof SAVED_BY)[]).flatMap((app) => {
    const folder = join(root, app);
    if (!existsSync(folder)) return [];
    return readdirSync(folder)
      .filter((name) => name.endsWith(".docx"))
      .sort()
      .map((name) => ({
        file: `saved/${app}/${name}`,
        origin: app,
        language: name.startsWith("zh-") ? ("zh" as const) : ("en" as const),
        domain: original(name)?.domain ?? "legal",
        licence: "Apache-2.0",
        source: `documents/synthetic/${name}, saved in ${SAVED_BY[app]}`,
      }));
  });
}

const sha256 = (bytes: Uint8Array) => createHash("sha256").update(bytes).digest("hex");
const link = (label: string, url: string) => (url.startsWith("http") ? `[${label}](${url})` : url);

function featureList(features: Features): string {
  return FEATURES.filter(([key]) => features[key] > 0)
    .map(([key, label]) => `${label} ${features[key]}`)
    .join(", ");
}

const SET_TITLES: Record<CorpusEntry["set"], string> = {
  documents: "Documents (30)",
  "tracked-changes": "Files with someone else's tracked changes (20)",
  breadth: "Breadth: more domains, languages and complexity",
  saved: "Files saved in Word for Mac, WPS and Google Docs",
};

function coreTable(entries: CorpusEntry[]): string {
  const rows = entries.map((entry) => {
    const authors = entry.features.revisionAuthors.join(", ");
    const source = entry.source.startsWith("http")
      ? link(entry.origin, entry.source)
      : entry.source || "synthetic";
    return `| \`${entry.file.split("/").slice(1).join("/")}\` | ${entry.language} | ${entry.domain} | ${entry.licence} | ${source} | ${featureKinds(entry.features)} | ${featureList(entry.features)}${authors ? ` (change authors: ${authors})` : ""} |`;
  });
  return [
    "| File | Lang | Domain | Licence | Source | Kinds | Features (count) |",
    "|---|---|---|---|---|---|---|",
    ...rows,
  ].join("\n");
}

type Row = { language: string; domain: string; features: Features; core: boolean };

function coverage(rows: Row[]): string {
  const languages = LANGUAGES.filter((language) => rows.some((row) => row.language === language));
  const domains = DOMAINS.filter((domain) => rows.some((row) => row.domain === domain));
  const header = (first: string, columns: readonly string[]) =>
    `| ${first} | ${columns.join(" | ")} | All |\n|---${"|---".repeat(columns.length + 1)}|`;
  const cell = (n: number, c: number) => (n === 0 ? "·" : c === n ? `${n}` : `${n} (${c})`);

  const byDomain = domains.map((domain) => {
    const mine = rows.filter((row) => row.domain === domain);
    const cells = languages.map((language) => {
      const found = mine.filter((row) => row.language === language);
      return cell(found.length, found.filter((row) => row.core).length);
    });
    return `| ${domain} | ${cells.join(" | ")} | ${cell(mine.length, mine.filter((row) => row.core).length)} |`;
  });
  const totals = languages.map((language) => {
    const found = rows.filter((row) => row.language === language);
    return cell(found.length, found.filter((row) => row.core).length);
  });

  const featureRows = (by: "language" | "domain", columns: readonly string[]) =>
    FEATURES.map(([key, label]) => {
      const withFeature = rows.filter((row) => row.features[key] > 0);
      const cells = columns.map((column) => {
        const found = withFeature.filter((row) => row[by] === column);
        return cell(found.length, found.filter((row) => row.core).length);
      });
      return `| ${label} | ${cells.join(" | ")} | ${cell(withFeature.length, withFeature.filter((row) => row.core).length)} |`;
    });

  return [
    "Each cell is the number of files, and in brackets how many of them are committed (the core set), when not all.",
    "",
    "**Domain by language**",
    "",
    header("Domain", languages),
    ...byDomain,
    `| All | ${totals.join(" | ")} | ${cell(rows.length, rows.filter((row) => row.core).length)} |`,
    "",
    "**Feature by language** (files with at least one)",
    "",
    header("Feature", languages),
    ...featureRows("language", languages),
    "",
    "**Feature by domain**",
    "",
    header("Feature", domains),
    ...featureRows("domain", domains),
  ].join("\n");
}

function extendedTable(entries: ExtendedEntry[]): string {
  const rows = entries.map((entry) => {
    const source = entry.url ? link(entry.source, entry.url) : entry.source;
    return `| \`${entry.name}\` | ${entry.language} | ${entry.domain} | ${entry.licence} | ${source} | ${featureKinds(entry.features)} | ${featureList(entry.features)} |`;
  });
  return [
    "| File | Lang | Domain | Licence | Source | Kinds | Features (count) |",
    "|---|---|---|---|---|---|---|",
    ...rows,
  ].join("\n");
}

function replaceBlock(path: string, name: string, block: string): void {
  const [start, end] = MARKERS(name);
  const text = readFileSync(path, "utf8");
  const from = text.indexOf(start);
  const to = text.indexOf(end);
  expect(from, `${path} needs ${start}`).toBeGreaterThan(-1);
  writeFileSync(path, `${text.slice(0, from + start.length)}\n\n${block}\n\n${text.slice(to)}`);
}

describe.skipIf(process.env.INCARNAMIND_WORD_UPDATE !== "1")("update the manifests", () => {
  it("writes manifest.json, the extended set's features and the tables", async () => {
    const listed = JSON.parse(readFileSync(join(WORD_DIR, "sources.json"), "utf8")) as Source[];
    const sources = [...listed, ...savedFiles(listed)];
    const entries: CorpusEntry[] = [];
    let coreBytes = 0;
    for (const source of sources) {
      const bytes = readFileSync(join(WORD_DIR, source.file));
      coreBytes += bytes.byteLength;
      entries.push({
        ...source,
        set: source.file.split("/")[0] as CorpusEntry["set"],
        sha256: sha256(bytes),
        features: (await inspectPackage(bytes)).features,
      });
    }
    writeFileSync(join(WORD_DIR, "manifest.json"), `${JSON.stringify(entries, null, 2)}\n`);

    const extended = readExtended();
    const cache = extendedCache();
    let extendedBytes = 0;
    let fetched = 0;
    for (const entry of extended) {
      const path = join(cache, entry.name);
      if (!existsSync(path)) continue;
      fetched++;
      extendedBytes += statSync(path).size;
      entry.features = (await inspectPackage(readFileSync(path))).features;
    }
    const complete = fetched === extended.length;
    if (complete) {
      writeFileSync(join(WORD_DIR, "extended.json"), `${JSON.stringify(extended, null, 2)}\n`);
    }

    const rows: Row[] = [
      ...entries.map((entry) => ({ ...entry, core: true })),
      ...extended.map((entry) => ({ ...entry, core: false })),
    ];
    const readme = join(WORD_DIR, "README.md");
    const sets = (Object.keys(SET_TITLES) as CorpusEntry["set"][]).filter((set) =>
      entries.some((entry) => entry.set === set),
    );
    replaceBlock(
      readme,
      "files",
      sets
        .map(
          (set) =>
            `### ${SET_TITLES[set]}\n\n${coreTable(entries.filter((entry) => entry.set === set))}`,
        )
        .join("\n\n"),
    );
    const mb = (bytes: number) => (bytes / 1_000_000).toFixed(1);
    replaceBlock(
      readme,
      "coverage",
      `${coverage(rows)}\n\n${entries.length} committed files (${mb(coreBytes)} MB) and ${extended.length} in the extended set (${mb(extendedBytes)} MB when fetched): ${rows.length} in all.`,
    );
    replaceBlock(join(WORD_DIR, "EXTENDED.md"), "files", extendedTable(extended));
    expect(complete, "fetch the extended set first: npm run eval:word:fetch").toBe(true);
  });
});
