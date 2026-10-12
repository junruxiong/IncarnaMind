/**
 * Writes manifest.json from the files under eval/word/ and sources.json (where each came from),
 * and the table in README.md. Run it after adding or changing a file:
 *
 *   INCARNAMIND_WORD_UPDATE=1 npm run eval:word
 */
import { createHash } from "node:crypto";
import { existsSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { type CorpusEntry, inspectPackage, WORD_DIR } from "./corpus";
import { FEATURE_COLUMNS, featureKinds } from "./features";

type Source = Pick<CorpusEntry, "file" | "origin" | "language" | "licence" | "source" | "known">;

const START = "<!-- files:start -->";
const END = "<!-- files:end -->";

const SECTIONS: { set: CorpusEntry["set"]; title: string }[] = [
  { set: "documents", title: "Documents" },
  { set: "tracked-changes", title: "Files with someone else's tracked changes" },
  { set: "saved", title: "Files saved in Word for Mac, WPS and Google Docs" },
];

const HEADINGS = [
  "Tracked",
  "Fn",
  "En",
  "Fields",
  "H/F",
  "Comments",
  "Tables",
  "Lists",
  "Controls",
  "Images",
];

function table(entries: CorpusEntry[]): string {
  const rows = entries.map((entry) => {
    const counts = FEATURE_COLUMNS.map((column) => String(entry.features[column] || "·"));
    const origin = entry.source.startsWith("https://")
      ? `[${entry.origin}](${entry.source})`
      : entry.source || entry.origin;
    const authors = entry.features.revisionAuthors.join(", ");
    return `| \`${entry.file.split("/").slice(1).join("/")}\` | ${entry.language} | ${entry.licence} | ${origin} | ${featureKinds(entry.features)} | ${counts.join(" | ")} | ${authors} |`;
  });
  return [
    `| File | Lang | Licence | Source | Kinds | ${HEADINGS.join(" | ")} | Change authors |`,
    `|---|---|---|---|---|${HEADINGS.map(() => "---").join("|")}|---|`,
    ...rows,
  ].join("\n");
}

const SAVED_BY = { "word-mac": "Word for Mac", wps: "WPS", "google-docs": "Google Docs" } as const;

/** The files saved by hand (TO-SAVE.md): saved/<app>/<name>.docx, each a re-save of a synthetic file. */
function savedFiles(): Source[] {
  const root = join(WORD_DIR, "saved");
  if (!existsSync(root)) return [];
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
        licence: "Apache-2.0",
        source: `documents/synthetic/${name}, saved in ${SAVED_BY[app]}`,
      }));
  });
}

describe.skipIf(process.env.INCARNAMIND_WORD_UPDATE !== "1")("update the manifest", () => {
  it("writes manifest.json and the README table", async () => {
    const sources = [
      ...(JSON.parse(readFileSync(join(WORD_DIR, "sources.json"), "utf8")) as Source[]),
      ...savedFiles(),
    ];
    const entries: CorpusEntry[] = [];
    for (const source of sources) {
      const bytes = readFileSync(join(WORD_DIR, source.file));
      const { features } = await inspectPackage(bytes);
      entries.push({
        ...source,
        set: source.file.split("/")[0] as CorpusEntry["set"],
        sha256: createHash("sha256").update(bytes).digest("hex"),
        features,
      });
    }
    writeFileSync(join(WORD_DIR, "manifest.json"), `${JSON.stringify(entries, null, 2)}\n`);

    const readme = readFileSync(join(WORD_DIR, "README.md"), "utf8");
    const block = SECTIONS.filter(({ set }) => entries.some((e) => e.set === set))
      .map(
        ({ set, title }) =>
          `### ${title} (${entries.filter((e) => e.set === set).length})\n\n${table(entries.filter((e) => e.set === set))}`,
      )
      .join("\n\n");
    const from = readme.indexOf(START);
    const to = readme.indexOf(END);
    expect(from).toBeGreaterThan(-1);
    writeFileSync(
      join(WORD_DIR, "README.md"),
      `${readme.slice(0, from + START.length)}\n\n${block}\n\n${readme.slice(to)}`,
    );
  });
});
