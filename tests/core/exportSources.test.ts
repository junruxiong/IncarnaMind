/**
 * A Numbered Mind's Bibliography in an export (#102): each cited Document once with the
 * Locations cited, a web page with its site, address and accessed day, and nothing from a
 * pending change.
 */
import { readFileSync } from "node:fs";
import type { JSONContent } from "@tiptap/core";
import { describe, expect, test } from "vitest";
import { type Document, MIND_SETTINGS_FIELD } from "../../src/core";
import { createTempDataFolder, manualClock, startCore } from "../helpers/core";
import { addAndProcess, waitForProcessing, writeSourceFile } from "../helpers/documents";
import { connectToMind } from "../helpers/mindClient";
import { writeMind } from "../helpers/minds";

const ARTICLE = readFileSync(new URL("../fixtures/web/article.html", import.meta.url), "utf8");

async function setUp() {
  const dataDir = await createTempDataFolder();
  const sources = await createTempDataFolder();
  const core = startCore(dataDir, { now: manualClock().now });
  const [tides, lease] = await addAndProcess(core, [
    await writeSourceFile(sources, "Tides.txt", "Spring tides happen at new moon."),
    await writeSourceFile(sources, "Lease.txt", "The tenant pays rent monthly."),
  ]);
  const saved = await core.saveWebPage({
    address: "https://encyclopedia.example/tea",
    html: ARTICLE,
  });
  if (saved.outcome !== "saved") throw new Error(saved.failure.message);
  const [tea] = await waitForProcessing(core, [saved.document.id]);
  const mind = await core.createMind({ title: "Tides" });
  const client = await connectToMind(core, mind.id);
  return {
    core,
    mind,
    client,
    tides: tides as Document,
    lease: lease as Document,
    tea: tea as Document,
  };
}

const text = (value: string): JSONContent => ({ type: "text", text: value });

const cite = (
  document: Document,
  location: object,
  extra: Record<string, unknown> = {},
): JSONContent => ({
  type: "citation",
  attrs: {
    passageId: "passage-1",
    documentId: document.id,
    documentName: document.name,
    contentHash: document.contentHash,
    location,
    quote: "q",
    check: "found",
    ...extra,
  },
});

const page = (from: number, to = from) => ({ kind: "page", from, to });

async function exported(
  setup: Awaited<ReturnType<typeof setUp>>,
  shown: boolean,
  language: "en" | "zh-CN" = "en",
) {
  const { core, mind, client, tides, lease, tea } = setup;
  await core.updateSettings({ user: { language } });
  writeMind(client, [
    {
      type: "paragraph",
      content: [
        text("Tides "),
        cite(tides, page(2)),
        text(" again "),
        cite(tides, page(3, 4)),
        text(" and "),
        cite(tides, page(2)),
        text(". Tea "),
        cite(tea, { kind: "section", heading: "History", paragraphs: { from: 2, to: 2 } }),
        text(". A pending change "),
        cite(lease, page(1), { proposal: { id: "p1", request: "r1", kind: "insert" } }),
        text("."),
      ],
    },
  ]);
  if (shown) client.doc.getMap(MIND_SETTINGS_FIELD).set("bibliography", "shown");
  await client.settled();
  return new TextDecoder().decode((await core.exportMind(mind.id, { format: "markdown" })).data);
}

describe("a Numbered Mind's Bibliography in an export", () => {
  test("lists each cited Document once, with the Locations cited", async () => {
    const setup = await setUp();
    const markdown = await exported(setup, true);
    const bibliography = markdown.slice(markdown.indexOf("## Bibliography"));
    expect(bibliography).toContain("Tides. Cited: p. 2; p. 3–4");
    expect(bibliography.match(/Tides\./g)).toHaveLength(1);
  });

  test("a web page has its title, site, address and the day the quoted version was fetched", async () => {
    const setup = await setUp();
    const markdown = await exported(setup, true);
    const fetched = setup.tea.web?.versions.find(
      (version) => version.contentHash === setup.tea.contentHash,
    )?.fetchedAt;
    expect(fetched).toBeDefined();
    const day = new Intl.DateTimeFormat("en-GB", {
      day: "numeric",
      month: "long",
      year: "numeric",
    }).format(new Date(fetched ?? ""));
    expect(markdown.replaceAll("\u00a0", " ")).toContain(
      `Tea, The Open Encyclopedia, https://encyclopedia.example/tea, accessed ${day}. Cited: § History, ¶ 2`,
    );
  });

  test("a pending change's Citation isn't counted", async () => {
    const setup = await setUp();
    const markdown = await exported(setup, true);
    expect(markdown.slice(markdown.indexOf("## Bibliography"))).not.toContain("Lease");
  });

  test("is in the interface language", async () => {
    const setup = await setUp();
    const markdown = await exported(setup, true, "zh-CN");
    expect(markdown).toContain("## 参考文献");
    expect(markdown).toContain("Tides。引用位置：第 2 页；第 3–4 页");
  });

  test("a Numbered Mind that doesn't show one exports none", async () => {
    const setup = await setUp();
    expect(await exported(setup, false)).not.toContain("Bibliography");
  });
});
