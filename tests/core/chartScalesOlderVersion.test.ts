import { describe, expect, test, vi } from "vitest";
import * as Y from "yjs";
import {
  CHART_NODE,
  CONTENT_SCHEMA_VERSION,
  CONTENT_SCHEMA_VERSION_KEY,
  MIND_CONTENT_FIELD,
  MIND_SETTINGS_FIELD,
  MindReadOnlyError,
} from "../../src/core";
import { createTempDataFolder, startCore } from "../helpers/core";

/*
 * The version before a chart's values kept their scale (#579): its core is
 * this one with the newest content schema it understands set back to 13.
 */
vi.mock("../../src/core/api", async (importOriginal) => ({
  ...(await importOriginal<object>()),
  CONTENT_SCHEMA_VERSION: 13,
}));

/** The content schema version chart scales came with: what a Mind written since records. */
const SCALES = 14;

const value = (shown: string, number: number, row: number) => ({
  shown,
  value: number,
  unit: { kind: "currency", code: "USD" },
  scale: 1e6,
  citation: null,
  from: { kind: "cell", row, column: 1 },
});

const data = (withScale: boolean) => ({
  labelHeading: "Quarter",
  series: ["2025"],
  rows: [
    { label: "Q1", values: [value("$341.5 million", 341_500_000, 1)] },
    { label: "Q2", values: [value("$362.0 million", 362_000_000, 2)] },
  ].map((row) => ({
    ...row,
    values: row.values.map(({ scale, ...rest }) => (withScale ? { ...rest, scale } : rest)),
  })),
  cutFrom: null,
});

/** A Mind as this version writes one with a chart: values with their scale, and the version it records. */
function mindWithChart(): Y.Doc {
  const doc = new Y.Doc();
  const chart = new Y.XmlElement(CHART_NODE);
  chart.setAttribute("id", "chart-1");
  chart.setAttribute("kind", "bar");
  chart.setAttribute("title", "Revenue by quarter");
  chart.setAttribute("data", data(true) as unknown as string);
  doc.getXmlFragment(MIND_CONTENT_FIELD).push([chart]);
  doc.getMap(MIND_SETTINGS_FIELD).set(CONTENT_SCHEMA_VERSION_KEY, SCALES);
  return doc;
}

const chartData = (doc: Y.Doc) => {
  const chart = doc.getXmlFragment(MIND_CONTENT_FIELD).get(0);
  return chart instanceof Y.XmlElement ? (chart.getAttribute("data") as unknown) : null;
};

describe("a version from before chart scales", () => {
  test("opens a Mind whose charts keep their scales read-only, so its Update can't drop them", async () => {
    expect(CONTENT_SCHEMA_VERSION).toBe(13);
    const core = startCore(await createTempDataFolder());
    const mind = await core.createMind({ title: "Acme" });
    await core.applyMindUpdate(mind.id, Y.encodeStateAsUpdate(mindWithChart()));

    const opened = await core.openMind(mind.id);
    expect(opened.access).toBe("read-only");
    const doc = new Y.Doc();
    Y.applyUpdate(doc, opened.state);
    expect(chartData(doc)).toEqual(data(true));

    // Its Update writes the chart's values again as it reads them, without their scale: refused.
    const sent: Uint8Array[] = [];
    doc.on("update", (update: Uint8Array) => sent.push(update));
    const chart = doc.getXmlFragment(MIND_CONTENT_FIELD).get(0);
    if (!(chart instanceof Y.XmlElement)) throw new Error("No chart.");
    chart.setAttribute("data", data(false) as unknown as string);
    await expect(core.applyMindUpdate(mind.id, sent[0] as Uint8Array)).rejects.toThrow(
      MindReadOnlyError,
    );
    const again = new Y.Doc();
    Y.applyUpdate(again, (await core.openMind(mind.id)).state);
    expect(chartData(again)).toEqual(data(true));
  });
});
