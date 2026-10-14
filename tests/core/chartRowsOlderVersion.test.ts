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
 * The version before a cut chart kept the rows it doesn't draw (#503): its
 * core is this one with the newest content schema it understands set back to 15.
 */
vi.mock("../../src/core/api", async (importOriginal) => ({
  ...(await importOriginal<object>()),
  CONTENT_SCHEMA_VERSION: 15,
}));

/** The content schema version these rows came with: what a Mind written since records. */
const CUT_ROWS = 16;

const row = (day: number) => ({
  label: `Day ${day}`,
  values: [
    {
      shown: String(40 + day),
      value: 40 + day,
      unit: { kind: "plain" },
      citation: null,
      from: { kind: "cell", row: day, column: 1 },
    },
  ],
});

/** A chart cut after its first two rows, keeping its third to list, or as an older Update writes it. */
const data = (withCutRows: boolean) => ({
  labelHeading: "Day",
  series: ["Orders"],
  rows: [row(1), row(2)],
  cutFrom: 3,
  ...(withCutRows ? { cutRows: [row(3)] } : {}),
});

function mindWithChart(): Y.Doc {
  const doc = new Y.Doc();
  const chart = new Y.XmlElement(CHART_NODE);
  chart.setAttribute("id", "chart-1");
  chart.setAttribute("kind", "line");
  chart.setAttribute("title", "Orders by day");
  chart.setAttribute("data", data(true) as unknown as string);
  doc.getXmlFragment(MIND_CONTENT_FIELD).push([chart]);
  doc.getMap(MIND_SETTINGS_FIELD).set(CONTENT_SCHEMA_VERSION_KEY, CUT_ROWS);
  return doc;
}

const chartData = (doc: Y.Doc) => {
  const chart = doc.getXmlFragment(MIND_CONTENT_FIELD).get(0);
  return chart instanceof Y.XmlElement ? (chart.getAttribute("data") as unknown) : null;
};

describe("a version from before a cut chart kept its other rows", () => {
  test("opens a Mind whose charts keep them read-only, so its Update can't drop them", async () => {
    expect(CONTENT_SCHEMA_VERSION).toBe(15);
    const core = startCore(await createTempDataFolder());
    const mind = await core.createMind({ title: "Orders" });
    await core.applyMindUpdate(mind.id, Y.encodeStateAsUpdate(mindWithChart()));

    const opened = await core.openMind(mind.id);
    expect(opened.access).toBe("read-only");
    const doc = new Y.Doc();
    Y.applyUpdate(doc, opened.state);
    expect(chartData(doc)).toEqual(data(true));

    // Its Update writes the chart's data again as it makes it, without the rows it doesn't draw: refused.
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
