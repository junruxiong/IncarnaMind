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
 * The version before charts (#436): its core is this one with the newest
 * content schema it understands set back to 6, which knows no chart node.
 */
vi.mock("../../src/core/api", async (importOriginal) => ({
  ...(await importOriginal<object>()),
  CONTENT_SCHEMA_VERSION: 6,
}));

/** The content schema version charts came with: what a Mind with a chart records. */
const CHARTS = 7;

/** A Mind as this version writes one with a chart: the chart, and the version it records. */
function mindWithChart(): Y.Doc {
  const doc = new Y.Doc();
  const chart = new Y.XmlElement(CHART_NODE);
  chart.setAttribute("id", "chart-1");
  chart.setAttribute("kind", "bar");
  chart.setAttribute("title", "Revenue by quarter");
  chart.setAttribute("data", {
    labelHeading: "Quarter",
    series: ["2025"],
    rows: [
      {
        label: "Q1",
        values: [
          {
            shown: "341.5",
            value: 341.5,
            unit: { kind: "plain" },
            citation: { documentId: "acme", documentName: "Acme", pageFrom: 2, pageTo: 2 },
            from: { kind: "cell", row: 1, column: 1 },
          },
        ],
      },
    ],
    cutFrom: null,
  } as unknown as string);
  doc.getXmlFragment(MIND_CONTENT_FIELD).push([chart]);
  doc.getMap(MIND_SETTINGS_FIELD).set(CONTENT_SCHEMA_VERSION_KEY, CHARTS);
  return doc;
}

describe("a version from before charts", () => {
  test("opens a Mind with a chart read-only, and keeps the chart as it is", async () => {
    expect(CONTENT_SCHEMA_VERSION).toBe(6);
    const core = startCore(await createTempDataFolder());
    const mind = await core.createMind({ title: "Acme" });
    const written = Y.encodeStateAsUpdate(mindWithChart());
    await core.applyMindUpdate(mind.id, written);

    const opened = await core.openMind(mind.id);
    expect(opened.access).toBe("read-only");
    const doc = new Y.Doc();
    Y.applyUpdate(doc, opened.state);
    const chart = doc.getXmlFragment(MIND_CONTENT_FIELD).get(0);
    expect(chart instanceof Y.XmlElement && chart.nodeName).toBe(CHART_NODE);

    // Its editor would drop the node it doesn't know: that change is refused, so nothing is lost.
    const sent: Uint8Array[] = [];
    doc.on("update", (update: Uint8Array) => sent.push(update));
    doc.getXmlFragment(MIND_CONTENT_FIELD).delete(0, 1);
    await expect(core.applyMindUpdate(mind.id, sent[0] as Uint8Array)).rejects.toThrow(
      MindReadOnlyError,
    );
    const again = new Y.Doc();
    Y.applyUpdate(again, (await core.openMind(mind.id)).state);
    expect(again.getXmlFragment(MIND_CONTENT_FIELD).length).toBe(1);
  });
});
