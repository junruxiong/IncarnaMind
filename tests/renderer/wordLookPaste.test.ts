import { Slice } from "@tiptap/pm/model";
import { describe, expect, onTestFinished, test } from "vitest";
import { WORD_LOOK_ATTRIBUTE, WORD_RUN_MARK } from "../../src/core";
import { wordLookOf } from "../../src/shared/wordLook";
import { createTempDataFolder, startCore } from "../helpers/core";
import { boundEditor } from "../helpers/editor";
import { contractFile } from "../helpers/wordLookFiles";
import { importWordBytes } from "../helpers/wordMinds";

describe("pasting into a Mind imported from Word", () => {
  test("keeps the looks the Mind has and drops the ones it doesn't, on paragraphs, cells and runs", async () => {
    const dataDir = await createTempDataFolder();
    const core = startCore(dataDir);
    onTestFinished(() => core.close());
    const { client } = await importWordBytes(core, dataDir, contractFile(), { origin: "own" });
    const look = wordLookOf(client.doc);
    if (!look) throw new Error("No look.");
    const [paragraphKey] = Object.keys(look.paragraphs);
    const [cellKey] = Object.keys(look.cells);
    const [runKey] = Object.keys(look.runs);
    const editor = boundEditor(client);

    const run = (text: string, k: string) => ({
      type: "text",
      text,
      marks: [{ type: WORD_RUN_MARK, attrs: { k } }],
    });
    const pasted = editor.schema.nodeFromJSON({
      type: "doc",
      content: [
        {
          type: "paragraph",
          attrs: { [WORD_LOOK_ATTRIBUTE]: paragraphKey },
          content: [run("Known", String(runKey))],
        },
        {
          type: "paragraph",
          attrs: { [WORD_LOOK_ATTRIBUTE]: "pfromanotherfile" },
          content: [run("Unknown", "rfromanotherfile")],
        },
        {
          type: "table",
          content: [
            {
              type: "tableRow",
              content: [
                {
                  type: "tableCell",
                  attrs: { [WORD_LOOK_ATTRIBUTE]: cellKey },
                  content: [
                    {
                      type: "paragraph",
                      content: [run("In", String(runKey)), run(" a cell", "rfromanotherfile")],
                    },
                  ],
                },
                {
                  type: "tableCell",
                  attrs: { [WORD_LOOK_ATTRIBUTE]: "cfromanotherfile" },
                  content: [{ type: "paragraph", content: [{ type: "text", text: "Plain" }] }],
                },
              ],
            },
          ],
        },
      ],
    });
    let slice = new Slice(pasted.content, 0, 0);
    for (const plugin of editor.state.plugins) {
      const transform = plugin.props.transformPasted;
      if (transform) slice = transform.call(plugin, slice, editor.view, false);
    }

    const looks: (string | null)[] = [];
    const runs: (string | null)[] = [];
    slice.content.descendants((node) => {
      if (node.type.name !== "text" && node.attrs[WORD_LOOK_ATTRIBUTE] !== undefined) {
        looks.push(node.attrs[WORD_LOOK_ATTRIBUTE]);
      }
      if (node.isText) {
        const mark = node.marks.find((each) => each.type.name === WORD_RUN_MARK);
        runs.push(mark ? String(mark.attrs.k) : null);
      }
      return true;
    });
    expect(looks).toEqual([paragraphKey, null, null, cellKey, null, null, null]);
    expect(runs).toEqual([runKey, null, runKey, null, null]);
  });
});
