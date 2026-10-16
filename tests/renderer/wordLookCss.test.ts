import type { EditorState } from "@tiptap/pm/state";
import { DecorationSet } from "@tiptap/pm/view";
import { describe, expect, onTestFinished, test } from "vitest";
import * as Y from "yjs";
import { MIND_WORD_LOOK_FIELD } from "../../src/core";
import { wordLookCss, wordLookScope } from "../../src/renderer/src/editor/wordLookCss";
import { wordLookOf } from "../../src/shared/wordLook";
import { createTempDataFolder, startCore } from "../helpers/core";
import { boundEditor } from "../helpers/editor";
import { longContract } from "../helpers/longContract";
import { importWordBytes } from "../helpers/wordMinds";

async function importedLook(bytes: Buffer) {
  const dataDir = await createTempDataFolder();
  const core = startCore(dataDir);
  onTestFinished(() => core.close());
  const { mindId, client } = await importWordBytes(core, dataDir, bytes, { origin: "own" });
  const look = wordLookOf(client.doc);
  if (!look) throw new Error("No look.");
  return { mindId, client, look };
}

const rulesOf = (css: string) => css.split("\n").filter(Boolean);

function decorationsIn(state: EditorState): number {
  let count = 0;
  for (const plugin of state.plugins) {
    const found = plugin.props.decorations?.call(plugin, state);
    if (found instanceof DecorationSet) count += found.find().length;
  }
  return count;
}

describe("the Word look's stylesheet", () => {
  test("has one rule a style, whatever the number of Blocks, each under the Mind's own scope", async () => {
    const short = await importedLook(longContract(8));
    const long = await importedLook(longContract());
    const shortRules = rulesOf(wordLookCss(short.mindId, short.look));
    const longRules = rulesOf(wordLookCss(long.mindId, long.look));
    expect(longRules).toHaveLength(shortRules.length);
    expect(longRules.length).toBeLessThan(20);
    for (const rule of longRules)
      expect(rule.startsWith(`.${wordLookScope(long.mindId)} `)).toBe(true);
  });

  test("the 304-page contract in its look has no decorations", async () => {
    const { client } = await importedLook(longContract());
    const editor = boundEditor(client);
    expect(editor.state.doc.childCount).toBeGreaterThan(2800);
    expect(decorationsIn(editor.state)).toBe(0);
  });

  test("a look another device wrote can't reach outside its rule", () => {
    const doc = new Y.Doc();
    const map = doc.getMap(MIND_WORD_LOOK_FIELD);
    map.set("runs", {
      r1: { font: 'Evil"; } body { display: none; } x { font-family: "', color: "red; }" },
      "r2] , body [x": { bold: true },
    });
    map.set("paragraphs", { p1: { align: "center; } body { display: none", left: "1px" } });
    map.set("body", 12);
    map.set("width", null);
    map.set("version", 1);
    const look = wordLookOf(doc);
    if (!look) throw new Error("No look.");
    expect(look.runs).toEqual({ r1: {} });
    expect(look.paragraphs).toEqual({ p1: {} });
    const css = wordLookCss("mind", look);
    expect(css).not.toContain("display");
    expect(css.split("{").length).toBe(css.split("}").length);
    expect(css.split("{").length - 1).toBe(rulesOf(css).length);
  });
});
