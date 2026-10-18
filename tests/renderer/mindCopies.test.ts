import { join } from "node:path";
import { describe, expect, test, vi } from "vitest";
import * as Y from "yjs";
import { type Core, DATABASE_FILE, MIND_CONTENT_FIELD } from "../../src/core";
import { openDatabase } from "../../src/core/storage";
import { createTempDataFolder, startCore } from "../helpers/core";

/*
 * A window's copy of a Mind after a change the core couldn't take (#1119): the
 * real core behind the bridge, the window's own copy, and a restart to see what
 * was stored.
 */

const bridge = vi.hoisted(() => ({
  core: null as Core | null,
  /** Calls of applyMindUpdate to drop before they reach the core, as a lost message would be. */
  dropping: 0,
}));

vi.mock("../../src/renderer/src/core", () => ({
  core: {
    on: (...args: Parameters<Core["on"]>) => bridge.core?.on(...args) ?? (() => undefined),
    openMind: (mindId: string) => {
      if (!bridge.core) throw new Error("No core.");
      return bridge.core.openMind(mindId);
    },
    closeMind: async (mindId: string) => bridge.core?.closeMind(mindId),
    applyMindUpdate: async (mindId: string, update: Uint8Array) => {
      if (!bridge.core) throw new Error("No core.");
      if (bridge.dropping > 0) {
        bridge.dropping -= 1;
        throw new Error("The message was lost.");
      }
      return bridge.core.applyMindUpdate(mindId, update);
    },
  },
}));

const { openMindCopy } = await import("../../src/renderer/src/mindCopies");

/** A Mind open in a window's copy, holding "BASE" once stored, and a way to type at its end. */
async function openCopy(dataDir: string) {
  const core = startCore(dataDir);
  bridge.core = core;
  const mind = await core.createMind({ title: "Notes" });
  const failures: unknown[] = [];
  const doc = await new Promise<Y.Doc>((resolve, reject) => {
    openMindCopy(
      mind.id,
      (copy) => (copy.doc ? resolve(copy.doc) : reject(new Error("No copy."))),
      (error) => failures.push(error),
    );
  });
  const stored = () => core.openMind(mind.id).then(({ state }) => textOf(state));
  const paragraph = new Y.XmlElement("paragraph");
  const text = new Y.XmlText("BASE");
  paragraph.insert(0, [text]);
  doc.getXmlFragment(MIND_CONTENT_FIELD).push([paragraph]);
  await vi.waitFor(async () => expect(await stored()).toBe("<paragraph>BASE</paragraph>"));
  return {
    core,
    mindId: mind.id,
    failures,
    stored,
    type: (words: string) => text.insert(text.length, words),
  };
}

const textOf = (state: Uint8Array) => {
  const doc = new Y.Doc();
  Y.applyUpdate(doc, state);
  return doc.getXmlFragment(MIND_CONTENT_FIELD).toString();
};

async function afterRestart(dataDir: string, mindId: string): Promise<string> {
  const core = startCore(dataDir);
  return textOf((await core.openMind(mindId)).state);
}

describe("A window's copy after a change the core couldn't take (#1119)", () => {
  // protects=the issue's case: one write fails and is reported once, storage comes back, and the typing after it is stored with the failed change, there after a restart; fails_when=the later typing shows in the window but is gone after a restart; no_claim=a real full disk.
  test("typing after a failed write is stored, with the failed change, and the failure is said once", async () => {
    const dataDir = await createTempDataFolder();
    const copy = await openCopy(dataDir);
    const db = openDatabase(join(dataDir, DATABASE_FILE));
    db.exec(
      "CREATE TRIGGER fail_writes BEFORE INSERT ON mind_updates BEGIN SELECT RAISE(ABORT, 'disk full'); END",
    );

    copy.type(" LOST");
    await vi.waitFor(() => expect(copy.failures).toHaveLength(1));
    db.exec("DROP TRIGGER fail_writes");
    db.close();
    copy.type(" AFTER");
    await vi.waitFor(async () =>
      expect(await copy.stored()).toBe("<paragraph>BASE LOST AFTER</paragraph>"),
    );
    copy.core.close();

    expect(copy.failures).toHaveLength(1);
    expect(String(copy.failures[0])).toContain("disk full");
    expect(await afterRestart(dataDir, copy.mindId)).toBe("<paragraph>BASE LOST AFTER</paragraph>");
  });

  // protects=a change that never reached the core is sent again inside the whole copy, so it and what follows are stored with nothing to report; fails_when=the lost change, and every change after it, never reach storage; no_claim=none beyond protects.
  test("a change that never reached the core goes again in the whole copy", async () => {
    const dataDir = await createTempDataFolder();
    const copy = await openCopy(dataDir);

    bridge.dropping = 1;
    copy.type(" ONE");
    copy.type(" TWO");
    await vi.waitFor(async () =>
      expect(await copy.stored()).toBe("<paragraph>BASE ONE TWO</paragraph>"),
    );
    copy.core.close();

    expect(copy.failures).toEqual([]);
    expect(await afterRestart(dataDir, copy.mindId)).toBe("<paragraph>BASE ONE TWO</paragraph>");
  });
});
