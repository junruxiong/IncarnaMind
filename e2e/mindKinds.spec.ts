import { join } from "node:path";
import { expect, test } from "@playwright/test";
import * as Y from "yjs";
import type { CoreBridge } from "../src/core/api";
import { openDatabase } from "../src/core/storage";
import { createDataFolder, dismissChatSetup, launchApp, removeDataFolder } from "./app";

/*
 * A Mind a newer version wrote opens read-only, with a notice, and is never
 * written to; a Mind of a kind this version doesn't know opens as "Update
 * IncarnaMind to open this" and not in the editor (ADR-0003, amendment).
 */

let dataDir: string;
test.beforeEach(async () => {
  dataDir = await createDataFolder();
});
test.afterEach(async () => {
  await removeDataFolder(dataDir);
});

/** What a newer version would have left: a node type this version doesn't know, and a newer schema version. */
function newerContent(): number[] {
  const doc = new Y.Doc();
  const paragraph = new Y.XmlElement("paragraph");
  paragraph.insert(0, [new Y.XmlText("Written on another computer")]);
  const future = new Y.XmlElement("futureWidget");
  future.setAttribute("layout", "grid");
  doc.getXmlFragment("blocks").push([paragraph, future]);
  doc.getMap("settings").set("schemaVersion", 2);
  return Array.from(Y.encodeStateAsUpdate(doc));
}

test("a Mind written by a newer version opens read-only with a notice, and its stored state is unchanged", async () => {
  const { app, window } = await launchApp(dataDir);
  await dismissChatSetup(window);
  const mindId = await window.evaluate(async (update) => {
    const core = (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind;
    const mind = await core.createMind({ title: "From the other computer" });
    await core.applyMindUpdate(mind.id, new Uint8Array(update));
    await core.closeMind(mind.id);
    return mind.id;
  }, newerContent());
  const stored = () =>
    window.evaluate(async (id) => {
      const core = (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind;
      return Array.from((await core.openMind(id)).state);
    }, mindId);
  const before = await stored();

  await window.getByTestId("mind-list-item").filter({ hasText: "From the other computer" }).click();
  const editor = window.getByTestId("mind-editor");
  await expect(editor).toContainText("Written on another computer");
  await expect(window.getByTestId("mind-notice-read-only")).toContainText("Read-only");
  await expect(window.getByTestId("mind-notice-read-only")).toContainText("Update IncarnaMind");
  await expect(editor).toHaveAttribute("contenteditable", "false");
  // No composer: nothing can be asked in it.
  await expect(window.getByTestId("composer-dock").getByRole("textbox")).toHaveCount(0);

  // Typing does nothing, and the stored state is exactly what it was.
  await editor.click();
  await window.keyboard.type("Lost words");
  await expect(editor).not.toContainText("Lost words");
  expect(await stored()).toEqual(before);
  await app.close();
});

test("a Mind of a kind this version doesn't know opens as Update IncarnaMind to open this", async () => {
  const first = await launchApp(dataDir);
  await dismissChatSetup(first.window);
  await first.window.evaluate(async () => {
    const core = (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind;
    await core.createMind({ title: "A deck" });
  });
  await first.app.close();
  const db = openDatabase(join(dataDir, "incarnamind.db"));
  db.run("UPDATE minds SET kind = 'deck' WHERE title = 'A deck'");
  db.close();

  const { app, window } = await launchApp(dataDir);
  await window.getByTestId("mind-list-item").filter({ hasText: "A deck" }).click();

  await expect(window.getByTestId("mind-notice-update")).toContainText(
    "Update IncarnaMind to open this",
  );
  await expect(window.getByTestId("mind-editor")).toHaveCount(0);
  await app.close();
});
