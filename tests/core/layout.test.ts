import { join } from "node:path";
import { describe, expect, test } from "vitest";
import {
  DATABASE_FILE,
  InvalidInputError,
  type LayoutChange,
  NotFoundError,
  type TabItem,
  type WindowLayout,
} from "../../src/core";
import { openDatabase } from "../../src/core/storage";
import { createTempDataFolder, startCore } from "../helpers/core";
import { addAndProcess, createSourceFolder, writeSourceFile } from "../helpers/documents";

/*
 * Tabs and groups (#264): everything that opens is a tab, in up to three
 * groups side by side, kept by the core per window and restored on restart.
 */

const WINDOW = "main";

/** What a layout shows, by name: each group's tabs, `*` before the shown one, `~` on a preview, `^` on a pinned one. */
function describeLayout(layout: WindowLayout, names: Record<string, string>): string[] {
  return layout.groups.map((group) =>
    group.tabs
      .map((tab) => {
        const name =
          tab.item.kind === "library"
            ? "Library"
            : (names[tab.item.kind === "mind" ? tab.item.mindId : tab.item.documentId] ?? "?");
        return `${tab.id === group.activeTabId ? "*" : ""}${tab.pinned ? "^" : ""}${tab.preview ? "~" : ""}${name}`;
      })
      .join(" "),
  );
}

const focusedIndex = (layout: WindowLayout) =>
  layout.groups.findIndex((group) => group.id === layout.focusedGroupId);

const tabOf = (layout: WindowLayout, item: TabItem) => {
  for (const group of layout.groups) {
    const tab = group.tabs.find((each) => JSON.stringify(each.item) === JSON.stringify(item));
    if (tab) return { group, tab };
  }
  throw new Error(`No tab shows ${JSON.stringify(item)}.`);
};

async function setUp() {
  const dataDir = await createTempDataFolder();
  const core = startCore(dataDir);
  const names: Record<string, string> = {};
  const mind = async (title: string) => {
    const created = await core.createMind({ title });
    names[created.id] = title;
    return { kind: "mind", mindId: created.id } as const;
  };
  const source = await createSourceFolder();
  const documents = await addAndProcess(core, [
    await writeSourceFile(source, "Report.md", "# Report\n\nParks are cooler at night.\n"),
    await writeSourceFile(source, "Paper.md", "# Paper\n\nCooling reaches one park width.\n"),
    await writeSourceFile(source, "Notes.txt", "Notes from the field.\n"),
  ]);
  const [report, paper, notes] = documents.map((document) => {
    names[document.id] = document.name;
    return { kind: "document", documentId: document.id } as const;
  }) as [TabItem, TabItem, TabItem];
  const change = (layoutChange: LayoutChange) => core.changeLayout(WINDOW, layoutChange);
  const shown = (layout: WindowLayout) => describeLayout(layout, names);
  return { core, dataDir, mind, report, paper, notes, change, shown, names };
}

describe("Tabs and groups", () => {
  test("a new window has one empty group", async () => {
    const { core } = await setUp();

    const layout = await core.getLayout(WINDOW);

    expect(layout.groups).toHaveLength(1);
    expect(layout.groups[0]?.tabs).toEqual([]);
    expect(layout.groups[0]?.activeTabId).toBeNull();
    expect(layout.focusedGroupId).toBe(layout.groups[0]?.id);
    expect(layout.reopenable).toBe(0);
  });

  test("a Mind opens in a new tab after the shown one, and opening it again shows its tab", async () => {
    const { mind, change, shown } = await setUp();
    const alpha = await mind("Alpha");
    const beta = await mind("Beta");
    const gamma = await mind("Gamma");

    await change({ type: "open", item: alpha });
    const both = await change({ type: "open", item: beta });
    expect(shown(both)).toEqual(["Alpha *Beta"]);
    expect(shown(await change({ type: "show", tabId: tabOf(both, alpha).tab.id }))).toEqual([
      "*Alpha Beta",
    ]);

    // After the shown tab, never replacing it.
    expect(shown(await change({ type: "open", item: gamma }))).toEqual(["Alpha *Gamma Beta"]);
    expect(shown(await change({ type: "open", item: beta }))).toEqual(["Alpha Gamma *Beta"]);
  });

  test("a Mind that isn't there doesn't open", async () => {
    const { change } = await setUp();

    await expect(change({ type: "open", item: { kind: "mind", mindId: "gone" } })).rejects.toThrow(
      NotFoundError,
    );
  });

  test("a Citation opens its Document as a preview to the right, the focus staying in the Mind; the next replaces it", async () => {
    const { mind, change, shown, report, paper, notes } = await setUp();
    const lit = await mind("Lit review");
    const opened = await change({ type: "open", item: lit });
    const from = opened.focusedGroupId;
    const cite = (item: TabItem, quote: string) =>
      change({
        type: "open",
        item,
        where: { in: "side", from },
        preview: true,
        focus: false,
        place: { pageFrom: 1, quote, citation: { check: "found", number: 1 } },
      });

    let layout = await cite(report, "Parks are cooler");
    expect(shown(layout)).toEqual(["*Lit review", "*~Report"]);
    expect(layout.focusedGroupId).toBe(from);
    expect(layout.groups[0]?.width).toBeCloseTo(0.5);
    const preview = tabOf(layout, report).tab;
    expect(preview.place).toEqual({
      pageFrom: 1,
      quote: "Parks are cooler",
      citation: { check: "found", number: 1 },
    });
    expect(preview.opened).toBe(1);

    // Three Citations in a row leave one tab: the preview, showing the last.
    await cite(paper, "Cooling reaches");
    layout = await cite(notes, "Notes");
    expect(shown(layout)).toEqual(["*Lit review", "*~Notes"]);
    expect(layout.groups[1]?.tabs[0]?.id).toBe(preview.id);

    // Kept, the next Citation opens beside it.
    layout = await change({ type: "keep", tabId: preview.id });
    layout = await cite(report, "Parks are cooler");
    expect(shown(layout)).toEqual(["*Lit review", "Notes *~Report"]);
    const openedBefore = tabOf(layout, notes).tab.opened;

    // A Citation to a Document already open in another group is shown there, at its quote.
    layout = await change({
      type: "open",
      item: notes,
      where: { in: "side", from },
      preview: true,
      focus: false,
      place: { quote: "field" },
    });
    expect(shown(layout)).toEqual(["*Lit review", "*Notes ~Report"]);
    expect(tabOf(layout, notes).tab.place).toEqual({ quote: "field" });
    expect(tabOf(layout, notes).tab.opened).toBe(openedBefore + 1);
  });

  test("Documents from a list open beside the Mind, or in the group showing a Document; the Library has one tab", async () => {
    const { mind, change, shown, report, paper } = await setUp();
    await change({ type: "open", item: await mind("Draft") });

    let layout = await change({
      type: "open",
      item: report,
      where: { in: "beside" },
      preview: true,
    });
    expect(shown(layout)).toEqual(["*Draft", "*~Report"]);
    expect(focusedIndex(layout)).toBe(1);
    // The focused group shows a Document: the next one replaces its preview there.
    layout = await change({ type: "open", item: paper, where: { in: "beside" }, preview: true });
    expect(shown(layout)).toEqual(["*Draft", "*~Paper"]);

    layout = await change({ type: "open", item: { kind: "library" } });
    expect(shown(layout)).toEqual(["*Draft", "~Paper *Library"]);
    layout = await change({ type: "focus", groupId: layout.groups[0]?.id as string });
    layout = await change({ type: "open", item: { kind: "library" }, where: { in: "side" } });
    expect(shown(layout)).toEqual(["*Draft", "~Paper *Library"]);
    expect(focusedIndex(layout)).toBe(1);
  });

  test("tabs move between groups and split off new ones, three at most; a group whose last tab goes goes too", async () => {
    const { mind, change, shown } = await setUp();
    const alpha = await mind("Alpha");
    const beta = await mind("Beta");
    const gamma = await mind("Gamma");
    for (const item of [alpha, beta, gamma]) await change({ type: "open", item });
    let layout = await change({ type: "open", item: alpha });

    // ⌘\ copies the shown tab into a new group on the right.
    layout = await change({
      type: "split",
      tabId: tabOf(layout, alpha).tab.id,
      side: "right",
      copy: true,
    });
    expect(shown(layout)).toEqual(["*Alpha Beta Gamma", "*Alpha"]);
    expect(focusedIndex(layout)).toBe(1);

    // Dragged to a group's left edge, Gamma moves into a new group there.
    layout = await change({
      type: "split",
      tabId: tabOf(layout, gamma).tab.id,
      side: "left",
      copy: false,
      groupId: layout.groups[0]?.id as string,
    });
    expect(shown(layout)).toEqual(["*Gamma", "*Alpha Beta", "*Alpha"]);
    const widths = layout.groups.map((group) => group.width);
    expect(widths.reduce((sum, width) => sum + width, 0)).toBeCloseTo(1);

    // Three groups at most.
    const full = await change({
      type: "split",
      tabId: tabOf(layout, beta).tab.id,
      side: "right",
      copy: true,
    });
    expect(shown(full)).toEqual(shown(layout));

    // Dropped on another group's strip, Beta moves there, before Alpha; the tab is shown and focused.
    layout = await change({
      type: "move",
      tabId: tabOf(layout, beta).tab.id,
      groupId: layout.groups[2]?.id as string,
      before: layout.groups[2]?.tabs[0]?.id ?? null,
    });
    expect(shown(layout)).toEqual(["*Gamma", "*Alpha", "*Beta Alpha"]);
    expect(focusedIndex(layout)).toBe(2);

    // Moved where it is open already, a tab goes, and the one there is shown.
    layout = await change({
      type: "move",
      tabId: layout.groups[1]?.tabs[0]?.id as string,
      groupId: layout.groups[2]?.id as string,
      before: null,
    });
    expect(shown(layout)).toEqual(["*Gamma", "Beta *Alpha"]);

    // Closing a group's last tab removes the group; its width goes to its neighbour.
    layout = await change({ type: "close", tabId: tabOf(layout, gamma).tab.id });
    expect(shown(layout)).toEqual(["Beta *Alpha"]);
    expect(layout.groups[0]?.width).toBe(1);
  });

  test("dragging a preview keeps it; double-click, a scroll or a selection keeps it through `keep`", async () => {
    const { mind, change, report } = await setUp();
    let layout = await change({ type: "open", item: await mind("Draft") });
    layout = await change({ type: "open", item: report, where: { in: "side" }, preview: true });
    const tab = tabOf(layout, report);

    layout = await change({ type: "move", tabId: tab.tab.id, groupId: tab.group.id, before: null });

    expect(tabOf(layout, report).tab.preview).toBe(false);
  });

  test("pinned tabs stay at the left of their group; a tab dropped among them is pinned", async () => {
    const { mind, change, shown } = await setUp();
    const a = await mind("A");
    const b = await mind("B");
    const c = await mind("C");
    let layout = await change({ type: "open", item: a });
    layout = await change({ type: "open", item: b });
    layout = await change({ type: "open", item: c });
    expect(shown(layout)).toEqual(["A B *C"]);

    layout = await change({ type: "pin", tabId: tabOf(layout, c).tab.id, pinned: true });
    expect(shown(layout)).toEqual(["*^C A B"]);

    layout = await change({
      type: "move",
      tabId: tabOf(layout, b).tab.id,
      groupId: layout.groups[0]?.id as string,
      before: tabOf(layout, c).tab.id,
    });
    expect(shown(layout)).toEqual(["*^B ^C A"]);

    layout = await change({ type: "pin", tabId: tabOf(layout, b).tab.id, pinned: false });
    expect(shown(layout)).toEqual(["^C *B A"]);
  });

  test("closed tabs reopen where they were, newest first, without those of deleted Minds", async () => {
    const { core, mind, change, shown } = await setUp();
    const a = await mind("A");
    const b = await mind("B");
    const c = await mind("C");
    const d = await mind("D");
    for (const item of [a, b, c, d]) await change({ type: "open", item });
    let layout = await change({ type: "open", item: b });

    layout = await change({ type: "closeOthers", tabId: tabOf(layout, b).tab.id });
    expect(shown(layout)).toEqual(["*B"]);
    expect(layout.reopenable).toBe(3);

    if (a.kind !== "mind") throw new Error("A Mind.");
    await core.deleteMind(a.mindId);
    layout = await change({ type: "reopen" });
    layout = await change({ type: "reopen" });
    expect(shown(layout)).toEqual(["B C *D"]);
    // A's tab was the last to reopen, but its Mind went: nothing is left to reopen.
    layout = await change({ type: "reopen" });
    expect(shown(layout)).toEqual(["B C *D"]);
    expect(layout.reopenable).toBe(0);
  });

  test("a deleted Mind's tab closes in every window, with an event", async () => {
    const { core, mind } = await setUp();
    const alpha = await mind("Alpha");
    const beta = await mind("Beta");
    await core.changeLayout("main", { type: "open", item: alpha });
    await core.changeLayout("main", { type: "open", item: beta });
    await core.changeLayout("second", { type: "open", item: alpha });
    const events: string[] = [];
    core.on("layout.changed", ({ windowId, layout }) =>
      events.push(`${windowId}:${layout.groups[0]?.tabs.length}`),
    );

    if (alpha.kind !== "mind") throw new Error("A Mind.");
    await core.deleteMind(alpha.mindId);

    expect(events.sort()).toEqual(["main:1", "second:0"]);
    expect((await core.getLayout("main")).groups[0]?.tabs.map((tab) => tab.item)).toEqual([beta]);
  });

  test("groups, tabs, previews, pins, widths and the focused group survive a restart; deleted Minds and Documents don't", async () => {
    const { core, dataDir, mind, change, shown, report, paper, names } = await setUp();
    const alpha = await mind("Alpha");
    const beta = await mind("Beta");
    await change({ type: "open", item: alpha });
    let layout = await change({ type: "open", item: beta });
    await change({ type: "pin", tabId: tabOf(layout, beta).tab.id, pinned: true });
    await change({
      type: "open",
      item: report,
      where: { in: "side" },
      place: { pageFrom: 2, quote: "Parks" },
    });
    await change({ type: "open", item: paper, where: { in: "group" }, preview: true });
    layout = await change({ type: "resize", widths: [600, 400] });
    const before = shown(layout);
    expect(before).toEqual(["*^Beta Alpha", "Report *~Paper"]);
    core.close();

    const restarted = startCore(dataDir);
    const after = await restarted.getLayout(WINDOW);
    expect(describeLayout(after, names)).toEqual(before);
    expect(after.groups.map((group) => group.width)).toEqual([0.6, 0.4]);
    expect(focusedIndex(after)).toBe(1);
    expect(tabOf(after, report).tab.place).toEqual({ pageFrom: 2, quote: "Parks" });

    if (alpha.kind !== "mind" || paper.kind !== "document") throw new Error("Kinds.");
    await restarted.deleteMind(alpha.mindId);
    await restarted.deleteDocument(paper.documentId);
    restarted.close();

    const again = startCore(dataDir);
    expect(describeLayout(await again.getLayout(WINDOW), names)).toEqual(["*^Beta", "*Report"]);
  });

  test("the Minds and Library open before groups come back as the first start's tabs", async () => {
    const dataDir = await createTempDataFolder();
    const first = startCore(dataDir);
    const alpha = await first.createMind({ title: "Alpha" });
    const beta = await first.createMind({ title: "Beta" });
    first.close();
    // As the version before groups kept them.
    const db = openDatabase(join(dataDir, DATABASE_FILE));
    const at = new Date().toISOString();
    for (const [key, value] of [
      ["openMinds", [alpha.id, "deleted-since", beta.id]],
      ["activeMind", beta.id],
      ["libraryTab", "open"],
    ] as const) {
      db.run(
        "INSERT INTO device_settings (id, key, value, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
        [crypto.randomUUID(), key, JSON.stringify(value), at, at],
      );
    }
    db.close();

    const core = startCore(dataDir);
    const layout = await core.getLayout(WINDOW);
    expect(describeLayout(layout, { [alpha.id]: "Alpha", [beta.id]: "Beta" })).toEqual([
      "Library Alpha *Beta",
    ]);
  });

  test("resize and equalize keep the widths shares of 1; bad input is refused", async () => {
    const { mind, change } = await setUp();
    let layout = await change({ type: "open", item: await mind("Alpha") });
    layout = await change({
      type: "split",
      tabId: layout.groups[0]?.tabs[0]?.id as string,
      side: "right",
      copy: true,
    });

    layout = await change({ type: "resize", widths: [3, 1] });
    expect(layout.groups.map((group) => group.width)).toEqual([0.75, 0.25]);
    layout = await change({ type: "equalize" });
    expect(layout.groups.map((group) => group.width)).toEqual([0.5, 0.5]);

    for (const bad of [
      { type: "resize", widths: [1] },
      { type: "resize", widths: [1, -1] },
      { type: "close", tabId: "no-such-tab" },
      { type: "focus", groupId: "no-such-group" },
      { type: "open", item: { kind: "web" } },
      { type: "split", tabId: layout.groups[0]?.tabs[0]?.id, side: "up", copy: true },
      { type: "teleport" },
    ]) {
      await expect(change(bad as unknown as LayoutChange)).rejects.toThrow(InvalidInputError);
    }
  });

  test("each window keeps its own layout, and a window is named by a short ID", async () => {
    const { core, mind } = await setUp();
    const alpha = await mind("Alpha");

    await core.changeLayout("main", { type: "open", item: alpha });

    expect((await core.getLayout("other")).groups[0]?.tabs).toEqual([]);
    expect((await core.getLayout("main")).groups[0]?.tabs).toHaveLength(1);
    await expect(core.getLayout("../main")).rejects.toThrow(InvalidInputError);
  });
});
