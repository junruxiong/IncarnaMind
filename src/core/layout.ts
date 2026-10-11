import { randomUUID } from "node:crypto";
import {
  type CitationCheck,
  type DocumentPlace,
  type LayoutChange,
  type LayoutGroup,
  type LayoutTab,
  MAX_TAB_GROUPS,
  type OpenWhere,
  type TabCitation,
  type TabItem,
  type WindowLayout,
} from "./api";
import { InvalidInputError, isRecord, NotFoundError } from "./errors";

/*
 * Each window's tabs and groups (VS Code's editor groups): kept here, saved
 * per window with the device's settings, and changed only through
 * `change`, so every client sees the same layout and it survives a restart.
 */

/** What is kept of a window's layout. */
interface Layout {
  groups: LayoutGroup[];
  focusedGroupId: string;
}

/** A tab closed in this session, and where it was, for ⇧⌘T. */
interface ClosedTab {
  tab: LayoutTab;
  groupId: string;
  groupIndex: number;
  index: number;
}

/** How many closed tabs ⇧⌘T can bring back. */
const MAX_CLOSED = 30;
/** Enough for any window; it bounds what one saved layout can hold. */
const MAX_TABS_PER_GROUP = 200;
const MAX_TEXT = 20_000;

const storageKey = (windowId: string) => `windowLayout.${windowId}`;

// ---------------------------------------------------------------------------
// Reading input: the renderer's changes, and layouts saved by any version.

const isId = (value: unknown): value is string =>
  typeof value === "string" && value.trim() !== "" && value.length <= 200;

const isUnit = (value: unknown): value is number =>
  typeof value === "number" && Number.isInteger(value) && value >= 1;

const CHECKS: readonly CitationCheck[] = ["checking", "found", "not-found", "cant-check"];

function parseItem(value: unknown): TabItem {
  if (isRecord(value)) {
    if (value.kind === "mind" && isId(value.mindId)) return { kind: "mind", mindId: value.mindId };
    if (value.kind === "library") return { kind: "library" };
    if (value.kind === "document" && isId(value.documentId)) {
      return { kind: "document", documentId: value.documentId };
    }
  }
  throw new InvalidInputError("A tab shows a Mind, the Library or a Document.");
}

function parseCitation(value: unknown): TabCitation {
  if (!isRecord(value) || !CHECKS.includes(value.check as CitationCheck)) {
    throw new InvalidInputError("A Citation's place needs its check.");
  }
  const citation: TabCitation = { check: value.check as CitationCheck };
  if (value.number !== undefined) {
    if (!isUnit(value.number)) throw new InvalidInputError("A Citation's number counts from 1.");
    citation.number = value.number;
  }
  for (const key of ["label", "contentHash"] as const) {
    const text = value[key];
    if (text === undefined) continue;
    if (typeof text !== "string" || text.length > 500) {
      throw new InvalidInputError(`A Citation's ${key} is short text.`);
    }
    citation[key] = text;
  }
  return citation;
}

function parsePlace(value: unknown): DocumentPlace {
  if (!isRecord(value)) throw new InvalidInputError("A place in a Document is an object.");
  const place: DocumentPlace = {};
  for (const key of ["pageFrom", "pageTo"] as const) {
    if (value[key] === undefined) continue;
    if (!isUnit(value[key])) throw new InvalidInputError("Units count from 1.");
    place[key] = value[key];
  }
  if (value.quote !== undefined) {
    if (typeof value.quote !== "string" || value.quote.length > MAX_TEXT) {
      throw new InvalidInputError("A quote is text.");
    }
    place.quote = value.quote;
  }
  if (value.citation !== undefined) place.citation = parseCitation(value.citation);
  return place;
}

function parseWhere(value: unknown): OpenWhere {
  if (value === undefined) return { in: "place" };
  if (isRecord(value)) {
    if (value.in === "place" || value.in === "beside") return { in: value.in };
    if (value.in === "side" && (value.from === undefined || isId(value.from))) {
      return value.from === undefined ? { in: "side" } : { in: "side", from: value.from };
    }
    if (
      value.in === "group" &&
      (value.groupId === undefined || isId(value.groupId)) &&
      (value.before === undefined || value.before === null || isId(value.before))
    ) {
      return {
        in: "group",
        ...(value.groupId !== undefined && { groupId: value.groupId }),
        ...(value.before !== undefined && { before: value.before }),
      };
    }
  }
  throw new InvalidInputError("Open it in place, to the side, beside or in a group.");
}

const optionalBoolean = (value: unknown, name: string): boolean | undefined => {
  if (value === undefined || typeof value === "boolean") return value;
  throw new InvalidInputError(`"${name}" is true or false.`);
};

function requireId(value: unknown, name: string): string {
  if (!isId(value)) throw new InvalidInputError(`The change needs its ${name}.`);
  return value;
}

/** A change as the renderer sent it, checked. */
export function parseLayoutChange(value: unknown): LayoutChange {
  if (!isRecord(value)) throw new InvalidInputError("A layout change is an object.");
  switch (value.type) {
    case "open": {
      const preview = optionalBoolean(value.preview, "preview");
      const focus = optionalBoolean(value.focus, "focus");
      return {
        type: "open",
        item: parseItem(value.item),
        where: parseWhere(value.where),
        ...(preview !== undefined && { preview }),
        ...(focus !== undefined && { focus }),
        ...(value.place !== undefined && { place: parsePlace(value.place) }),
      };
    }
    case "show":
    case "close":
    case "closeOthers":
    case "closeToTheRight":
    case "keep":
      return { type: value.type, tabId: requireId(value.tabId, "tab") };
    case "closeGroup":
    case "focus":
      return { type: value.type, groupId: requireId(value.groupId, "group") };
    case "reopen":
    case "equalize":
      return { type: value.type };
    case "move":
      return {
        type: "move",
        tabId: requireId(value.tabId, "tab"),
        groupId: requireId(value.groupId, "group"),
        before: value.before === null ? null : requireId(value.before, "tab to go before"),
      };
    case "split": {
      if (value.side !== "left" && value.side !== "right") {
        throw new InvalidInputError("A split goes left or right.");
      }
      const copy = optionalBoolean(value.copy, "copy");
      if (copy === undefined) throw new InvalidInputError("A split copies the tab or moves it.");
      return {
        type: "split",
        tabId: requireId(value.tabId, "tab"),
        side: value.side,
        copy,
        ...(value.groupId !== undefined && { groupId: requireId(value.groupId, "group") }),
      };
    }
    case "pin": {
      const pinned = optionalBoolean(value.pinned, "pinned");
      if (pinned === undefined) throw new InvalidInputError("Pin or unpin it.");
      return { type: "pin", tabId: requireId(value.tabId, "tab"), pinned };
    }
    case "resize": {
      const widths = value.widths;
      if (
        !Array.isArray(widths) ||
        !widths.every((width) => typeof width === "number" && Number.isFinite(width) && width > 0)
      ) {
        throw new InvalidInputError("Widths are positive numbers, one per group.");
      }
      return { type: "resize", widths: widths as number[] };
    }
    default:
      throw new InvalidInputError("That isn't a layout change.");
  }
}

/** A tab saved by any version: what this one can't read is left out. */
function readSavedTab(value: unknown): LayoutTab | null {
  if (!isRecord(value) || !isId(value.id)) return null;
  try {
    const item = parseItem(value.item);
    return {
      id: value.id,
      item,
      preview: value.preview === true && item.kind === "document",
      pinned: value.pinned === true,
      place: value.place == null ? null : parsePlace(value.place),
      opened: isUnit(value.opened) ? value.opened : 0,
    };
  } catch {
    return null;
  }
}

function readSavedLayout(value: unknown): Layout | null {
  if (!isRecord(value) || !Array.isArray(value.groups)) return null;
  const groups: LayoutGroup[] = [];
  const seen = new Set<string>();
  for (const saved of value.groups.slice(0, MAX_TAB_GROUPS)) {
    if (!isRecord(saved) || !isId(saved.id) || !Array.isArray(saved.tabs)) continue;
    const tabs = saved.tabs
      .slice(0, MAX_TABS_PER_GROUP)
      .map(readSavedTab)
      .filter((tab): tab is LayoutTab => tab !== null && !seen.has(tab.id));
    for (const tab of tabs) seen.add(tab.id);
    const width = typeof saved.width === "number" && saved.width > 0 ? saved.width : 1;
    groups.push({
      id: saved.id,
      tabs,
      activeTabId: typeof saved.activeTabId === "string" ? saved.activeTabId : null,
      width,
    });
  }
  if (groups.length === 0) return null;
  const focused = groups.find((group) => group.id === value.focusedGroupId) ?? groups[0];
  return { groups, focusedGroupId: (focused as LayoutGroup).id };
}

// ---------------------------------------------------------------------------
// The changes.

const sameItem = (a: TabItem, b: TabItem): boolean =>
  a.kind === b.kind &&
  (a.kind === "mind"
    ? a.mindId === (b as typeof a).mindId
    : a.kind === "document"
      ? a.documentId === (b as typeof a).documentId
      : true);

function emptyLayout(newId: () => string): Layout {
  const group: LayoutGroup = { id: newId(), tabs: [], activeTabId: null, width: 1 };
  return { groups: [group], focusedGroupId: group.id };
}

const focusedGroup = (layout: Layout): LayoutGroup =>
  (layout.groups.find((group) => group.id === layout.focusedGroupId) ??
    layout.groups[0]) as LayoutGroup;

function requireGroup(layout: Layout, groupId: string): LayoutGroup {
  const group = layout.groups.find((each) => each.id === groupId);
  if (!group) throw new InvalidInputError("This window has no such group.");
  return group;
}

function requireTab(layout: Layout, tabId: string): { group: LayoutGroup; tab: LayoutTab } {
  for (const group of layout.groups) {
    const tab = group.tabs.find((each) => each.id === tabId);
    if (tab) return { group, tab };
  }
  throw new InvalidInputError("This window has no such tab.");
}

/** Where it is open, the focused group first. */
function findOpen(
  layout: Layout,
  item: TabItem,
  except?: LayoutGroup,
): { group: LayoutGroup; tab: LayoutTab } | null {
  const focused = focusedGroup(layout);
  for (const group of [focused, ...layout.groups.filter((each) => each !== focused)]) {
    if (group === except) continue;
    const tab = group.tabs.find((each) => sameItem(each.item, item));
    if (tab) return { group, tab };
  }
  return null;
}

/**
 * Puts a tab in a group at `index`, or after the shown tab. Among the pinned
 * tabs it is pinned; after them it isn't.
 */
function insert(group: LayoutGroup, tab: LayoutTab, index?: number): void {
  const pinned = group.tabs.filter((each) => each.pinned).length;
  const active = group.tabs.findIndex((each) => each.id === group.activeTabId);
  const wanted = index ?? Math.max(active === -1 ? group.tabs.length : active + 1, pinned);
  const at = Math.max(0, Math.min(wanted, group.tabs.length));
  tab.pinned = at < pinned || (tab.pinned && at === pinned);
  group.tabs.splice(at, 0, tab);
}

/** Takes a tab out of its group; the tab after it (or else before it) is shown if it was. */
function takeOut(group: LayoutGroup, tab: LayoutTab): number {
  const index = group.tabs.indexOf(tab);
  group.tabs.splice(index, 1);
  if (group.activeTabId === tab.id) {
    group.activeTabId = group.tabs[Math.min(index, group.tabs.length - 1)]?.id ?? null;
  }
  return index;
}

/** A new, empty group at `at`, taking half of `beside`'s width. */
function addGroup(layout: Layout, at: number, beside: LayoutGroup, newId: () => string) {
  const group: LayoutGroup = { id: newId(), tabs: [], activeTabId: null, width: beside.width / 2 };
  beside.width /= 2;
  layout.groups.splice(at, 0, group);
  return group;
}

/** The group right of `from`, made if there is room, else the one left of it. */
function sideGroup(layout: Layout, from: LayoutGroup, newId: () => string): LayoutGroup {
  if (from.tabs.length === 0) return from;
  const index = layout.groups.indexOf(from);
  const right = layout.groups[index + 1];
  if (right) return right;
  if (layout.groups.length < MAX_TAB_GROUPS) return addGroup(layout, index + 1, from, newId);
  return layout.groups[index - 1] ?? from;
}

function show(layout: Layout, group: LayoutGroup, tab: LayoutTab, focus: boolean): void {
  group.activeTabId = tab.id;
  if (focus) layout.focusedGroupId = group.id;
}

/**
 * Keeps a layout whole: no empty group beside others (its width goes to its
 * neighbour, and so does the focus), a shown tab in every group that has
 * tabs, the pinned tabs first, and widths that add up to 1.
 */
function tidy(layout: Layout): void {
  for (let index = 0; index < layout.groups.length && layout.groups.length > 1; ) {
    const group = layout.groups[index] as LayoutGroup;
    if (group.tabs.length > 0) {
      index++;
      continue;
    }
    layout.groups.splice(index, 1);
    const neighbour = (layout.groups[index - 1] ?? layout.groups[index]) as LayoutGroup;
    neighbour.width += group.width;
    if (layout.focusedGroupId === group.id) layout.focusedGroupId = neighbour.id;
  }
  for (const group of layout.groups) {
    group.tabs = [
      ...group.tabs.filter((tab) => tab.pinned),
      ...group.tabs.filter((tab) => !tab.pinned),
    ];
    if (!group.tabs.some((tab) => tab.id === group.activeTabId)) {
      group.activeTabId = group.tabs[0]?.id ?? null;
    }
  }
  if (!layout.groups.some((group) => group.id === layout.focusedGroupId)) {
    layout.focusedGroupId = (layout.groups[0] as LayoutGroup).id;
  }
  const total = layout.groups.reduce((sum, group) => sum + group.width, 0);
  for (const group of layout.groups) {
    group.width = total > 0 && group.width > 0 ? group.width / total : 1 / layout.groups.length;
  }
}

function open(
  layout: Layout,
  change: Extract<LayoutChange, { type: "open" }>,
  newId: () => string,
): void {
  const { item, place } = change;
  const where = change.where ?? { in: "place" };
  const focus = change.focus !== false;
  const preview = change.preview === true && item.kind === "document";
  const reveal = (group: LayoutGroup, tab: LayoutTab) => {
    if (change.preview === false) tab.preview = false;
    if (place) {
      tab.place = place;
      tab.opened += 1;
    }
    show(layout, group, tab, focus);
  };

  const from =
    where.in === "side" && where.from !== undefined
      ? requireGroup(layout, where.from)
      : focusedGroup(layout);
  // The Library has one tab per window; anything else is revealed where it is open,
  // except in the group it is opened from to the side, which keeps what it shows.
  if (where.in !== "group" || item.kind === "library") {
    const open = findOpen(
      layout,
      item,
      where.in === "side" && item.kind !== "library" ? from : undefined,
    );
    if (open) {
      reveal(open.group, open.tab);
      return;
    }
  }

  let target: LayoutGroup;
  let before: string | null | undefined;
  if (where.in === "group") {
    target =
      where.groupId === undefined ? focusedGroup(layout) : requireGroup(layout, where.groupId);
    before = where.before;
  } else if (where.in === "side") {
    target = sideGroup(layout, from, newId);
  } else if (where.in === "beside") {
    const shown = from.tabs.find((tab) => tab.id === from.activeTabId);
    target = shown?.item.kind === "document" ? from : sideGroup(layout, from, newId);
  } else {
    target = from;
  }

  const here = target.tabs.find((tab) => sameItem(tab.item, item));
  const replaced = preview ? target.tabs.find((tab) => tab.preview) : undefined;
  if (here) {
    reveal(target, here);
  } else if (replaced) {
    replaced.item = item;
    replaced.place = place ?? null;
    replaced.opened += 1;
    show(layout, target, replaced, focus);
  } else {
    const tab: LayoutTab = {
      id: newId(),
      item,
      preview,
      pinned: false,
      place: place ?? null,
      opened: place ? 1 : 0,
    };
    const at = before ? target.tabs.findIndex((each) => each.id === before) : -1;
    insert(target, tab, before === undefined ? undefined : at === -1 ? target.tabs.length : at);
    show(layout, target, tab, focus);
  }
}

/** A window's layout and the tabs it closed, with every change it can take. */
class WindowState {
  constructor(
    readonly layout: Layout,
    private readonly newId: () => string,
    readonly closed: ClosedTab[] = [],
  ) {}

  private close(group: LayoutGroup, tab: LayoutTab): void {
    const groupIndex = this.layout.groups.indexOf(group);
    const index = takeOut(group, tab);
    this.closed.push({ tab, groupId: group.id, groupIndex, index });
    if (this.closed.length > MAX_CLOSED) this.closed.shift();
  }

  /** Closes tabs right to left, so ⇧⌘T brings them back where they were. */
  private closeAll(group: LayoutGroup, tabs: LayoutTab[]): void {
    for (const tab of [...tabs].reverse()) this.close(group, tab);
  }

  apply(change: LayoutChange, valid: (item: TabItem) => boolean): void {
    const { layout, newId } = this;
    switch (change.type) {
      case "open":
        open(layout, change, newId);
        break;
      case "show": {
        const { group, tab } = requireTab(layout, change.tabId);
        show(layout, group, tab, true);
        break;
      }
      case "close": {
        const { group, tab } = requireTab(layout, change.tabId);
        this.close(group, tab);
        break;
      }
      case "closeOthers": {
        const { group, tab } = requireTab(layout, change.tabId);
        this.closeAll(
          group,
          group.tabs.filter((each) => each !== tab && !each.pinned),
        );
        show(layout, group, tab, true);
        break;
      }
      case "closeToTheRight": {
        const { group, tab } = requireTab(layout, change.tabId);
        const after = group.tabs.slice(group.tabs.indexOf(tab) + 1);
        this.closeAll(
          group,
          after.filter((each) => !each.pinned),
        );
        break;
      }
      case "closeGroup": {
        const group = requireGroup(layout, change.groupId);
        this.closeAll(group, [...group.tabs]);
        break;
      }
      case "reopen":
        this.reopen(valid);
        break;
      case "move":
        this.move(change.tabId, change.groupId, change.before);
        break;
      case "split":
        this.split(change);
        break;
      case "keep":
        requireTab(layout, change.tabId).tab.preview = false;
        break;
      case "pin": {
        const { group, tab } = requireTab(layout, change.tabId);
        group.tabs.splice(group.tabs.indexOf(tab), 1);
        tab.pinned = change.pinned;
        if (change.pinned) tab.preview = false;
        // At the end of the pinned tabs: the last one pinned, or the first one unpinned.
        const pinned = group.tabs.filter((each) => each.pinned).length;
        group.tabs.splice(pinned, 0, tab);
        break;
      }
      case "focus":
        layout.focusedGroupId = requireGroup(layout, change.groupId).id;
        break;
      case "resize":
        if (change.widths.length !== layout.groups.length) {
          throw new InvalidInputError("Give one width per group.");
        }
        layout.groups.forEach((group, index) => {
          group.width = change.widths[index] as number;
        });
        break;
      case "equalize":
        for (const group of layout.groups) group.width = 1;
        break;
    }
    tidy(layout);
  }

  private reopen(valid: (item: TabItem) => boolean): void {
    const { layout } = this;
    let closed = this.closed.pop();
    while (closed && !valid(closed.tab.item)) closed = this.closed.pop();
    if (!closed) return;
    const { tab, index, groupId, groupIndex } = closed;
    // Opened again since it closed: that tab is shown, not a second one. The
    // Library has one tab per window; anything else one per group.
    const library = tab.item.kind === "library" ? findOpen(layout, tab.item) : null;
    if (library) {
      show(layout, library.group, library.tab, true);
      return;
    }
    const last = layout.groups.at(-1) as LayoutGroup;
    const only = layout.groups.length === 1 && last.tabs.length === 0 ? last : undefined;
    // Its group, if it is still there; else a new one where it was, if there is room.
    const at = Math.min(groupIndex, layout.groups.length);
    const neighbour = (layout.groups[at] ?? last) as LayoutGroup;
    const group =
      only ??
      layout.groups.find((each) => each.id === groupId) ??
      (layout.groups.length < MAX_TAB_GROUPS
        ? addGroup(layout, at, neighbour, this.newId)
        : neighbour);
    const twin = group.tabs.find((each) => sameItem(each.item, tab.item));
    if (twin) {
      show(layout, group, twin, true);
      return;
    }
    insert(group, tab, index);
    show(layout, group, tab, true);
  }

  private move(tabId: string, groupId: string, wanted: string | null): void {
    const { layout } = this;
    const { group: from, tab } = requireTab(layout, tabId);
    const to = requireGroup(layout, groupId);
    tab.preview = false;
    const before = wanted === tab.id ? (from.tabs[from.tabs.indexOf(tab) + 1]?.id ?? null) : wanted;
    takeOut(from, tab);
    const twin = from === to ? undefined : to.tabs.find((each) => sameItem(each.item, tab.item));
    if (twin) {
      // Already open there: that tab is shown, and this one goes.
      twin.preview = false;
      show(layout, to, twin, true);
      return;
    }
    const index = before === null ? -1 : to.tabs.findIndex((each) => each.id === before);
    insert(to, tab, index === -1 ? to.tabs.length : index);
    show(layout, to, tab, true);
  }

  private split(change: Extract<LayoutChange, { type: "split" }>): void {
    const { layout } = this;
    if (layout.groups.length >= MAX_TAB_GROUPS) return;
    const { group: from, tab } = requireTab(layout, change.tabId);
    const beside = change.groupId === undefined ? from : requireGroup(layout, change.groupId);
    // The Library has one tab per window: it moves instead.
    const copy = change.copy && tab.item.kind !== "library";
    if (!copy && from === beside && from.tabs.length === 1) return;
    const index = layout.groups.indexOf(beside);
    const group = addGroup(layout, change.side === "left" ? index : index + 1, beside, this.newId);
    const moved: LayoutTab = copy ? { ...structuredClone(tab), id: this.newId() } : tab;
    if (!copy) takeOut(from, tab);
    moved.preview = false;
    moved.pinned = false;
    group.tabs.push(moved);
    show(layout, group, moved, true);
  }
}

// ---------------------------------------------------------------------------

export interface LayoutDependencies {
  /** Reads and writes a per-device value (the settings' own store). */
  readValue(key: string): unknown;
  writeValue(key: string, value: unknown): void;
  mindExists(mindId: string): boolean;
  documentExists(documentId: string): boolean;
  changed(windowId: string, layout: WindowLayout): void;
  newId?: () => string;
}

export type Layouts = ReturnType<typeof createLayouts>;

const WINDOW_ID = /^[A-Za-z0-9_-]{1,64}$/;

function parseWindowId(value: unknown): string {
  if (typeof value !== "string" || !WINDOW_ID.test(value)) {
    throw new InvalidInputError("A window is named by a short ID.");
  }
  return value;
}

export function createLayouts(deps: LayoutDependencies) {
  const newId = deps.newId ?? randomUUID;
  const windows = new Map<string, WindowState>();

  const exists = (item: TabItem) =>
    item.kind === "library" ||
    (item.kind === "mind" ? deps.mindExists(item.mindId) : deps.documentExists(item.documentId));

  /**
   * The tabs the app had before groups (the Minds open, and the Library's tab),
   * so the first start after the update opens as the last one closed.
   */
  const fromEarlierVersion = (): Layout | null => {
    const minds = deps.readValue("openMinds");
    const active = deps.readValue("activeMind");
    const library = deps.readValue("libraryTab");
    const tabs: LayoutTab[] = [];
    const tab = (item: TabItem): LayoutTab => ({
      id: newId(),
      item,
      preview: false,
      pinned: false,
      place: null,
      opened: 0,
    });
    if (library === "open" || library === "shown") tabs.push(tab({ kind: "library" }));
    if (Array.isArray(minds)) {
      for (const mindId of minds) if (isId(mindId)) tabs.push(tab({ kind: "mind", mindId }));
    }
    if (tabs.length === 0) return null;
    const shown =
      library === "shown"
        ? tabs[0]
        : tabs.find((each) => each.item.kind === "mind" && each.item.mindId === active);
    const group: LayoutGroup = {
      id: newId(),
      tabs,
      activeTabId: (shown ?? tabs[0])?.id ?? null,
      width: 1,
    };
    return { groups: [group], focusedGroupId: group.id };
  };

  /** Leaves out the tabs of what isn't there any more; says whether any went. */
  const prune = (layout: Layout, keep: (item: TabItem) => boolean): boolean => {
    let removed = false;
    for (const group of layout.groups) {
      const kept = group.tabs.filter((tab) => keep(tab.item));
      removed ||= kept.length !== group.tabs.length;
      group.tabs = kept;
    }
    return removed;
  };

  const stateOf = (windowId: string): WindowState => {
    let state = windows.get(windowId);
    if (state) return state;
    const layout =
      readSavedLayout(deps.readValue(storageKey(windowId))) ??
      (windowId === "main" ? fromEarlierVersion() : null) ??
      emptyLayout(newId);
    prune(layout, exists);
    tidy(layout);
    state = new WindowState(layout, newId);
    windows.set(windowId, state);
    return state;
  };

  const view = (state: WindowState): WindowLayout => ({
    ...structuredClone(state.layout),
    reopenable: state.closed.length,
  });

  const save = (windowId: string, state: WindowState): WindowLayout => {
    deps.writeValue(storageKey(windowId), state.layout);
    deps.changed(windowId, view(state));
    return view(state);
  };

  return {
    get(windowId: unknown): WindowLayout {
      return view(stateOf(parseWindowId(windowId)));
    },

    change(windowId: unknown, input: unknown): WindowLayout {
      const id = parseWindowId(windowId);
      const change = parseLayoutChange(input);
      if (change.type === "open" && change.item.kind === "mind") {
        if (!deps.mindExists(change.item.mindId)) throw new NotFoundError("That Mind isn't there.");
      }
      const state = stateOf(id);
      state.apply(change, exists);
      return save(id, state);
    },

    /**
     * Closes the tabs of Minds deleted since, in every window open. A deleted
     * Document's tab stays until closed: its viewer says what happened.
     */
    forgetDeletedMinds(): void {
      const keep = (item: TabItem) => item.kind !== "mind" || deps.mindExists(item.mindId);
      for (const [windowId, state] of windows) {
        if (!prune(state.layout, keep)) continue;
        tidy(state.layout);
        save(windowId, state);
      }
    },
  };
}
