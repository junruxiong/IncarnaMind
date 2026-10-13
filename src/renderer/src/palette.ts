/**
 * ⌘K: Minds, Documents, Folders and Tags by name and the registry's commands,
 * matched here as the User types, then the words inside Minds and Documents,
 * which the core finds once typing pauses.
 */
import { create } from "zustand";
import type {
  Document,
  DocumentGroupAssignment,
  DocumentKind,
  DocumentTag,
  DocumentTextMatch,
  LibraryGroup,
  Mind,
  MindTextMatch,
  Tag,
  TextRange,
} from "../../core/api";
import type { CommandId } from "../../shared/commands";
import { displayName } from "../../shared/documentNames";
import { formatLocation } from "../../shared/locations";
import { mindName } from "../../shared/naming";
import type { TagColour } from "../../shared/tagColours";
import { shortcutLabel } from "./commands/keymap";
import {
  COMMAND_GROUPS,
  type CommandContext,
  type CommandGroup,
  commands,
  isEnabled,
  labelOf,
} from "./commands/registry";
import { noteHolding } from "./editor/noteKeymap";
import { cursorAtBlock } from "./editor/notePlace";
import { KIND_LABELS } from "./fileInfo";
import { matchName, type NameMatch, queryWords } from "./matcher";
import type { ObjectRef, Translate } from "./objects";
import { useAppStore } from "./store";

interface PaletteState {
  open: boolean;
  show(): void;
  hide(): void;
}

export const usePalette = create<PaletteState>()((set) => ({
  open: false,
  show: () => set({ open: true }),
  hide: () => set({ open: false }),
}));

/** Commands run from ⌘K act on the window; the selection comes with #230. */
export const PALETTE_CONTEXT: CommandContext = { selection: [], focus: null, surface: "palette" };

export type PaletteItem =
  | {
      kind: "object";
      key: string;
      ref: ObjectRef;
      name: string;
      ranges: TextRange[];
      meta: string;
      documentKind?: DocumentKind;
      tagColour?: TagColour;
    }
  | {
      kind: "command";
      key: string;
      id: CommandId;
      name: string;
      ranges: TextRange[];
      shortcut: string | null;
    }
  | {
      kind: "note";
      key: string;
      mindId: string;
      blockId: string;
      name: string;
      meta: string;
      excerpt: string;
      highlights: TextRange[];
    }
  | {
      kind: "passage";
      key: string;
      documentId: string;
      documentKind: DocumentKind;
      name: string;
      meta: string;
      pageFrom: number | null;
      pageTo: number | null;
      excerpt: string;
      highlights: TextRange[];
    };

export type PaletteGroupId =
  | "recent"
  | "minds"
  | "documents"
  | "foldersAndTags"
  | "actions"
  | "notes"
  | "documentText";

export interface PaletteGroup {
  id: PaletteGroupId;
  items: PaletteItem[];
}

/** A Document as the palette reads it. */
export type NamedDocument = Pick<Document, "id" | "name" | "title" | "kind"> & {
  tags: readonly Pick<DocumentTag, "tagId">[];
};

/** What names are found among, as the palette reads them. */
export interface NamedThings {
  minds: readonly Mind[];
  documents: readonly NamedDocument[];
  folders: readonly Pick<LibraryGroup, "id" | "name">[];
  assignments: readonly Pick<DocumentGroupAssignment, "documentId" | "groupId">[];
  tags: readonly Pick<Tag, "id" | "name" | "colour">[];
}

/** How many of each group show; the rest is a word or two more away. */
const SHOWN = { names: 5, actions: 6, recent: 5, words: 5 } as const;

/** What is typed: the words, and whether ">" asks for actions only, as in VS Code. */
export function readQuery(query: string): {
  text: string;
  words: string[];
  actionsOnly: boolean;
} {
  const trimmed = query.trim();
  const actionsOnly = trimmed.startsWith(">");
  const text = actionsOnly ? trimmed.slice(1).trim() : trimmed;
  return { text, words: queryWords(text), actionsOnly };
}

/** The words inside are searched for once the User has typed this much. */
export const searchesWords = (text: string) => text.length >= 2 || /\p{Script=Han}/u.test(text);

const best = <T extends { match: NameMatch }>(found: T[], count: number) =>
  found.sort((a, b) => b.match.score - a.match.score).slice(0, count);

const count = (
  t: Translate,
  n: number,
  one: Parameters<Translate>[0],
  many: Parameters<Translate>[0],
) => t(n === 1 ? one : many, { count: n });

/** A Mind's name, or "Untitled". */
const nameOfMind = (mind: Mind, t: Translate) => mindName(mind) || t("mind.untitled");

function mindItem(mind: Mind, match: NameMatch | null, folders: Map<string, string>, t: Translate) {
  const folder = mind.folderId ? folders.get(mind.folderId) : undefined;
  return {
    kind: "object",
    key: `mind:${mind.id}`,
    ref: { kind: "mind", id: mind.id },
    name: nameOfMind(mind, t),
    ranges: match?.ranges ?? [],
    meta: [t("palette.kind.mind"), folder].filter(Boolean).join(" · "),
  } satisfies PaletteItem;
}

/** Minds, Documents, Folders and Tags whose names hold every word typed, best first. */
export function nameGroups(
  words: readonly string[],
  things: NamedThings,
  t: Translate,
): PaletteGroup[] {
  const folders = new Map(things.folders.map((folder) => [folder.id, folder.name]));
  const folderOf = new Map(
    things.assignments.flatMap((each) => (each.groupId ? [[each.documentId, each.groupId]] : [])),
  );

  const minds = best(
    things.minds.flatMap((mind) => {
      const match = matchName(words, nameOfMind(mind, t));
      return match ? [{ mind, match }] : [];
    }),
    SHOWN.names,
  ).map(({ mind, match }) => mindItem(mind, match, folders, t));

  const documents = best(
    things.documents.flatMap((document) => {
      const match = matchName(words, displayName(document));
      return match ? [{ document, match }] : [];
    }),
    SHOWN.names,
  ).map(({ document, match }): PaletteItem => {
    const groupId = folderOf.get(document.id);
    return {
      kind: "object",
      key: `document:${document.id}`,
      ref: { kind: "document", id: document.id },
      name: displayName(document),
      ranges: match.ranges,
      meta: [t(KIND_LABELS[document.kind]), groupId ? folders.get(groupId) : undefined]
        .filter(Boolean)
        .join(" · "),
      documentKind: document.kind,
    };
  });

  const inFolder = new Map<string, number>();
  for (const groupId of folderOf.values()) inFolder.set(groupId, (inFolder.get(groupId) ?? 0) + 1);
  const tagged = new Map<string, number>();
  for (const document of things.documents) {
    for (const link of document.tags) tagged.set(link.tagId, (tagged.get(link.tagId) ?? 0) + 1);
  }
  const foldersAndTags = best(
    [
      ...things.folders.flatMap((folder) => {
        const match = matchName(words, folder.name);
        if (!match) return [];
        const documents = inFolder.get(folder.id) ?? 0;
        const item: PaletteItem = {
          kind: "object",
          key: `folder:${folder.id}`,
          ref: { kind: "folder", id: folder.id },
          name: folder.name,
          ranges: match.ranges,
          meta: [
            t("palette.kind.folder"),
            count(t, documents, "library.count.one", "library.count"),
          ].join(" · "),
        };
        return [{ item, match }];
      }),
      ...things.tags.flatMap((tag) => {
        const match = matchName(words, tag.name);
        if (!match) return [];
        const documents = tagged.get(tag.id) ?? 0;
        const item: PaletteItem = {
          kind: "object",
          key: `tag:${tag.id}`,
          ref: { kind: "tag", id: tag.id },
          name: tag.name,
          ranges: match.ranges,
          meta: [
            t("palette.kind.tag"),
            count(t, documents, "palette.tag.count.one", "palette.tag.count"),
          ].join(" · "),
          tagColour: tag.colour,
        };
        return [{ item, match }];
      }),
    ],
    SHOWN.names,
  ).map(({ item }) => item);

  return [
    { id: "minds" as const, items: minds },
    { id: "documents" as const, items: documents },
    { id: "foldersAndTags" as const, items: foldersAndTags },
  ].filter((group) => group.items.length > 0);
}

/** With nothing typed: the Minds edited last. */
export function recentGroup(things: NamedThings, t: Translate): PaletteGroup[] {
  const folders = new Map(things.folders.map((folder) => [folder.id, folder.name]));
  const items = [...things.minds]
    .sort((a, b) => b.updatedAt.localeCompare(a.updatedAt))
    .slice(0, SHOWN.recent)
    .map((mind) => mindItem(mind, null, folders, t));
  return items.length > 0 ? [{ id: "recent", items }] : [];
}

/** With nothing typed, what makes something new comes first, then the menus' order. */
const browsingOrder = (group: CommandGroup) =>
  group === "main" ? -1 : COMMAND_GROUPS.indexOf(group);

/**
 * The registry's commands the palette can run here, with their shortcuts: those
 * whose name or keywords hold every word typed, best first, or all of them
 * with nothing typed.
 */
export function actionGroup(
  words: readonly string[],
  ctx: CommandContext,
  t: Translate,
): PaletteGroup[] {
  const offered = commands
    .list(ctx)
    .filter((command) => command.palette !== false && isEnabled(command, ctx));
  const items = offered.flatMap((command) => {
    const name = labelOf(command, ctx, t);
    const byName = words.length > 0 ? matchName(words, name) : null;
    const byKeyword =
      byName || words.length === 0
        ? null
        : (command.keywords ?? [])
            .map((key) => matchName(words, t(key)))
            .find((match) => match !== null);
    if (words.length > 0 && !byName && !byKeyword) return [];
    const item: PaletteItem = {
      kind: "command",
      key: `command:${command.id}`,
      id: command.id,
      name,
      ranges: byName?.ranges ?? [],
      shortcut: command.shortcut ? shortcutLabel(command.id) : null,
    };
    // A match by keyword ranks below one by name.
    const score = byName?.score ?? (byKeyword ? byKeyword.score - 4000 : 0);
    return [{ item, score, group: command.group }];
  });
  if (words.length > 0) items.sort((a, b) => b.score - a.score);
  else items.sort((a, b) => browsingOrder(a.group) - browsingOrder(b.group));
  const shown = (words.length > 0 ? items.slice(0, SHOWN.actions) : items).map(({ item }) => item);
  return shown.length > 0 ? [{ id: "actions", items: shown }] : [];
}

/** "In your notes": the Minds the core found the words in, with the Block that holds them. */
export function noteGroup(
  found: readonly MindTextMatch[],
  minds: readonly Mind[],
  t: Translate,
): PaletteGroup[] {
  const byId = new Map(minds.map((mind) => [mind.id, mind]));
  const items = found.flatMap((match): PaletteItem[] => {
    const mind = byId.get(match.mindId);
    if (!mind) return [];
    return [
      {
        kind: "note",
        key: `note:${match.mindId}`,
        mindId: match.mindId,
        blockId: match.blockId,
        name: nameOfMind(mind, t),
        meta: match.heading,
        excerpt: match.excerpt,
        highlights: match.highlights,
      },
    ];
  });
  return items.length > 0 ? [{ id: "notes", items: items.slice(0, SHOWN.words) }] : [];
}

/** "In your Documents": the Documents the core found the words in, with the Passage that holds them. */
export function documentTextGroup(
  found: readonly DocumentTextMatch[],
  documents: readonly NamedDocument[],
  t: Translate,
): PaletteGroup[] {
  const byId = new Map(documents.map((document) => [document.id, document]));
  const items = found.flatMap((match): PaletteItem[] => {
    const document = byId.get(match.documentId);
    if (!document) return [];
    const unit = document.kind === "pdf" ? "page" : document.kind === "pptx" ? "slide" : null;
    const meta =
      unit && match.pageFrom !== null
        ? formatLocation(
            { kind: unit, from: match.pageFrom, to: match.pageTo ?? match.pageFrom },
            t,
          )
        : "";
    return [
      {
        kind: "passage",
        key: `passage:${match.passageId}`,
        documentId: match.documentId,
        documentKind: document.kind,
        name: displayName(document),
        meta,
        pageFrom: match.pageFrom,
        pageTo: match.pageTo,
        excerpt: match.excerpt,
        highlights: match.highlights,
      },
    ];
  });
  return items.length > 0 ? [{ id: "documentText", items: items.slice(0, SHOWN.words) }] : [];
}

/** Resolves with the open note that holds the Block, once its Mind shows; null if it doesn't. */
function noteShowing(blockId: string, timeoutMs = 5_000) {
  const until = performance.now() + timeoutMs;
  return new Promise<ReturnType<typeof noteHolding>>((resolve) => {
    const look = () => {
      const editor = noteHolding(blockId);
      if (editor || performance.now() > until) resolve(editor);
      else requestAnimationFrame(look);
    };
    look();
  });
}

/**
 * Does what choosing an item does: opens what it names, or runs its command.
 * `inTab` (⌘↵) opens a Document in a tab of its own rather than beside the Mind.
 */
export async function runItem(item: PaletteItem, { inTab = false } = {}): Promise<void> {
  const store = useAppStore.getState();
  switch (item.kind) {
    case "object": {
      const id: CommandId =
        inTab && item.ref.kind === "document" ? "document.openTab" : `${item.ref.kind}.open`;
      await commands.run(id, { ...PALETTE_CONTEXT, focus: item.ref });
      return;
    }
    case "command":
      await commands.run(item.id, PALETTE_CONTEXT);
      return;
    case "note": {
      store.openMind(item.mindId);
      const editor = await noteShowing(item.blockId);
      if (editor) cursorAtBlock(editor, item.blockId);
      return;
    }
    case "passage": {
      const first = item.highlights[0];
      store.openDocument(
        {
          documentId: item.documentId,
          ...(item.pageFrom !== null && {
            pageFrom: item.pageFrom,
            pageTo: item.pageTo ?? item.pageFrom,
          }),
          ...(first && { quote: item.excerpt.slice(first.start, first.end) }),
        },
        { as: inTab ? "here" : "list" },
      );
      return;
    }
  }
}
