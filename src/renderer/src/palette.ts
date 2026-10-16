/**
 * ⌘K: every object the registry finds by name (./finding) and the commands
 * that apply where the User is, matched here as the User types; then the words
 * inside Minds, Documents and conversations, which the core finds once typing
 * pauses; and, when nothing matches, asking about what was typed.
 */
import { create } from "zustand";
import type {
  ConversationMatch,
  Document,
  DocumentKind,
  DocumentTextMatch,
  Mind,
  MindTextMatch,
  TextRange,
} from "../../core/api";
import type { Language } from "../../core/language";
import type { CommandId } from "../../shared/commands";
import { displayName } from "../../shared/documentNames";
import { translate } from "../../shared/i18n";
import { formatLocation } from "../../shared/locations";
import { mindName } from "../../shared/naming";
import { focusComposer } from "./commands/assistant";
import { objectAt, shortcutLabel, surfaceOf } from "./commands/keymap";
import {
  COMMAND_GROUPS,
  type CommandContext,
  type CommandGroup,
  commands,
  isEnabled,
  labelOf,
} from "./commands/registry";
import { useComposer } from "./composer";
import { conversationLabel, openConversation } from "./conversations";
import { noteHolding } from "./editor/noteKeymap";
import { cursorAtBlock } from "./editor/notePlace";
import { type FindableKind, type FoundObject, findable } from "./finding";
import { currentTabOf } from "./layoutSlice";
import { matchName, type NameMatch, queryWords } from "./matcher";
import type { ObjectRef, Translate } from "./objects";
import { useSelection } from "./selection";
import { refKey } from "./selectionModel";
import { useAppStore } from "./store";

const NOTHING: CommandContext = { selection: [], focus: null, surface: "palette" };

/** What the tab in front shows, as an object. */
function objectInFront(): ObjectRef | null {
  const item = currentTabOf(useAppStore.getState().layout)?.item;
  if (item?.kind === "mind") return { kind: "mind", id: item.mindId };
  if (item?.kind === "document") return { kind: "document", id: item.documentId };
  return null;
}

/**
 * What commands run from ⌘K act on: the selection while the focus is in the
 * list that holds it, else the row with the focus, else what the tab in front
 * shows, as a list's own keys and the shortcuts on the note decide.
 */
export function contextAt(active: Element | null): CommandContext {
  const { surface, items } = useSelection.getState();
  if (surface !== null && items.length > 0 && surfaceOf(active) === surface) {
    return { ...NOTHING, selection: items };
  }
  return { ...NOTHING, focus: objectAt(active) ?? objectInFront() };
}

interface PaletteState {
  open: boolean;
  /** What its commands act on: the window as it was when ⌘K opened. */
  context: CommandContext;
  show(): void;
  hide(): void;
}

export const usePalette = create<PaletteState>()((set) => ({
  open: false,
  context: NOTHING,
  show: () => set({ open: true, context: contextAt(document.activeElement) }),
  hide: () => set({ open: false }),
}));

export type PaletteItem =
  | ({ kind: "object"; key: string; ranges: TextRange[] } & FoundObject)
  | {
      kind: "command";
      key: string;
      id: CommandId;
      name: string;
      ranges: TextRange[];
      /** What it acts on, when one thing: "Supplier review: Aldermoor Freight". */
      meta: string;
      /** What it is for, under its name; "" for most. */
      purpose: string;
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
    }
  | {
      kind: "said";
      key: string;
      conversationId: string;
      /** The Answer that holds the words, to show; null when it is a Question. */
      answerId: string | null;
      name: string;
      meta: string;
      excerpt: string;
      highlights: TextRange[];
    }
  | { kind: "ask"; key: string; text: string; name: string; meta: string };

export interface PaletteGroup {
  /** A registered kind's group, or the palette's own: recent, actions, notes, documentText, conversationText, ask. */
  id: string;
  /** The registry's title for an object group; the others are the palette's own. */
  title?: FindableKind["group"]["title"];
  items: PaletteItem[];
}

/** The objects of each registered kind, read once for each change. */
export interface Findings {
  kinds: readonly { kind: FindableKind; objects: readonly FoundObject[] }[];
  /** An object's name, by `refKey`. */
  names: ReadonlyMap<string, string>;
}

export function findings(t: Translate): Findings {
  const kinds = findable.all().map((kind) => ({ kind, objects: kind.objects(t) }));
  const names = new Map<string, string>();
  for (const { objects } of kinds) {
    for (const object of objects) names.set(refKey(object.ref), object.name);
  }
  return { kinds, names };
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

/** A Mind's name, or "Untitled". */
const nameOfMind = (mind: Mind, t: Translate) => mindName(mind) || t("mind.untitled");

/** The registry's objects whose names hold every word typed, best first, by group. */
export function nameGroups(words: readonly string[], found: Findings): PaletteGroup[] {
  const groups = new Map<
    string,
    PaletteGroup & { matched: { item: PaletteItem; match: NameMatch }[] }
  >();
  for (const { kind, objects } of found.kinds) {
    const group = groups.get(kind.group.id) ?? {
      id: kind.group.id,
      title: kind.group.title,
      items: [],
      matched: [],
    };
    groups.set(kind.group.id, group);
    for (const object of objects) {
      const match = matchName(words, object.name);
      if (!match) continue;
      const key = refKey(object.ref);
      group.matched.push({ item: { ...object, kind: "object", key, ranges: match.ranges }, match });
    }
  }
  return [...groups.values()].flatMap(({ id, title, matched }) => {
    const items = best(matched, SHOWN.names).map(({ item }) => item);
    return items.length > 0 ? [{ id, title, items }] : [];
  });
}

/** With nothing typed: the Minds edited last. */
export function recentGroup(minds: readonly Mind[], found: Findings): PaletteGroup[] {
  const objects = new Map(
    found.kinds.flatMap(({ objects }) => objects.map((object) => [refKey(object.ref), object])),
  );
  const items = [...minds]
    .sort((a, b) => b.updatedAt.localeCompare(a.updatedAt))
    .slice(0, SHOWN.recent)
    .flatMap((mind): PaletteItem[] => {
      const key = refKey({ kind: "mind", id: mind.id });
      const object = objects.get(key);
      return object ? [{ ...object, kind: "object", key, ranges: [] }] : [];
    });
  return items.length > 0 ? [{ id: "recent", items }] : [];
}

/** With nothing typed, what makes something new comes first, then the menus' order. */
const browsingOrder = (group: CommandGroup) =>
  group === "main" ? -1 : COMMAND_GROUPS.indexOf(group);

/**
 * The registry's commands that apply here, with their shortcuts and, when they
 * act on one thing, its name: those whose name or keywords, in English or
 * Chinese, hold every word typed, best first, or all of them with nothing typed.
 */
export function actionGroup(
  words: readonly string[],
  ctx: CommandContext,
  language: Language,
  names: Findings["names"],
): PaletteGroup[] {
  const t: Translate = (key, params) => translate(language, key, params);
  const otherLanguage = language === "zh-CN" ? "en" : "zh-CN";
  const other: Translate = (key, params) => translate(otherLanguage, key, params);
  const targets = ctx.selection.length > 0 ? ctx.selection : ctx.focus ? [ctx.focus] : [];
  const [target] = targets;
  const targetName = targets.length === 1 && target ? (names.get(refKey(target)) ?? "") : "";
  const offered = commands
    .list(ctx)
    .filter((command) => command.palette !== false && isEnabled(command, ctx));
  const items = offered.flatMap((command) => {
    const name = labelOf(command, ctx, t);
    const byName = words.length > 0 ? matchName(words, name) : null;
    const elsewhere =
      byName || words.length === 0
        ? null
        : [
            labelOf(command, ctx, other),
            ...(command.keywords ?? []).flatMap((key) => [t(key), other(key)]),
          ]
            .map((each) => matchName(words, each))
            .find((match) => match !== null);
    if (words.length > 0 && !byName && !elsewhere) return [];
    const item: PaletteItem = {
      kind: "command",
      key: `command:${command.id}`,
      id: command.id,
      name,
      ranges: byName?.ranges ?? [],
      meta: command.appliesTo ? targetName : "",
      purpose: command.purpose ? t(command.purpose) : "",
      shortcut: command.shortcut ? shortcutLabel(command.id) : null,
    };
    // A match by keyword or in the other language ranks below one by its name.
    const score = byName?.score ?? (elsewhere ? elsewhere.score - 4000 : 0);
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

/** A Document as the palette reads it. */
export type NamedDocument = Pick<Document, "id" | "name" | "title" | "kind">;

/** Where a Passage found is, for a PDF's page or a deck's slide; "" for the other kinds. */
export function passagePlace(
  kind: DocumentKind,
  match: Pick<DocumentTextMatch, "pageFrom" | "pageTo">,
  t: Translate,
): string {
  const unit = kind === "pdf" ? "page" : kind === "pptx" ? "slide" : null;
  if (!unit || match.pageFrom === null) return "";
  return formatLocation(
    { kind: unit, from: match.pageFrom, to: match.pageTo ?? match.pageFrom },
    t,
  );
}

/** Where a Document found by its words opens: at the Passage, with the first words found marked. */
export function atPassage(
  found: Pick<DocumentTextMatch, "pageFrom" | "pageTo" | "excerpt" | "highlights">,
): { pageFrom?: number; pageTo?: number; quote?: string } {
  const first = found.highlights[0];
  return {
    ...(found.pageFrom !== null && {
      pageFrom: found.pageFrom,
      pageTo: found.pageTo ?? found.pageFrom,
    }),
    ...(first && { quote: found.excerpt.slice(first.start, first.end) }),
  };
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
    const meta = passagePlace(document.kind, match, t);
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

/** "In your conversations": the conversations whose messages hold the words, with the first that does. */
export function conversationTextGroup(found: readonly ConversationMatch[]): PaletteGroup[] {
  const items = found.flatMap(({ conversation, messages }): PaletteItem[] => {
    const [message] = messages;
    if (!message) return [];
    return [
      {
        kind: "said",
        key: `said:${conversation.id}`,
        conversationId: conversation.id,
        answerId: message.kind === "answer" ? message.messageId : null,
        name: conversationLabel(conversation),
        meta: "",
        excerpt: message.excerpt,
        highlights: message.highlights,
      },
    ];
  });
  return items.length > 0 ? [{ id: "conversationText", items: items.slice(0, SHOWN.words) }] : [];
}

/** When nothing matches: asking about what was typed, in the assistant's open conversation. */
export function askGroup(text: string, conversation: string, t: Translate): PaletteGroup[] {
  return [
    {
      id: "ask",
      items: [
        {
          kind: "ask",
          key: "ask",
          text,
          name: t("palette.ask", { text }),
          meta: t("palette.ask.in", { conversation }),
        },
      ],
    },
  ];
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
 * Does what choosing an item does: opens what it names, runs its command on
 * what ⌘K acts on, or asks. `inTab` (⌘↵) opens a Document in a tab of its own.
 */
export async function runItem(
  item: PaletteItem,
  { inTab = false, context = NOTHING }: { inTab?: boolean; context?: CommandContext } = {},
): Promise<void> {
  const store = useAppStore.getState();
  switch (item.kind) {
    case "object":
      await findable.of(item.ref.kind)?.open(item.ref, { inTab });
      return;
    case "command":
      await commands.run(item.id, context);
      return;
    case "note": {
      store.openMind(item.mindId);
      const editor = await noteShowing(item.blockId);
      if (editor) cursorAtBlock(editor, item.blockId);
      return;
    }
    case "passage":
      store.openDocument(
        { documentId: item.documentId, ...atPassage(item) },
        { as: inTab ? "here" : "list" },
      );
      return;
    case "said":
      await openConversation(item.conversationId, item.answerId ?? undefined);
      return;
    case "ask":
      focusComposer();
      useComposer.getState().requestAsk(item.text);
      return;
  }
}
