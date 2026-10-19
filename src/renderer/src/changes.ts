/**
 * The changes the undo stack takes back, each with the core call that undoes
 * it and the one that does it again (docs/designs/interaction-system.md, C.6):
 * a rename by the other name, a delete by `restore`, Tags by putting their
 * links back as they were. Each says what happened in the words of the undo
 * line, and what it was, for the Edit menu.
 */
import type { LinkedFolderLayout, RestoreRef, TagLinkState } from "../../core/api";
import { core } from "./core";
import type { Words } from "./lineNames";
import type { Translate } from "./objects";
import type { Undoable } from "./undo";

/** A name the User gave, or the words for what it is called when it has none. */
type Shown = string | Words;

const shownIn = (t: Translate, shown: Shown) => (typeof shown === "string" ? shown : shown(t));

/** Renamed: undone by the old name, done again by the new one. */
export function renamed(
  shown: Shown,
  rename: (name: string) => Promise<unknown>,
  { before, after }: { before: string; after: string },
): Undoable {
  return {
    said: (t) => t("object.renamed", { name: shownIn(t, shown) }),
    name: (t) => t("undo.name.rename"),
    undo: async () => {
      await rename(before);
    },
    redo: async () => {
      await rename(after);
    },
  };
}

/**
 * Sent to Recently deleted: undone by restoring it, done again by deleting it again.
 * Several are Documents, and are said by their count; one by its name.
 */
export function deleted(
  shown: Shown,
  refs: RestoreRef[],
  remove: () => Promise<unknown>,
): Undoable {
  const count = refs.length;
  return {
    said: (t) =>
      count > 1
        ? t("object.deleted.documents", { count })
        : t("object.deleted", { name: shownIn(t, shown) }),
    name: (t) =>
      count > 1
        ? t("undo.name.delete.documents", { count })
        : t("undo.name.delete", { name: shownIn(t, shown) }),
    undo: () => core.restore(refs),
    redo: async () => {
      await remove();
    },
  };
}

/**
 * A Mind made: undone by sending it to Recently deleted, as deleting it does, done again by
 * bringing it back as it is, with what was written in it since. One with a title is said by its name.
 */
export function mindMade(
  shown: string,
  remove: () => Promise<unknown>,
  bringBack: () => Promise<unknown>,
): Undoable {
  return {
    said: (t) => (shown ? t("undo.mindStarted", { name: shown }) : t("undo.mindMade")),
    name: (t) => t("menu.newMind"),
    undo: async () => {
      await remove();
    },
    redo: async () => {
      await bringBack();
    },
  };
}

/** A Linked folder paused or resumed: undone and done again by setting it as it was and as it is. */
export function linkedFolderPaused(
  shown: string,
  paused: boolean,
  setPaused: (paused: boolean) => Promise<unknown>,
): Undoable {
  return {
    said: (t) =>
      t(paused ? "undo.linkedFolderPaused" : "undo.linkedFolderResumed", { name: shown }),
    name: (t) => t(paused ? "linkedFolders.pause" : "linkedFolders.resume"),
    undo: async () => {
      await setPaused(!paused);
    },
    redo: async () => {
      await setPaused(paused);
    },
  };
}

/** A Linked folder shown as Folders or as a flat list: undone by the layout it had, done again by the new one. */
export function linkedFolderShown(
  shown: string,
  layouts: { before: LinkedFolderLayout; after: LinkedFolderLayout },
  setLayout: (layout: LinkedFolderLayout) => Promise<unknown>,
): Undoable {
  const { before, after } = layouts;
  const flat = after === "flat";
  return {
    said: (t) => t(flat ? "undo.linkedFolderFlat" : "undo.linkedFolderTree", { name: shown }),
    name: (t) => t(flat ? "linkedFolders.layout.flat" : "linkedFolders.layout.tree"),
    undo: async () => {
      await setLayout(before);
    },
    redo: async () => {
      await setLayout(after);
    },
  };
}

/**
 * A Tag added to or taken off Documents: undone by putting each link back as it
 * was, with who chose it and no removal recorded, so automatic tagging carries on
 * as before; done again as the User did it.
 */
export function retagged(
  change: { added: boolean; tag: string; names: readonly string[] },
  before: TagLinkState[],
  again: () => Promise<unknown>,
): Undoable {
  const { added, tag, names } = change;
  const [only] = names;
  const count = names.length;
  return {
    said: (t) =>
      only !== undefined && count === 1
        ? t(added ? "undo.tagAdded.one" : "undo.tagRemoved.one", { name: only, tag })
        : t(added ? "undo.tagAdded" : "undo.tagRemoved", { count, tag }),
    name: (t) => t(added ? "undo.name.tagAdded" : "undo.name.tagRemoved"),
    undo: async () => {
      await core.revertDocumentTags(before);
    },
    redo: async () => {
      await again();
    },
  };
}
