/**
 * The changes the undo stack takes back, each with the core call that undoes
 * it and the one that does it again (docs/designs/interaction-system.md, C.6):
 * a rename by the other name, a delete by `restore`, Tags by putting their
 * links back as they were. Each says what happened in the words of the undo
 * line, and what it was, for the Edit menu.
 */
import type { Document, RestoreRef, TagLinkState } from "../../core/api";
import { core } from "./core";
import type { Translate } from "./objects";
import type { Undoable } from "./undo";

/** Renamed: undone by the old name, done again by the new one. */
export function renamed(
  t: Translate,
  shown: string,
  rename: (name: string) => Promise<unknown>,
  { before, after }: { before: string; after: string },
): Undoable {
  return {
    message: t("object.renamed", { name: shown }),
    name: t("undo.name.rename"),
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
  t: Translate,
  shown: string,
  refs: RestoreRef[],
  remove: () => Promise<unknown>,
): Undoable {
  const count = refs.length;
  return {
    message:
      count > 1 ? t("object.deleted.documents", { count }) : t("object.deleted", { name: shown }),
    name:
      count > 1
        ? t("undo.name.delete.documents", { count })
        : t("undo.name.delete", { name: shown }),
    undo: () => core.restore(refs),
    redo: async () => {
      await remove();
    },
  };
}

/** How each Document carries the Tag now, to put back with `revertDocumentTags`. */
export function tagLinks(
  documents: readonly Pick<Document, "id" | "tags">[],
  documentIds: readonly string[],
  tagId: string,
): TagLinkState[] {
  return documentIds.map((documentId) => {
    const link = documents
      .find((document) => document.id === documentId)
      ?.tags.find((each) => each.tagId === tagId);
    return {
      documentId,
      tagId,
      link: link
        ? { source: link.source, confidence: link.confidence, needsReview: link.needsReview }
        : null,
    };
  });
}

/**
 * A Tag added to or taken off Documents: undone by putting each link back as it
 * was, with who chose it and no removal recorded, so automatic tagging carries on
 * as before; done again as the User did it.
 */
export function retagged(
  t: Translate,
  change: { added: boolean; tag: string; names: readonly string[] },
  before: TagLinkState[],
  again: () => Promise<unknown>,
): Undoable {
  const { added, tag, names } = change;
  const [only] = names;
  const message =
    only !== undefined && names.length === 1
      ? t(added ? "undo.tagAdded.one" : "undo.tagRemoved.one", { name: only, tag })
      : t(added ? "undo.tagAdded" : "undo.tagRemoved", { count: names.length, tag });
  return {
    message,
    name: t(added ? "undo.name.tagAdded" : "undo.name.tagRemoved"),
    undo: async () => {
      await core.revertDocumentTags(before);
    },
    redo: async () => {
      await again();
    },
  };
}
