import { displayName } from "../../shared/documentNames";
import { conversationName } from "../../shared/naming";
import { deleted, renamed } from "./changes";
import { GripIcon } from "./components/icons";
import {
  ConversationLineIcon,
  DocumentLineIcon,
  FolderLineIcon,
  LooseDocumentsLineIcon,
  MindLineIcon,
  TagLineIcon,
} from "./components/lineIcons";
import { conversationById } from "./conversations";
import { core } from "./core";
import { findBlock } from "./editor/blockCommands";
import { noteHolding } from "./editor/noteKeymap";
import { translateNow } from "./i18n";
import { folderName } from "./linkedFolders";
import type { ObjectKind, ObjectType, Translate } from "./objects";
import { useAppStore } from "./store";
import { findTab, tabTitle } from "./tabs";
import { type Undoable, useUndo } from "./undo";

const state = () => useAppStore.getState();

/** A change made here goes on the undo stack, and back to whoever asked for it. */
const offered = (change: Undoable): Undoable => {
  useUndo.getState().push(change);
  return change;
};
const ICON = "size-4 shrink-0 text-ink-meta";

/** A conversation's name, as History shows it: "New chat" when it has none. */
const conversationLabel = (id: string, t: Translate) => {
  const conversation = conversationById(id);
  return conversation ? conversationName(conversation) || t("conversation.untitled") : "";
};

/**
 * How each kind is named and shown. Citations join once their menus give them
 * an identity (#223).
 */
export const objectTypes: Partial<Record<ObjectKind, ObjectType>> = {
  mind: {
    label: (ref, t) => tabTitle(state(), { kind: "mind", mindId: ref.id }, t),
    icon: () => <MindLineIcon className={ICON} />,
  },
  conversation: {
    label: (ref, t) => conversationLabel(ref.id, t),
    icon: () => <ConversationLineIcon className={ICON} />,
    async rename(ref, name) {
      const before = conversationById(ref.id)?.title ?? "";
      const conversation = await core.renameConversation(ref.id, name);
      const shown = conversationName(conversation) || translateNow("conversation.untitled");
      const rename = (title: string) => core.renameConversation(ref.id, title);
      return offered(renamed(translateNow, shown, rename, { before, after: conversation.title }));
    },
    async remove(refs) {
      const [only] = refs;
      const shown = only ? conversationLabel(only.id, translateNow) : "";
      const remove = async () => {
        for (const ref of refs) await core.deleteConversation(ref.id);
      };
      await remove();
      // Recently deleted gives them back, with every message.
      const change = deleted(
        translateNow,
        shown,
        refs.map((ref) => ({ kind: "conversation", id: ref.id })),
        remove,
      );
      return offered(
        refs.length === 1
          ? change
          : {
              ...change,
              message: translateNow("object.deleted.conversations", { count: refs.length }),
              name: translateNow("undo.name.delete.conversations", { count: refs.length }),
            },
      );
    },
  },
  document: {
    label: (ref) => {
      const document = state().documents.find((each) => each.id === ref.id);
      return document ? displayName(document) : "";
    },
    icon: (ref) => (
      <DocumentLineIcon
        kind={state().documents.find((each) => each.id === ref.id)?.kind ?? "text"}
        className={ICON}
      />
    ),
  },
  folder: {
    // An in-app Folder, or a folder of a Linked folder, as the Search scope names them.
    label: (ref) =>
      state().library?.groups.find((group) => group.id === ref.id)?.name ??
      state().folders.find((folder) => folder.id === ref.id)?.name ??
      "",
    icon: () => <FolderLineIcon className={ICON} />,
  },
  tag: {
    label: (ref) => state().tags.find((tag) => tag.id === ref.id)?.name ?? "",
    icon: () => <TagLineIcon className={ICON} />,
  },
  linkedFolder: {
    label: (ref) => {
      const linked = state().linkedFolders.find((each) => each.id === ref.id);
      return linked ? folderName(linked.path) : "";
    },
    icon: () => <FolderLineIcon className={ICON} />,
  },
  block: {
    label: (ref) => {
      const doc = noteHolding(ref.id)?.state.doc;
      return (doc && findBlock(doc, ref.id)?.node.textContent.trim()) ?? "";
    },
    icon: () => <GripIcon className={ICON} />,
  },
  tab: {
    label: (ref, t) => {
      const tab = findTab(state().layout, ref.id);
      return tab ? tabTitle(state(), tab.item, t) : "";
    },
    icon: (ref) => {
      const item = findTab(state().layout, ref.id)?.item;
      if (item?.kind === "library") return <LooseDocumentsLineIcon className={ICON} />;
      if (item?.kind !== "document") return <MindLineIcon className={ICON} />;
      const kind = state().documents.find((each) => each.id === item.documentId)?.kind;
      return <DocumentLineIcon kind={kind ?? "text"} className={ICON} />;
    },
  },
};
