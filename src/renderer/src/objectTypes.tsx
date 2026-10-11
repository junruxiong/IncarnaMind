import { displayName } from "../../shared/documentNames";
import { conversationName } from "../../shared/naming";
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

const state = () => useAppStore.getState();
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
      const renamed = await core.renameConversation(ref.id, name);
      return {
        message: translateNow("object.renamed", {
          name: conversationName(renamed) || translateNow("conversation.untitled"),
        }),
        undo: async () => {
          await core.renameConversation(ref.id, before);
        },
      };
    },
    async remove(refs) {
      const message =
        refs.length === 1
          ? translateNow("object.deleted", {
              name: conversationLabel(refs[0]?.id ?? "", translateNow),
            })
          : translateNow("object.deleted.conversations", { count: refs.length });
      for (const ref of refs) await core.deleteConversation(ref.id);
      return {
        message,
        // Recently deleted gives them back, with every message.
        undo: () => core.restore(refs.map((ref) => ({ kind: "conversation", id: ref.id }))),
      };
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
