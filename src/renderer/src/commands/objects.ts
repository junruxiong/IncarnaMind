import type { LinkedFolder } from "../../../core/api";
import { displayName } from "../../../shared/documentNames";
import { files } from "../core";
import { languageNow } from "../i18n";
import type { LibraryBridge } from "../libraryBridges";
import { fileStatusLabel, folderName, formatCount, revealLabelKey } from "../linkedFolders";
import { useRequests } from "../requests";
import { useAppStore } from "../store";
import { type Command, type CommandContext, needsHere } from "./registry";
import { documentsOf, everyDocument, idsOf } from "./targets";

const store = () => useAppStore.getState();
const reportError = (error: unknown) => store().reportError(error);
const RENAME_KEYS = { mac: "Enter", other: "F2" } as const;

const mindOf = (ctx: CommandContext) => {
  const id = idsOf(ctx, "mind")[0];
  return store().minds.find((each) => each.id === id);
};
const folderOf = (ctx: CommandContext) => {
  const id = idsOf(ctx, "folder")[0];
  return store().library?.groups.find((each) => each.id === id);
};
const linkedOf = (ctx: CommandContext): LinkedFolder | undefined => {
  const id = idsOf(ctx, "linkedFolder")[0];
  return store().linkedFolders.find((each) => each.id === id);
};

/** The Question a Document's menu asks about it. */
function bridgeOf(ctx: CommandContext): LibraryBridge | null {
  const [document] = documentsOf(ctx);
  if (!document) return null;
  return {
    kind: "documents",
    title: displayName(document),
    count: 1,
    scope: { folderIds: [], tagIds: [], documentIds: [document.id] },
  };
}

/** Why a Document's file can't be opened or shown now, if it can't. */
const fileReason = (ctx: CommandContext) => {
  for (const document of documentsOf(ctx)) {
    const state = fileStatusLabel(document.fileStatus);
    if (state) return { reason: state.reason };
  }
  return true as const;
};

const onFiles = (ctx: CommandContext) => everyDocument(ctx, (document) => !document.web);

export const mindCommands: readonly Command[] = [
  {
    id: "mind.open",
    label: "mind.open",
    group: "open",
    appliesTo: ["mind"],
    reversible: true,
    run: (ctx) => store().openMind(idsOf(ctx, "mind")[0] ?? ""),
  },
  {
    id: "mind.openSide",
    label: "mind.openSide",
    group: "open",
    appliesTo: ["mind"],
    reversible: true,
    run: (ctx) => store().openMind(idsOf(ctx, "mind")[0] ?? "", { side: true }),
  },
  {
    id: "mind.export",
    label: "mind.export",
    group: "main",
    appliesTo: ["mind"],
    reversible: true,
    run(ctx) {
      const mind = mindOf(ctx);
      if (mind) useRequests.getState().ask({ exportMind: mind });
    },
  },
  {
    id: "mind.moveTo",
    label: "mind.moveTo",
    rowKeys: "CmdOrCtrl+Shift+M",
    group: "move",
    appliesTo: ["mind"],
    reversible: true,
    ...needsHere("mind.moveTo"),
  },
  {
    id: "mind.rename",
    label: "mind.rename",
    rowKeys: RENAME_KEYS,
    group: "edit",
    appliesTo: ["mind"],
    reversible: true,
    ...needsHere("mind.rename"),
  },
  {
    id: "mind.delete",
    label: "mind.deleteAction",
    group: "danger",
    destructive: true,
    appliesTo: ["mind"],
    reversible: { confirm: "mind.delete.body" },
    run(ctx) {
      const mind = mindOf(ctx);
      if (mind) useRequests.getState().ask({ deleteMind: mind });
    },
  },
];

export const documentCommands: readonly Command[] = [
  {
    id: "document.open",
    label: "mind.open",
    group: "open",
    appliesTo: ["document"],
    reversible: true,
    run: (ctx) => store().openDocument({ documentId: idsOf(ctx, "document")[0] ?? "" }),
  },
  {
    id: "document.openTab",
    label: "document.openTab",
    group: "open",
    appliesTo: ["document"],
    reversible: true,
    run: (ctx) =>
      store().openDocument({ documentId: idsOf(ctx, "document")[0] ?? "" }, { as: "here" }),
  },
  {
    id: "document.ask",
    label: "documents.ask",
    group: "main",
    appliesTo: ["document"],
    // One Document, until a selection can be asked about (#225).
    when: (ctx) => documentsOf(ctx).length === 1,
    reversible: true,
    run(ctx) {
      const bridge = bridgeOf(ctx);
      if (bridge) return store().askAbout(bridge);
    },
  },
  {
    id: "document.startMind",
    label: "documents.startMind",
    group: "main",
    appliesTo: ["document"],
    when: (ctx) => documentsOf(ctx).length === 1,
    reversible: true,
    run(ctx) {
      const bridge = bridgeOf(ctx);
      if (bridge) return store().startMindFrom(bridge);
    },
  },
  {
    id: "document.retry",
    label: "documents.retryAction",
    group: "main",
    appliesTo: ["document"],
    // Only a failed one whose file is there to be read again.
    when: (ctx) =>
      everyDocument(
        ctx,
        (document) => document.status === "failed" && document.fileStatus === "available",
      ),
    reversible: true,
    run: async (ctx) => {
      await Promise.all(idsOf(ctx, "document").map((id) => store().retryDocument(id)));
    },
  },
  {
    id: "document.rename",
    label: "documents.renameAction",
    rowKeys: RENAME_KEYS,
    group: "edit",
    appliesTo: ["document"],
    reversible: true,
    ...needsHere("document.rename"),
  },
  {
    id: "document.openExternally",
    label: "documents.copy.open",
    group: "file",
    appliesTo: ["document"],
    when: onFiles,
    // Its file isn't where it was: the item says why.
    enabled: fileReason,
    reversible: true,
    run(ctx) {
      for (const id of idsOf(ctx, "document")) files.openDocumentExternally(id).catch(reportError);
    },
  },
  {
    id: "document.showInFolder",
    label: "documents.copy.showInFolder",
    group: "file",
    appliesTo: ["document"],
    when: onFiles,
    enabled: fileReason,
    reversible: true,
    run(ctx) {
      for (const id of idsOf(ctx, "document")) files.showDocumentInFolder(id).catch(reportError);
    },
  },
  {
    id: "document.delete",
    label: (ctx, t) =>
      t(
        documentsOf(ctx).every((document) => document.fileStatus === "missing")
          ? "documents.removeAction"
          : "documents.deleteAction",
      ),
    group: "danger",
    destructive: true,
    appliesTo: ["document"],
    // Until Undo replaces the question (#228), one Document at a time.
    when: (ctx) => documentsOf(ctx).length === 1,
    reversible: { confirm: "documents.delete.body" },
    run(ctx) {
      const [document] = documentsOf(ctx);
      if (document) useRequests.getState().ask({ deleteDocument: document });
    },
  },
];

export const folderCommands: readonly Command[] = [
  {
    id: "folder.open",
    label: "folder.open",
    group: "open",
    appliesTo: ["folder"],
    reversible: true,
    run: (ctx) => store().openLibrary(idsOf(ctx, "folder")[0]),
  },
  {
    id: "folder.newMind",
    label: "folder.newMindHere",
    group: "main",
    appliesTo: ["folder"],
    reversible: true,
    run: async (ctx) => {
      await store().createMind({ folderId: idsOf(ctx, "folder")[0] ?? null });
    },
  },
  {
    id: "folder.rename",
    label: "folder.rename",
    rowKeys: RENAME_KEYS,
    group: "edit",
    appliesTo: ["folder"],
    reversible: true,
    ...needsHere("folder.rename"),
  },
  {
    id: "folder.delete",
    label: "folder.delete",
    group: "danger",
    destructive: true,
    appliesTo: ["folder"],
    reversible: { confirm: "folder.delete.body" },
    run(ctx) {
      const folder = folderOf(ctx);
      if (folder) useRequests.getState().ask({ deleteFolder: folder });
    },
  },
];

export const tagCommands: readonly Command[] = [
  {
    id: "tag.open",
    label: "mind.open",
    group: "open",
    appliesTo: ["tag"],
    reversible: true,
    run(ctx) {
      // Its Documents, in the Library: the Tag filter the Library and the sidebar share.
      store().setTagFilter(idsOf(ctx, "tag").slice(0, 1));
      store().openLibrary();
    },
  },
  {
    // Renaming and deleting a Tag are in Manage Tags until Tags are edited in place (#205).
    id: "tag.manage",
    label: "tags.menu.manage",
    group: "edit",
    appliesTo: ["tag"],
    reversible: true,
    run: () => store().openTagsDialog(),
  },
];

const REVEAL = revealLabelKey(navigator.platform);

export const linkedFolderCommands: readonly Command[] = [
  {
    id: "linkedFolder.pause",
    label: (ctx, t) =>
      t(linkedOf(ctx)?.status === "paused" ? "linkedFolders.resume" : "linkedFolders.pause"),
    group: "main",
    appliesTo: ["linkedFolder"],
    reversible: true,
    run: async (ctx) => {
      const linked = linkedOf(ctx);
      if (linked) await store().setLinkedFolderPaused(linked.id, linked.status !== "paused");
    },
  },
  {
    id: "linkedFolder.layout",
    label: (ctx, t) =>
      t(
        linkedOf(ctx)?.layout === "flat"
          ? "linkedFolders.layout.tree"
          : "linkedFolders.layout.flat",
      ),
    group: "main",
    appliesTo: ["linkedFolder"],
    reversible: true,
    run: async (ctx) => {
      const linked = linkedOf(ctx);
      if (linked)
        await store().setLinkedFolderLayout(linked.id, linked.layout === "flat" ? "tree" : "flat");
    },
  },
  {
    id: "linkedFolder.download",
    label: (ctx, t) => {
      const count = linkedOf(ctx)?.onlineOnly.files ?? 0;
      return count === 1
        ? t("linkedFolders.downloadOnlineOnly.one")
        : t("linkedFolders.downloadOnlineOnly.other", { count: formatCount(count, languageNow()) });
    },
    group: "main",
    appliesTo: ["linkedFolder"],
    when(ctx) {
      const online = linkedOf(ctx)?.onlineOnly;
      return online !== undefined && !online.downloading && online.files > 0;
    },
    reversible: true,
    run: async (ctx) => {
      const linked = linkedOf(ctx);
      if (linked) await store().downloadOnlineOnlyFiles(linked.id);
    },
  },
  {
    id: "linkedFolder.reveal",
    label: REVEAL,
    group: "file",
    appliesTo: ["linkedFolder"],
    enabled: (ctx) =>
      linkedOf(ctx)?.status === "unavailable"
        ? { reason: "linkedFolders.reveal.unavailableReason" }
        : true,
    reversible: true,
    run: async (ctx) => {
      const linked = linkedOf(ctx);
      if (linked) await store().showLinkedFolder(linked.id);
    },
  },
  {
    id: "linkedFolder.unlink",
    label: "linkedFolders.unlinkAction",
    group: "danger",
    destructive: true,
    appliesTo: ["linkedFolder"],
    reversible: { confirm: "linkedFolders.unlink.body" },
    run(ctx) {
      const linked = linkedOf(ctx);
      if (linked) useRequests.getState().ask({ unlink: { linked, name: folderName(linked.path) } });
    },
  },
];

/** What only the Answer in the note can do: its text is in the editor. */
export const answerCommands: readonly Command[] = [
  {
    id: "answer.copy",
    label: "object.copy",
    group: "main",
    appliesTo: ["answer"],
    reversible: true,
    ...needsHere("answer.copy"),
  },
  {
    id: "answer.regenerate",
    label: "answer.regenerate",
    group: "main",
    appliesTo: ["answer"],
    reversible: true,
    ...needsHere("answer.regenerate"),
  },
];

export const citationCommands: readonly Command[] = [
  {
    id: "citation.remove",
    label: "citation.remove",
    group: "danger",
    appliesTo: ["citation"],
    // The note's own undo (⌘Z) brings it back.
    reversible: true,
    ...needsHere("citation.remove"),
  },
];
