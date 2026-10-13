import type { LinkedFolder } from "../../../core/api";
import { displayName } from "../../../shared/documentNames";
import { core, files } from "../core";
import { showFileLabelKey } from "../fileInfo";
import { languageNow, translateNow } from "../i18n";
import { bridgeFrom, type LibraryBridge } from "../libraryBridges";
import { fileStatusLabel, folderName, formatCount } from "../linkedFolders";
import { useRequests } from "../requests";
import { useAppStore } from "../store";
import { findTab } from "../tabs";
import { deleteTag } from "../tagChanges";
import { type Command, type CommandContext, needsHere } from "./registry";
import { documentsOf, everyDocument, idsOf, onlyOne } from "./targets";

const store = () => useAppStore.getState();
const reportError = (error: unknown) => store().reportError(error);
const RENAME_KEYS = "F2";

/** The Mind a Mind's row, or the tab showing it, stands for. */
const mindOf = (ctx: CommandContext) => {
  const target = ctx.selection[0] ?? ctx.focus;
  const shown = target?.kind === "tab" ? findTab(store().layout, target.id)?.item : undefined;
  const id =
    shown?.kind === "mind" ? shown.mindId : target?.kind === "mind" ? target.id : undefined;
  return store().minds.find((each) => each.id === id);
};
const ofMind = (ctx: CommandContext) => mindOf(ctx) !== undefined;
const folderOf = (ctx: CommandContext) => {
  const id = idsOf(ctx, "folder")[0];
  return store().library?.groups.find((each) => each.id === id);
};
const linkedOf = (ctx: CommandContext): LinkedFolder | undefined => {
  const id = idsOf(ctx, "linkedFolder")[0];
  return store().linkedFolders.find((each) => each.id === id);
};

/** How many Documents a selection holds, or 0 for one: its commands then say "these 3". */
const many = (ctx: CommandContext): number => {
  const count = documentsOf(ctx).length;
  return count > 1 ? count : 0;
};

function bridgeOf(ctx: CommandContext): LibraryBridge | null {
  const documents = documentsOf(ctx);
  const [first] = documents;
  if (!first) return null;
  return {
    kind: "documents",
    title:
      documents.length === 1
        ? displayName(first)
        : translateNow("library.count", { count: documents.length }),
    count: documents.length,
    scope: { folderIds: [], tagIds: [], documentIds: documents.map((each) => each.id) },
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
    when: onlyOne,
    reversible: true,
    run: (ctx) => store().openMind(idsOf(ctx, "mind")[0] ?? ""),
  },
  {
    id: "mind.openSide",
    label: "mind.openSide",
    group: "open",
    appliesTo: ["mind"],
    when: onlyOne,
    reversible: true,
    run: (ctx) => store().openMind(idsOf(ctx, "mind")[0] ?? "", { side: true }),
  },
  // Exporting (Export as…, Export options…) is in ./exports.
  {
    id: "mind.moveTo",
    label: "mind.moveTo",
    rowKeys: "CmdOrCtrl+Shift+M",
    group: "move",
    appliesTo: ["mind", "tab"],
    when: (ctx) => ofMind(ctx) && ctx.here?.["mind.moveTo"] !== undefined,
    reversible: true,
    run: (ctx) => ctx.here?.["mind.moveTo"]?.(),
  },
  {
    id: "mind.rename",
    label: "mind.rename",
    rowKeys: RENAME_KEYS,
    group: "edit",
    appliesTo: ["mind"],
    reversible: true,
    ...needsHere("mind.rename"),
    when: (ctx) => onlyOne(ctx) && ctx.here?.["mind.rename"] !== undefined,
  },
  {
    id: "mind.delete",
    label: "mind.deleteAction",
    group: "danger",
    destructive: true,
    appliesTo: ["mind", "tab"],
    when: (ctx) => onlyOne(ctx) && ofMind(ctx),
    reversible: true,
    async run(ctx) {
      const mind = mindOf(ctx);
      if (mind) await store().deleteMind(mind.id);
    },
  },
];

export const documentCommands: readonly Command[] = [
  {
    id: "document.open",
    label: "mind.open",
    group: "open",
    appliesTo: ["document"],
    when: onlyOne,
    reversible: true,
    run: (ctx) => store().openDocument({ documentId: idsOf(ctx, "document")[0] ?? "" }),
  },
  {
    id: "document.openTab",
    label: "document.openTab",
    group: "open",
    appliesTo: ["document"],
    when: onlyOne,
    reversible: true,
    run: (ctx) =>
      store().openDocument({ documentId: idsOf(ctx, "document")[0] ?? "" }, { as: "here" }),
  },
  {
    id: "document.ask",
    label: (ctx, t) =>
      many(ctx) ? t("library.ask.documents", { count: many(ctx) }) : t("documents.ask"),
    group: "main",
    appliesTo: ["document"],
    when: (ctx) => documentsOf(ctx).length >= 1,
    reversible: true,
    run(ctx) {
      const bridge = bridgeOf(ctx);
      if (bridge) return store().askAbout(bridge);
    },
  },
  {
    id: "document.startMind",
    label: (ctx, t) =>
      many(ctx) ? t("documents.startMind.many", { count: many(ctx) }) : t("documents.startMind"),
    group: "main",
    appliesTo: ["document"],
    when: (ctx) => documentsOf(ctx).length >= 1,
    reversible: true,
    run(ctx) {
      const bridge = bridgeOf(ctx);
      if (bridge) return store().startMindFrom(bridge);
    },
  },
  {
    id: "document.moveTo",
    label: "documents.moveTo",
    rowKeys: "CmdOrCtrl+Shift+M",
    group: "move",
    appliesTo: ["document"],
    reversible: true,
    ...needsHere("document.moveTo"),
  },
  {
    id: "document.retry",
    label: (ctx, t) =>
      many(ctx)
        ? t("documents.retryAction.many", { count: many(ctx) })
        : t("documents.retryAction"),
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
    id: "document.tags",
    label: "document.tags",
    group: "tags",
    appliesTo: ["document"],
    reversible: true,
    ...needsHere("document.tags"),
  },
  {
    id: "document.rename",
    label: "documents.renameAction",
    rowKeys: RENAME_KEYS,
    group: "edit",
    appliesTo: ["document"],
    reversible: true,
    ...needsHere("document.rename"),
    when: (ctx) => onlyOne(ctx) && ctx.here?.["document.rename"] !== undefined,
  },
  {
    id: "document.openExternally",
    label: "documents.copy.open",
    group: "file",
    appliesTo: ["document"],
    when: (ctx) => onlyOne(ctx) && onFiles(ctx),
    // Its file isn't where it was: the item says why.
    enabled: fileReason,
    reversible: true,
    run(ctx) {
      for (const id of idsOf(ctx, "document")) files.openDocumentExternally(id).catch(reportError);
    },
  },
  {
    id: "document.showInFolder",
    label: (_ctx, t) => t(showFileLabelKey(navigator.platform)),
    group: "file",
    appliesTo: ["document"],
    when: (ctx) => onlyOne(ctx) && onFiles(ctx),
    enabled: fileReason,
    reversible: true,
    run(ctx) {
      for (const id of idsOf(ctx, "document")) files.showDocumentInFolder(id).catch(reportError);
    },
  },
  {
    id: "document.delete",
    label: (ctx, t) =>
      many(ctx)
        ? t("documents.deleteAction.many", { count: many(ctx) })
        : t(
            documentsOf(ctx).every((document) => document.fileStatus === "missing")
              ? "documents.removeAction"
              : "documents.deleteAction",
          ),
    group: "danger",
    destructive: true,
    appliesTo: ["document"],
    reversible: true,
    async run(ctx) {
      await store().deleteDocuments(documentsOf(ctx).map((document) => document.id));
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
    id: "folder.ask",
    label: "library.ask.folder",
    group: "main",
    appliesTo: ["folder"],
    when: (ctx) => folderBridge(ctx) !== null,
    reversible: true,
    run(ctx) {
      const bridge = folderBridge(ctx);
      if (bridge) return store().askAbout(bridge);
    },
  },
  {
    id: "folder.organize",
    label: "folder.organize",
    group: "main",
    appliesTo: ["folder"],
    enabled: (ctx) =>
      folderDocumentIds(ctx).length > 0 ? true : { reason: "folder.organize.empty" },
    // Suggestions are reviewed and taken back from the Library.
    reversible: true,
    async run(ctx) {
      if (!store().library?.settings.classifier) return store().openSettings("models");
      await core.classifyDocuments(folderDocumentIds(ctx));
    },
  },
  {
    id: "folder.editDescription",
    label: "folder.editDescription",
    group: "main",
    appliesTo: ["folder"],
    reversible: true,
    run(ctx) {
      const id = idsOf(ctx, "folder")[0];
      if (!id) return;
      store().openLibrary(id);
      useRequests.getState().ask({ editFolder: id });
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
    reversible: true,
    async run(ctx) {
      const folder = folderOf(ctx);
      if (folder) await store().deleteFolder(folder.id);
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
    id: "tag.rename",
    label: "tag.rename",
    rowKeys: RENAME_KEYS,
    group: "edit",
    appliesTo: ["tag"],
    reversible: true,
    run(ctx) {
      // In its row where it has a field; elsewhere, in the Tags dialog.
      if (ctx.here?.["tag.rename"]) return ctx.here["tag.rename"]();
      useRequests.getState().ask({ manageTag: { id: idsOf(ctx, "tag")[0] ?? "" } });
      store().openTagsDialog();
    },
  },
  {
    id: "tag.delete",
    label: "tag.delete",
    group: "danger",
    destructive: true,
    appliesTo: ["tag"],
    reversible: true,
    run: (ctx) => deleteTag(idsOf(ctx, "tag")[0] ?? "").catch(reportError),
  },
  {
    id: "tag.manage",
    label: "tags.menu.manage",
    group: "edit",
    appliesTo: ["tag"],
    reversible: true,
    run: () => store().openTagsDialog(),
  },
];

const folderDocumentIds = (ctx: CommandContext): string[] => {
  const id = idsOf(ctx, "folder")[0];
  return (store().library?.assignments ?? []).flatMap((item) =>
    item.groupId === id ? [item.documentId] : [],
  );
};

const folderBridge = (ctx: CommandContext): LibraryBridge | null => {
  const folder = folderOf(ctx);
  return folder
    ? bridgeFrom(
        { kind: "folder", id: folder.id, name: folder.name },
        folderDocumentIds(ctx),
        false,
      )
    : null;
};

const REVEAL = showFileLabelKey(navigator.platform);

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
    id: "linkedFolder.locate",
    label: "linkedFolders.locateAction",
    group: "file",
    appliesTo: ["linkedFolder"],
    // Locating it again, where it was, puts it back.
    reversible: true,
    run: async (ctx) => {
      const linked = linkedOf(ctx);
      if (linked) await store().locateLinkedFolder(linked.id);
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
