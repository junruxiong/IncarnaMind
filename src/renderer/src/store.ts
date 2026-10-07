import { create } from "zustand";
import type {
  ChatReadiness,
  DeviceSettings,
  Document,
  EmbeddingModelStatus,
  EmbeddingSettings,
  Folder,
  LinkedFolder,
  LinkedFolderLayout,
  LinkedFolderPreview,
  Mind,
  Settings,
  SettingsPatch,
  Skill,
  Tag,
} from "../../core/api";
import type { DocumentLocation } from "../../shared/documentViewer";
import { core, files } from "./core";

type Status = { kind: "loading" } | { kind: "ready" } | { kind: "failed"; message: string };

/** The pages of the Settings dialog, in the order its list shows them. */
export const settingsPages = [
  "general",
  "chat-model",
  "search",
  "connectors",
  "skills",
  "approvals",
  "privacy",
] as const;

export type SettingsPage = (typeof settingsPages)[number];

const isSettingsPage = (page: unknown): page is SettingsPage =>
  settingsPages.some((each) => each === page);

/**
 * A folder the User picked to link, while the link dialog shows what linking
 * it would take: the preview once it is counted, or why it couldn't be.
 */
export interface LinkingFolder {
  path: string;
  preview: LinkedFolderPreview | null;
  error: string | null;
}

/** What the Document viewer shows: one Document, opened at a location. */
export interface ViewerTarget extends DocumentLocation {
  /** Counts every `openDocument` call, so opening the same Document again re-applies its location. */
  request: number;
}

interface AppState {
  status: Status;
  minds: Mind[];
  /** The Minds open as tabs in the Mind pane, by ID, in order. Kept per device across restarts. */
  tabs: string[];
  /** The Mind shown: the active tab, or null with no tab open. */
  openMindId: string | null;
  /** Set once loaded. */
  settings: Settings | null;
  /** Whether Questions can be asked. Set once loaded, then follows the core's event. */
  chatReadiness: ChatReadiness | null;
  /** The last action that failed, shown until dismissed. */
  actionError: string | null;
  /** The Document viewer panel on the right. Closed on launch; opening a Document opens it. */
  viewerOpen: boolean;
  /** The Document the viewer shows, if any. It may since have been deleted: the viewer then says so. */
  viewerTarget: ViewerTarget | null;
  /** Most recently added first. */
  documents: Document[];
  /** Names of the files the last add couldn't take, shown until dismissed. */
  skippedFiles: string[];
  /** The built-in embedding model and its download. Set once loaded, then follows the core's event. */
  embeddingModel: EmbeddingModelStatus | null;
  /** The embedding model search uses, local mode and any rebuild. Set once loaded, then follows the core's event. */
  embedding: EmbeddingSettings | null;
  settingsOpen: boolean;
  /** The page Settings shows. */
  settingsPage: SettingsPage;
  /**
   * Every Folder of every Linked folder, flat, in name order: each Linked
   * folder's own Folder and those inside it. The sidebar builds the tree from each `parentId`.
   */
  folders: Folder[];
  /** The Linked folders, in path order. Set once loaded, then follows the core's event. */
  linkedFolders: LinkedFolder[];
  /** The folder the link dialog asks about, until the User links it or cancels. */
  linking: LinkingFolder | null;
  /** Every Tag, in name order. */
  tags: Tag[];
  /** The Tag whose Documents the sidebar shows. Null: any. */
  tagFilter: string | null;
  /** The Documents with that Tag, by id, as the core last listed them. Null until listed. */
  filteredDocumentIds: ReadonlySet<string> | null;
  tagsDialogOpen: boolean;
  /** Every Skill, in name order, on or off. Set once loaded, then follows the core's event. */
  skills: Skill[];

  load(): Promise<void>;
  /** Creates a Mind and opens it in a new tab, at the end. */
  createMind(): Promise<void>;
  /**
   * Shows a Mind: its tab if it is open; otherwise it opens in the current tab
   * or, with `newTab` (⌘-click, middle-click), in a new tab after the current one.
   */
  openMind(id: string, options?: { newTab?: boolean }): void;
  /** Closes a Mind's tab; if it was shown, the tab after it (or else before it) is shown. */
  closeTab(id: string): void;
  /** Moves a tab to a place among the tabs (0 is first). */
  moveTab(id: string, index: number): void;
  renameMind(id: string, title: string): Promise<void>;
  deleteMind(id: string): Promise<void>;
  /** Shows a failure that happened outside a store action, e.g. while saving an edit. */
  reportError(error: unknown): void;
  openViewer(): void;
  closeViewer(): void;
  toggleViewer(): void;
  /**
   * Opens Settings at a page: the general one unless asked otherwise, e.g.
   * Chat model to set one up, or Privacy to allow a declined flow.
   */
  openSettings(page?: SettingsPage): void;
  closeSettings(): void;
  showSettingsPage(page: SettingsPage): void;
  /**
   * Shows a Document in the viewer, opening the panel; it replaces whatever the
   * viewer showed. Optionally at a page range, highlighting a quote (Citations).
   */
  openDocument(location: DocumentLocation): void;
  updateSettings(patch: SettingsPatch): Promise<void>;
  /** Changes pane widths on screen only, e.g. while dragging a divider. */
  previewLayout(layout: Partial<DeviceSettings>): void;
  dismissActionError(): void;
  /** Adds dropped or picked files as Documents. */
  addDocuments(picked: readonly File[]): Promise<void>;
  renameDocument(id: string, name: string): Promise<void>;
  deleteDocument(id: string): Promise<void>;
  dismissSkippedFiles(): void;
  /** Downloads the embedding model again after a failure. */
  downloadEmbeddingModel(): Promise<void>;
  /** Tries the chosen embedding provider again after an error. */
  retryEmbedding(): Promise<void>;
  /**
   * Asks for a folder with the system's folder picker, then opens the link
   * dialog on it: what linking it would take, counted from its files' metadata.
   */
  addLinkedFolder(): Promise<void>;
  /** Links the folder the link dialog asks about, shown in `layout`, and closes the dialog. */
  confirmLinkedFolder(layout: LinkedFolderLayout): Promise<void>;
  /** Closes the link dialog without linking. */
  cancelLinkedFolder(): void;
  /** Pauses or resumes a Linked folder's indexing. */
  setLinkedFolderPaused(linkedFolderId: string, paused: boolean): Promise<void>;
  /** Shows a Linked folder as Folders or as a flat list. */
  setLinkedFolderLayout(linkedFolderId: string, layout: LinkedFolderLayout): Promise<void>;
  /** Downloads a Linked folder's online-only files and indexes them. */
  downloadOnlineOnlyFiles(linkedFolderId: string): Promise<void>;
  /** Shows a Linked folder in the system's file manager. */
  showLinkedFolder(linkedFolderId: string): Promise<void>;
  /** Unlinks a folder: its Documents leave the index; nothing on disk changes. */
  removeLinkedFolder(linkedFolderId: string): Promise<void>;
  /** Shows only the Documents with a Tag; null shows them whatever their Tags. */
  filterByTag(tagId: string | null): Promise<void>;
  addDocumentTag(documentId: string, tagId: string): Promise<void>;
  removeDocumentTag(documentId: string, tagId: string): Promise<void>;
  /** Recomputes the automatic Tags of these Documents, or of all of them. */
  retagDocuments(documentIds?: string[]): Promise<void>;
  openTagsDialog(): void;
  closeTagsDialog(): void;
  setSkillEnabled(skillId: string, enabled: boolean): Promise<void>;
  removeSkill(skillId: string): Promise<void>;
  /** Copies a Skill (a built-in one) as the User's own. */
  duplicateSkill(skillId: string): Promise<void>;
  /** Installs the built-in Skills the User removed again. */
  restoreBuiltInSkills(): Promise<void>;
}

const messageOf = (error: unknown) => (error instanceof Error ? error.message : String(error));

const fileName = (path: string) => path.split(/[\\/]/).at(-1) ?? path;

/** Puts a Document in the list: in place if it's there, otherwise first. */
const upsert = (documents: Document[], item: Document) =>
  documents.some((each) => each.id === item.id)
    ? documents.map((each) => (each.id === item.id ? item : each))
    : [item, ...documents];

/** Puts a Linked folder in the list in place, if it's there. New ones arrive with the core's event. */
const replaceLinked = (linkedFolders: LinkedFolder[], linked: LinkedFolder) =>
  linkedFolders.map((each) => (each.id === linked.id ? linked : each));

/** Open tabs and the shown one, after `id`'s tab closes: the one after it is shown, or else before. */
function withoutTab(
  tabs: readonly string[],
  openMindId: string | null,
  id: string,
): { tabs: string[]; openMindId: string | null } {
  const index = tabs.indexOf(id);
  if (index === -1) return { tabs: [...tabs], openMindId };
  const rest = tabs.filter((each) => each !== id);
  if (openMindId !== id) return { tabs: rest, openMindId };
  return { tabs: rest, openMindId: rest[Math.min(index, rest.length - 1)] ?? null };
}

/** Saves the open tabs, their order and the shown one, for this device. A failure only loses that. */
function saveTabs(tabs: readonly string[], openMindId: string | null): void {
  core
    .updateSettings({ device: { openMinds: [...tabs], activeMind: openMindId } })
    .catch(() => undefined);
}

/** The Documents the sidebar lists: all of them, or those with the Tag it filters by. */
export const selectVisibleDocuments = (state: AppState): Document[] => {
  const ids = state.filteredDocumentIds;
  if (state.tagFilter === null || ids === null) return state.documents;
  return state.documents.filter((item) => ids.has(item.id));
};

export const useAppStore = create<AppState>()((set, get) => {
  /** Runs an action, reporting a failure instead of throwing. */
  const attempt = async (action: () => Promise<void>) => {
    try {
      await action();
    } catch (error) {
      set({ actionError: messageOf(error) });
    }
  };

  /** Changes the tabs, and saves them if they changed. */
  const setTabs = (next: { tabs: string[]; openMindId: string | null }) => {
    const { tabs, openMindId } = get();
    if (next.openMindId === openMindId && next.tabs.join("\n") === tabs.join("\n")) return;
    set(next);
    saveTabs(next.tabs, next.openMindId);
  };

  return {
    status: { kind: "loading" },
    minds: [],
    tabs: [],
    openMindId: null,
    settings: null,
    chatReadiness: null,
    actionError: null,
    viewerOpen: false,
    viewerTarget: null,
    documents: [],
    skippedFiles: [],
    embeddingModel: null,
    embedding: null,
    settingsOpen: false,
    settingsPage: "general",
    folders: [],
    linkedFolders: [],
    linking: null,
    tags: [],
    tagFilter: null,
    filteredDocumentIds: null,
    tagsDialogOpen: false,
    skills: [],

    async load() {
      try {
        const [
          minds,
          settings,
          documents,
          chatReadiness,
          folders,
          embeddingModel,
          tags,
          skills,
          embedding,
          linkedFolders,
        ] = await Promise.all([
          core.listMinds(),
          core.getSettings(),
          core.listDocuments(),
          core.getChatReadiness(),
          core.listFolders(),
          core.getEmbeddingModel(),
          core.listTags(),
          core.listSkills(),
          core.getEmbeddingSettings(),
          core.listLinkedFolders(),
        ]);
        // The tabs open at the last quit come back, without Minds deleted since.
        const tabs = settings.device.openMinds.filter((id) => minds.some((mind) => mind.id === id));
        const active = settings.device.activeMind;
        set({
          linkedFolders,
          minds,
          tabs,
          openMindId: active !== null && tabs.includes(active) ? active : (tabs[0] ?? null),
          settings,
          documents,
          chatReadiness,
          folders,
          embeddingModel,
          tags,
          skills,
          embedding,
          status: { kind: "ready" },
        });
      } catch (error) {
        set({ status: { kind: "failed", message: messageOf(error) } });
      }
    },

    createMind: () =>
      attempt(async () => {
        const mind = await core.createMind();
        // The "minds.changed" event may have listed it already.
        set((state) => ({
          minds: [mind, ...state.minds.filter((each) => each.id !== mind.id)],
        }));
        const { tabs } = get();
        setTabs({
          tabs: [...tabs.filter((each) => each !== mind.id), mind.id],
          openMindId: mind.id,
        });
      }),

    openMind(id, options) {
      const { tabs, openMindId } = get();
      if (tabs.includes(id)) {
        setTabs({ tabs: [...tabs], openMindId: id });
        return;
      }
      const current = openMindId === null ? -1 : tabs.indexOf(openMindId);
      const next = [...tabs];
      if (current === -1) next.push(id);
      else if (options?.newTab) next.splice(current + 1, 0, id);
      else next[current] = id;
      setTabs({ tabs: next, openMindId: id });
    },

    closeTab(id) {
      const { tabs, openMindId } = get();
      setTabs(withoutTab(tabs, openMindId, id));
    },

    moveTab(id, index) {
      const { tabs, openMindId } = get();
      if (!tabs.includes(id)) return;
      const next = tabs.filter((each) => each !== id);
      next.splice(Math.max(0, Math.min(index, next.length)), 0, id);
      setTabs({ tabs: next, openMindId });
    },

    // The list follows the core's "minds.changed" event, which arrives before these calls return.
    renameMind: (id, title) =>
      attempt(async () => {
        await core.renameMind(id, title);
      }),

    deleteMind: (id) =>
      attempt(async () => {
        await core.deleteMind(id);
      }),

    reportError(error) {
      set({ actionError: messageOf(error) });
    },

    openViewer() {
      set({ viewerOpen: true });
    },

    // Closing forgets the Document, so its PDF is released.
    closeViewer() {
      set({ viewerOpen: false, viewerTarget: null });
    },

    toggleViewer() {
      set((state) => ({
        viewerOpen: !state.viewerOpen,
        viewerTarget: state.viewerOpen ? null : state.viewerTarget,
      }));
    },

    openDocument(location) {
      const { documentId, pageFrom, pageTo, quote, citation } = location;
      set((state) => ({
        viewerOpen: true,
        viewerTarget: {
          documentId,
          pageFrom,
          pageTo,
          quote,
          citation,
          request: (state.viewerTarget?.request ?? 0) + 1,
        },
      }));
    },

    openSettings(page) {
      // Also a click handler: anything but a page name opens the general page.
      set({ settingsOpen: true, settingsPage: isSettingsPage(page) ? page : "general" });
    },

    showSettingsPage(page) {
      set({ settingsPage: page });
    },

    closeSettings() {
      set({ settingsOpen: false });
    },

    updateSettings: (patch) =>
      attempt(async () => {
        set({ settings: await core.updateSettings(patch) });
      }),

    previewLayout(layout) {
      const { settings } = get();
      if (settings) set({ settings: { ...settings, device: { ...settings.device, ...layout } } });
    },

    dismissActionError() {
      set({ actionError: null });
    },

    addDocuments: (picked) =>
      attempt(async () => {
        const paths: string[] = [];
        const notOnDisk: string[] = [];
        for (const file of picked) {
          const path = files.pathForFile(file);
          if (path) paths.push(path);
          else notOnDisk.push(file.name);
        }
        const result =
          paths.length > 0 ? await core.addDocuments(paths) : { documents: [], skipped: [] };
        set((state) => {
          // Status events may already have brought newer copies of these; keep those.
          const known = new Set(state.documents.map((each) => each.id));
          const added = new Map<string, Document>();
          for (const item of result.documents) {
            if (!known.has(item.id)) added.set(item.id, item);
          }
          return {
            documents: [...[...added.values()].reverse(), ...state.documents],
            skippedFiles: [
              ...notOnDisk,
              ...result.skipped.map((skipped) => fileName(skipped.path)),
            ],
          };
        });
      }),

    renameDocument: (id, name) =>
      attempt(async () => {
        const renamed = await core.renameDocument(id, name);
        set((state) => ({ documents: upsert(state.documents, renamed) }));
      }),

    deleteDocument: (id) =>
      attempt(async () => {
        await core.deleteDocument(id);
        set((state) => ({ documents: state.documents.filter((each) => each.id !== id) }));
      }),

    dismissSkippedFiles() {
      set({ skippedFiles: [] });
    },

    downloadEmbeddingModel: () =>
      attempt(async () => {
        set({ embeddingModel: await core.downloadEmbeddingModel() });
      }),

    retryEmbedding: () =>
      attempt(async () => {
        set({ embedding: await core.retryEmbedding() });
      }),

    addLinkedFolder: () =>
      attempt(async () => {
        const path = await files.pickLinkedFolder();
        if (!path) return;
        // The dialog opens at once; a big folder takes a moment to count.
        set({ linking: { path, preview: null, error: null } });
        try {
          const preview = await core.previewLinkedFolder(path);
          if (get().linking?.path === path) set({ linking: { path, preview, error: null } });
        } catch (error) {
          if (get().linking?.path === path) {
            set({ linking: { path, preview: null, error: messageOf(error) } });
          }
        }
      }),

    // Its Documents, Folders and progress arrive with the core's events.
    confirmLinkedFolder: (layout) =>
      attempt(async () => {
        const preview = get().linking?.preview;
        set({ linking: null });
        if (!preview) return;
        const linked = await core.addLinkedFolder(preview.path);
        // Inside a Linked folder already: nothing was linked, so its layout stays as the User set it.
        if (preview.insideLinkedFolderId !== null) return;
        // Set now, before its first scan ends, so the scan's own suggestion doesn't replace it.
        const shown = await core.setLinkedFolderLayout(linked.id, layout);
        set((state) => ({ linkedFolders: replaceLinked(state.linkedFolders, shown) }));
      }),

    cancelLinkedFolder() {
      set({ linking: null });
    },

    setLinkedFolderPaused: (linkedFolderId, paused) =>
      attempt(async () => {
        const linked = await core.setLinkedFolderPaused(linkedFolderId, paused);
        set((state) => ({ linkedFolders: replaceLinked(state.linkedFolders, linked) }));
      }),

    setLinkedFolderLayout: (linkedFolderId, layout) =>
      attempt(async () => {
        const linked = await core.setLinkedFolderLayout(linkedFolderId, layout);
        set((state) => ({ linkedFolders: replaceLinked(state.linkedFolders, linked) }));
      }),

    downloadOnlineOnlyFiles: (linkedFolderId) =>
      attempt(async () => {
        const linked = await core.downloadOnlineOnlyFiles(linkedFolderId);
        set((state) => ({ linkedFolders: replaceLinked(state.linkedFolders, linked) }));
      }),

    showLinkedFolder: (linkedFolderId) =>
      attempt(async () => {
        await files.showLinkedFolder(linkedFolderId);
      }),

    // The list, its Folders and Documents follow the core's events.
    removeLinkedFolder: (linkedFolderId) =>
      attempt(async () => {
        await core.removeLinkedFolder(linkedFolderId);
      }),

    async filterByTag(tagId) {
      if (tagId === get().tagFilter) return;
      set({ tagFilter: tagId, filteredDocumentIds: null });
      await refreshFilter();
    },

    // The Documents follow the core's "documents.tagged" event, which arrives before these calls return.
    addDocumentTag: (documentId, tagId) =>
      attempt(async () => {
        await core.addDocumentTag(documentId, tagId);
      }),

    removeDocumentTag: (documentId, tagId) =>
      attempt(async () => {
        await core.removeDocumentTag(documentId, tagId);
      }),

    retagDocuments: (documentIds) =>
      attempt(async () => {
        await core.retagDocuments(documentIds);
      }),

    openTagsDialog() {
      set({ tagsDialogOpen: true });
    },

    closeTagsDialog() {
      set({ tagsDialogOpen: false });
    },

    // The list follows the core's "skills.changed" event, which arrives before these calls return.
    setSkillEnabled: (skillId, enabled) =>
      attempt(async () => {
        await core.setSkillEnabled(skillId, enabled);
      }),

    removeSkill: (skillId) =>
      attempt(async () => {
        await core.removeSkill(skillId);
      }),

    duplicateSkill: (skillId) =>
      attempt(async () => {
        await core.duplicateSkill(skillId);
      }),

    restoreBuiltInSkills: () =>
      attempt(async () => {
        await core.restoreBuiltInSkills();
      }),
  };
});

/** Counts filter requests, so a slow answer to an old one never overwrites a newer one. */
let filterRequests = 0;

/** Asks the core which Documents have the Tag the sidebar filters by. */
async function refreshFilter(): Promise<void> {
  const request = ++filterRequests;
  const { tagFilter: tagId } = useAppStore.getState();
  if (tagId === null) {
    useAppStore.setState({ filteredDocumentIds: null });
    return;
  }
  try {
    const listed = await core.listDocuments({ tagId });
    if (request === filterRequests) {
      useAppStore.setState({ filteredDocumentIds: new Set(listed.map((item) => item.id)) });
    }
  } catch (error) {
    if (request !== filterRequests) return;
    // The Tag was deleted meanwhile: drop the filter. Anything else is a failure.
    if (useAppStore.getState().tags.some((tag) => tag.id === tagId)) {
      useAppStore.setState({ actionError: messageOf(error) });
      return;
    }
    useAppStore.setState({ tagFilter: null, filteredDocumentIds: null });
    filterRequests++; // nothing to filter by: an answer still to come is for an old filter
  }
}

/** Refreshes the filtered list, if the sidebar is filtered. */
function refreshFilterIfAny(): void {
  if (useAppStore.getState().tagFilter !== null) void refreshFilter();
  else filterRequests++; // nothing to filter by: an answer still to come is for an old filter
}

// Settings can change outside this window (another window, or the core itself), so follow the core's event.
core.on("settings.changed", (settings) => useAppStore.setState({ settings }));

// The same for the list of Minds: created, renamed, deleted, or reordered by an edit.
// A deleted Mind's tab closes, as if closed by hand.
core.on("minds.changed", (minds) => {
  const { tabs, openMindId, status } = useAppStore.getState();
  useAppStore.setState({ minds });
  // Before loading, the tabs aren't known yet: loading filters them.
  if (status.kind !== "ready") return;
  let next = { tabs, openMindId };
  for (const id of tabs) {
    if (!minds.some((mind) => mind.id === id)) next = withoutTab(next.tabs, next.openMindId, id);
  }
  if (next.tabs.length !== tabs.length) {
    useAppStore.setState(next);
    saveTabs(next.tabs, next.openMindId);
  }
});

// Processing happens in the background: follow each Document's status as the core reports it.
core.on("document.status", (changed) =>
  useAppStore.setState((state) => ({ documents: upsert(state.documents, changed) })),
);

core.on("chatReadiness.changed", (chatReadiness) => useAppStore.setState({ chatReadiness }));

// The embedding model downloads in the background: follow its state and progress.
core.on("embeddingModel.status", (embeddingModel) => useAppStore.setState({ embeddingModel }));

// The embedding model can change in Settings or by local mode, and a rebuild reports its progress.
core.on("embedding.changed", (embedding) => useAppStore.setState({ embedding }));

// Folders change through this window or another: follow the list. The sidebar's tree follows it.
core.on("folders.changed", (folders) => useAppStore.setState({ folders }));

// Linked folders added, removed, or scanning, paused, out of reach, or making progress.
core.on("linkedFolders.changed", (linkedFolders) => useAppStore.setState({ linkedFolders }));

// Documents removed from the index, e.g. with their Linked folder.
core.on("documents.removed", (removed) => {
  const gone = new Set(removed);
  useAppStore.setState((state) => ({
    documents: state.documents.filter((each) => !gone.has(each.id)),
  }));
});

// Documents whose files moved to another Folder on disk.
core.on("documents.moved", (moved) => {
  useAppStore.setState((state) => ({
    documents: moved.reduce((documents, item) => upsert(documents, item), state.documents),
  }));
});

// Tags change through this window or another: follow the list, and drop a filter by a deleted Tag.
core.on("tags.changed", (tags) => {
  const { tagFilter } = useAppStore.getState();
  useAppStore.setState({ tags });
  if (tagFilter !== null && !tags.some((tag) => tag.id === tagFilter)) {
    useAppStore.setState({ tagFilter: null, filteredDocumentIds: null });
    refreshFilterIfAny();
  }
});

// Skills are imported, turned on or off, or removed, through this window or another.
core.on("skills.changed", (skills) => useAppStore.setState({ skills }));

// Documents' Tags, or their tagging, changed: by the User, by automatic tagging, or with a deleted Tag.
core.on("documents.tagged", (tagged) => {
  useAppStore.setState((state) => ({
    documents: tagged.reduce((documents, item) => upsert(documents, item), state.documents),
  }));
  if (useAppStore.getState().tagFilter !== null) void refreshFilter();
});
