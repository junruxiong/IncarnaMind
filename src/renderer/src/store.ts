import { create } from "zustand";
import type {
  ChatReadiness,
  DeviceSettings,
  Document,
  EmbeddingModelStatus,
  Folder,
  Mind,
  Settings,
  SettingsPatch,
  Skill,
  Tag,
} from "../../core/api";
import type { DocumentLocation } from "../../shared/documentViewer";
import { core, files } from "./core";

type Status = { kind: "loading" } | { kind: "ready" } | { kind: "failed"; message: string };

/** What the Document viewer shows: one Document, opened at a location. */
export interface ViewerTarget extends DocumentLocation {
  /** Counts every `openDocument` call, so opening the same Document again re-applies its location. */
  request: number;
}

interface AppState {
  status: Status;
  minds: Mind[];
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
  settingsOpen: boolean;
  /** Every Folder, flat, in name order. The sidebar builds the tree from each `parentId`. */
  folders: Folder[];
  /** The Folder whose Documents the sidebar shows, sub-Folders included. Null shows every Document. */
  folderFilter: string | null;
  /** Every Tag, in name order. */
  tags: Tag[];
  /** The Tag whose Documents the sidebar shows, within `folderFilter` if that is set too. Null: any. */
  tagFilter: string | null;
  /** The Documents matching both filters, by id, as the core last listed them. Null until listed. */
  filteredDocumentIds: ReadonlySet<string> | null;
  tagsDialogOpen: boolean;
  /** Every Skill, in name order, on or off. Set once loaded, then follows the core's event. */
  skills: Skill[];

  load(): Promise<void>;
  createMind(): Promise<void>;
  openMind(id: string): void;
  renameMind(id: string, title: string): Promise<void>;
  deleteMind(id: string): Promise<void>;
  /** Shows a failure that happened outside a store action, e.g. while saving an edit. */
  reportError(error: unknown): void;
  openViewer(): void;
  closeViewer(): void;
  toggleViewer(): void;
  openSettings(): void;
  closeSettings(): void;
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
  /** Files a Document in a Folder, or unfiles it with null. */
  moveDocument(documentId: string, folderId: string | null): Promise<void>;
  /** Shows only the Documents in a Folder and its sub-Folders; null shows them all. */
  filterByFolder(folderId: string | null): Promise<void>;
  createFolder(name: string, parentId: string | null): Promise<void>;
  renameFolder(id: string, name: string): Promise<void>;
  moveFolder(id: string, parentId: string | null): Promise<void>;
  deleteFolder(id: string): Promise<void>;
  /** Shows only the Documents with a Tag (in the filtered Folder, if any); null shows them whatever their Tags. */
  filterByTag(tagId: string | null): Promise<void>;
  addDocumentTag(documentId: string, tagId: string): Promise<void>;
  removeDocumentTag(documentId: string, tagId: string): Promise<void>;
  /** Recomputes the automatic Tags of these Documents, or of all of them. */
  retagDocuments(documentIds?: string[]): Promise<void>;
  openTagsDialog(): void;
  closeTagsDialog(): void;
  setSkillEnabled(skillId: string, enabled: boolean): Promise<void>;
  removeSkill(skillId: string): Promise<void>;
}

const messageOf = (error: unknown) => (error instanceof Error ? error.message : String(error));

const fileName = (path: string) => path.split(/[\\/]/).at(-1) ?? path;

/** Puts a Document in the list: in place if it's there, otherwise first. */
const upsert = (documents: Document[], item: Document) =>
  documents.some((each) => each.id === item.id)
    ? documents.map((each) => (each.id === item.id ? item : each))
    : [item, ...documents];

/** The Documents the sidebar lists: all of them, or those matching its Folder and Tag filters. */
export const selectVisibleDocuments = (state: AppState): Document[] => {
  const ids = state.filteredDocumentIds;
  if ((state.folderFilter === null && state.tagFilter === null) || ids === null) {
    return state.documents;
  }
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

  return {
    status: { kind: "loading" },
    minds: [],
    openMindId: null,
    settings: null,
    chatReadiness: null,
    actionError: null,
    viewerOpen: false,
    viewerTarget: null,
    documents: [],
    skippedFiles: [],
    embeddingModel: null,
    settingsOpen: false,
    folders: [],
    folderFilter: null,
    tags: [],
    tagFilter: null,
    filteredDocumentIds: null,
    tagsDialogOpen: false,
    skills: [],

    async load() {
      try {
        const [minds, settings, documents, chatReadiness, folders, embeddingModel, tags, skills] =
          await Promise.all([
            core.listMinds(),
            core.getSettings(),
            core.listDocuments(),
            core.getChatReadiness(),
            core.listFolders(),
            core.getEmbeddingModel(),
            core.listTags(),
            core.listSkills(),
          ]);
        set({
          minds,
          settings,
          documents,
          chatReadiness,
          folders,
          embeddingModel,
          tags,
          skills,
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
          openMindId: mind.id,
        }));
      }),

    openMind(id) {
      set({ openMindId: id });
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
      const { documentId, pageFrom, pageTo, quote } = location;
      set((state) => ({
        viewerOpen: true,
        viewerTarget: {
          documentId,
          pageFrom,
          pageTo,
          quote,
          request: (state.viewerTarget?.request ?? 0) + 1,
        },
      }));
    },

    openSettings() {
      set({ settingsOpen: true });
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

    // The lists follow the core's "documents.moved" and "folders.changed" events, which
    // arrive before these calls return.
    moveDocument: (documentId, folderId) =>
      attempt(async () => {
        await core.moveDocument(documentId, folderId);
      }),

    async filterByFolder(folderId) {
      if (folderId === get().folderFilter) return;
      set({ folderFilter: folderId, filteredDocumentIds: null });
      await refreshFilter();
    },

    createFolder: (name, parentId) =>
      attempt(async () => {
        await core.createFolder({ name, parentId });
      }),

    renameFolder: (id, name) =>
      attempt(async () => {
        await core.renameFolder(id, name);
      }),

    moveFolder: (id, parentId) =>
      attempt(async () => {
        await core.moveFolder(id, parentId);
      }),

    deleteFolder: (id) =>
      attempt(async () => {
        await core.deleteFolder(id);
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
  };
});

/** Counts filter requests, so a slow answer to an old one never overwrites a newer one. */
let filterRequests = 0;

/** Asks the core which Documents match the filters: in the Folder (sub-Folders included), with the Tag. */
async function refreshFilter(): Promise<void> {
  const request = ++filterRequests;
  const { folderFilter: folderId, tagFilter: tagId } = useAppStore.getState();
  if (folderId === null && tagId === null) {
    useAppStore.setState({ filteredDocumentIds: null });
    return;
  }
  try {
    const listed = await core.listDocuments({
      ...(folderId !== null && { folderId, includeSubfolders: true }),
      ...(tagId !== null && { tagId }),
    });
    if (request === filterRequests) {
      useAppStore.setState({ filteredDocumentIds: new Set(listed.map((item) => item.id)) });
    }
  } catch (error) {
    if (request !== filterRequests) return;
    const { folders, tags } = useAppStore.getState();
    const folderGone = folderId !== null && !folders.some((folder) => folder.id === folderId);
    const tagGone = tagId !== null && !tags.some((tag) => tag.id === tagId);
    // The Folder or Tag was deleted meanwhile: drop that filter. Anything else is a failure.
    if (!folderGone && !tagGone) {
      useAppStore.setState({ actionError: messageOf(error) });
      return;
    }
    useAppStore.setState({
      folderFilter: folderGone ? null : folderId,
      tagFilter: tagGone ? null : tagId,
      filteredDocumentIds: null,
    });
    void refreshFilter();
  }
}

/** Refreshes the filtered list, if the sidebar is filtered. */
function refreshFilterIfAny(): void {
  const { folderFilter, tagFilter } = useAppStore.getState();
  if (folderFilter !== null || tagFilter !== null) void refreshFilter();
  else filterRequests++; // nothing to filter by: an answer still to come is for an old filter
}

// Settings can change outside this window (another window, or the core itself), so follow the core's event.
core.on("settings.changed", (settings) => useAppStore.setState({ settings }));

// The same for the list of Minds: created, renamed, deleted, or reordered by an edit.
core.on("minds.changed", (minds) =>
  useAppStore.setState((state) => ({
    minds,
    openMindId: minds.some((mind) => mind.id === state.openMindId) ? state.openMindId : null,
  })),
);

// Processing happens in the background: follow each Document's status as the core reports it.
core.on("document.status", (changed) =>
  useAppStore.setState((state) => ({ documents: upsert(state.documents, changed) })),
);

core.on("chatReadiness.changed", (chatReadiness) => useAppStore.setState({ chatReadiness }));

// The embedding model downloads in the background: follow its state and progress.
core.on("embeddingModel.status", (embeddingModel) => useAppStore.setState({ embeddingModel }));

// Folders change through this window or another: follow the list, and keep the filter right.
core.on("folders.changed", (folders) => {
  const { folderFilter } = useAppStore.getState();
  useAppStore.setState({ folders });
  if (folderFilter !== null && !folders.some((folder) => folder.id === folderFilter)) {
    // The filtered Folder was deleted, perhaps with a parent: drop that filter.
    useAppStore.setState({ folderFilter: null, filteredDocumentIds: null });
  }
  // Moving a Folder can change which Documents are below the filtered one.
  refreshFilterIfAny();
});

// Documents moved between Folders, or unfiled by a Folder's deletion.
core.on("documents.moved", (moved) => {
  useAppStore.setState((state) => ({
    documents: moved.reduce((documents, item) => upsert(documents, item), state.documents),
  }));
  refreshFilterIfAny();
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
