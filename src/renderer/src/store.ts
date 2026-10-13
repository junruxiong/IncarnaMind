import { create } from "zustand";
import type {
  ChatReadiness,
  DeviceSettings,
  Document,
  EmbeddingModelStatus,
  EmbeddingSettings,
  Examples,
  Folder,
  FolderMove,
  GettingStarted,
  KeptCitationText,
  LayoutChange,
  LayoutTab,
  LibraryGroup,
  LibraryGroupInput,
  LibrarySnapshot,
  LinkedFolder,
  LinkedFolderLayout,
  LinkedFolderPreview,
  Mind,
  PrivacySettings,
  RerankSettings,
  SearchScope,
  Settings,
  SettingsPatch,
  Skill,
  Tag,
} from "../../core/api";
import { resolveLanguage } from "../../core/language";
import type { UiUsageEvent } from "../../core/usageEvents";
import { displayName } from "../../shared/documentNames";
import type { DocumentLocation } from "../../shared/documentViewer";
import { type MessageKey, type MessageParams, translate } from "../../shared/i18n";
import { mindName } from "../../shared/naming";
import { deleted, renamed, retagged, tagLinks } from "./changes";
import { core, files } from "./core";
import { LIBRARY_DELETED } from "./deletedItems";
import { applyAssignments, applyDocumentChanges, createDocumentUpdates } from "./documentUpdates";
import { errorMessage } from "./errors";
import {
  createLayoutSlice,
  currentMindIdOf,
  followLayout,
  groupsMerged,
  type LayoutSlice,
  WINDOW_ID,
} from "./layoutSlice";
import { type LibraryBridge, mindToAskIn } from "./libraryBridges";
import { matchesTags, NEEDS_REVIEW } from "./libraryFilters";
import {
  foldersMovedFrom,
  itemsOf,
  type MoveItems,
  madeMessage,
  movedLocally,
  moveMessage,
  moveName,
  undoMoves,
} from "./moves";
import { untitledFolderName } from "./newFolder";
import { useNotices } from "./notices";
import { useRequests } from "./requests";
import { resolveSettingsPage, type SettingsPage } from "./settingsPages";
import { type Undoable, UndoRefused, useUndo } from "./undo";

type Status = { kind: "loading" } | { kind: "ready" } | { kind: "failed"; message: string };

/**
 * A folder the User picked to link, while the link dialog shows what linking
 * it would take: the preview once it is counted, or why it couldn't be.
 */
export interface LinkingFolder {
  path: string;
  preview: LinkedFolderPreview | null;
  error: string | null;
}

/** What a Document tab shows: its Document, opened at a location. */
export interface ViewerTarget extends DocumentLocation {
  /** Counts every opening of the tab at a place, so opening it there again re-applies it. */
  request: number;
}

/**
 * How a Document opens:
 * - "citation": from a Citation or link in a group (`from`), as a preview in the
 *   group to its right, the focus staying where it was (unless the groups are merged);
 * - "list": from the sidebar, ⌘K or a test, as a preview beside the Mind;
 * - "library": from a Library row in group `from`, as a preview to its right;
 * - "here": a normal tab in group `from` (⌘-click in the Library).
 *
 * A Document already open goes to its tab, in any group. `keep`, for
 * "citation": a preview there is kept (Open the page, in a Citation's card).
 */
export interface OpenDocumentHow {
  as: "citation" | "list" | "library" | "here";
  from?: string;
  keep?: boolean;
}

interface AppState extends LayoutSlice {
  status: Status;
  minds: Mind[];
  /**
   * A Mind just created, whose title takes the focus once it shows (see
   * `MindTitle`), so typing names it instead of pressing the button again.
   */
  titleToFocus: string | null;
  /** Set once loaded. */
  settings: Settings | null;
  /** Whether Questions can be asked. Set once loaded, then follows the core's event. */
  chatReadiness: ChatReadiness | null;
  /** Most recently added first. */
  documents: Document[];
  /**
   * The text kept of Documents unlinked with their Linked folder, so the
   * Citations that quote it can still be checked. Set once loaded, then follows the core's event.
   */
  keptCitationTexts: KeptCitationText[];
  /** Names of the files the last add couldn't take, shown until dismissed. */
  skippedFiles: string[];
  /** Names of the files the last add found in IncarnaMind already, shown until dismissed. */
  alreadyAdded: string[];
  /** The built-in embedding model and its download. Set once loaded, then follows the core's event. */
  embeddingModel: EmbeddingModelStatus | null;
  /** The embedding model search uses, local mode and any rebuild. Set once loaded, then follows the core's event. */
  embedding: EmbeddingSettings | null;
  /** Reranking, and the built-in reranking model's download. Set once loaded, then follows the core's event. */
  rerank: RerankSettings | null;
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
  /** More folders dropped at once, which the link dialog asks about in turn. */
  linkQueue: LinkingFolder[];
  /** Every Tag, in name order. */
  tags: Tag[];
  /**
   * The Tags whose Documents the sidebar and the Library show: those with any
   * of them (`NEEDS_REVIEW` stands for a Tag awaiting review). Empty: all.
   */
  tagFilter: readonly string[];
  tagsDialogOpen: boolean;
  library: LibrarySnapshot | null;
  /**
   * What the Library was asked to show: "all"; a Folder's id, "unsorted" or
   * `LIBRARY_DELETED`, whose section the page brings into view (then "all"
   * again); or "starters", the starter Folders at its top.
   */
  libraryFilter: string;
  refreshLibrary(): Promise<void>;
  /**
   * Shows the Library (one tab per window), at a Folder's section if given:
   * its tab where it is open, else a new tab in the focused group, or to the side.
   */
  openLibrary(filter?: string, options?: { side?: boolean }): void;
  /** Every Skill, in name order, on or off. Set once loaded, then follows the core's event. */
  skills: Skill[];
  /** The example Mind and its Documents (onboarding). Null until loaded. */
  examples: Examples | null;
  /**
   * The privacy choices on this device: the first run asks about usage data
   * from them. Set once loaded, then follows the core's event.
   */
  privacy: PrivacySettings | null;
  /**
   * The Mind whose editor should start a Question at its end once it shows,
   * e.g. a new one, with the Search scope it starts with (from the Library), if any.
   */
  questionToStart: { mindId: string; scope: SearchScope | null } | null;

  load(): Promise<void>;
  /**
   * Creates a Mind, in a Folder if given, and opens it in a new tab after the
   * shown one, in the focused group or `groupId`, with its title focused, or
   * with a Question started in it (with a Search scope, if given).
   */
  createMind(options?: {
    startQuestion?: boolean;
    title?: string;
    scope?: SearchScope | null;
    folderId?: string | null;
    groupId?: string;
  }): Promise<void>;
  /**
   * Moves Minds and Documents into a Folder, or out of every Folder (null):
   * shown at once, then as the core has it, with "Moved … · Undo" (and ⌘Z).
   * The sidebar's drag and drop and "Move to…" share it.
   */
  moveToFolder(items: MoveItems, folderId: string | null): Promise<void>;
  /**
   * Makes "Untitled Folder" (numbered if that is taken) and asks the sidebar to
   * open its name field. With `items`, the Minds and Documents go into it at
   * once. One step to undo, with its first name (`nameNewFolder`). Resolves to
   * the Folder's id, or null if it couldn't be made.
   */
  createFolder(items?: MoveItems): Promise<string | null>;
  /** Names the Folder just made: part of making it, so one Undo takes both back. */
  nameNewFolder(folderId: string, name: string): Promise<void>;
  /** Renames a Folder, keeping its description. */
  renameFolder(folderId: string, name: string): Promise<void>;
  /** Changes a Folder's name, description or evidence choice, with Undo. Throws what the core throws. */
  editFolder(folderId: string, input: LibraryGroupInput): Promise<void>;
  /** Deletes a Folder, with Undo: its Documents and Minds go to Not in a Folder. */
  deleteFolder(folderId: string): Promise<void>;
  /** Lists a Mind just made and opens it in a new tab after the shown one, in the focused group or `groupId`. */
  showNewMind(mind: Mind, groupId?: string): Promise<void>;
  /** Called once the new Mind's title has the focus. */
  titleFocused(): void;
  /** Starts a Question at the end of the open Mind, or of a new Mind if none is open. */
  startQuestion(): void;
  /** Called once the Question asked for is started. */
  questionStarted(): void;
  /**
   * "Ask about this Folder" in the Library: a Question with the bridge's
   * Search scope at the end of the most recent Mind (see `mindToAskIn`), or
   * of a new Mind if there is none, with the cursor in it.
   */
  askAbout(bridge: LibraryBridge): Promise<void>;
  /**
   * "Start a Mind from this Folder": a new Mind titled as the Library's
   * sheet, whose first Block is a Question with the bridge's Search scope.
   */
  startMindFrom(bridge: LibraryBridge): Promise<void>;
  /** Opens the example Mind, making the examples again if they were removed. */
  openExamples(): Promise<void>;
  /** Deletes the example Mind and its Documents. */
  removeExamples(): Promise<void>;
  /**
   * Ticks steps of the "Get started" checklist, or hides it. Only once it
   * started (a first run), and only what changes.
   */
  updateGettingStarted(patch: Partial<GettingStarted>): void;
  /**
   * Shows a Mind: its tab where it is open, else a new tab in the focused
   * group after the shown one; with `side` (⌘-click), in the group to the right.
   */
  openMind(id: string, options?: { side?: boolean }): void;
  /** Closes a tab, offering to reopen it; if it was shown, the tab after it (or else before it) is. */
  closeTab(tabId: string): void;
  /** ⇧⌘T: reopens the tab closed last, where it was. */
  reopenTab(): void;
  /** Renames a Mind, with Undo; `typing`: as the User types its title, which becomes one step once they leave it (`titleLeft`). */
  renameMind(id: string, title: string, options?: { typing?: boolean }): Promise<void>;
  /** The User left a Mind's title: what they typed, from `before` to `after`, becomes one step to undo. */
  titleLeft(id: string, before: string, after: string): void;
  deleteMind(id: string): Promise<void>;
  /** Shows a failure that happened outside a store action, e.g. while saving an edit. */
  reportError(error: unknown): void;
  /**
   * Opens Settings at a page: the general one unless asked otherwise, e.g.
   * Models to set one up, or Privacy to allow a declined flow.
   */
  openSettings(page?: SettingsPage): void;
  closeSettings(): void;
  showSettingsPage(page: SettingsPage): void;
  /**
   * Shows a Document in a Document tab (see `OpenDocumentHow`), optionally at a
   * page range, highlighting a quote (Citations).
   */
  openDocument(location: DocumentLocation, how?: OpenDocumentHow): void;
  updateSettings(patch: SettingsPatch): Promise<void>;
  /** Changes pane widths on screen only, e.g. while dragging a divider. */
  previewLayout(layout: Partial<DeviceSettings>): void;
  /**
   * Adds picked files as Documents. Files already in IncarnaMind, and those
   * it can't take, are named in the footer, once.
   */
  addDocuments(picked: readonly File[]): Promise<void>;
  /**
   * What dropping on the window does: files are added as Documents, and each
   * folder opens the link dialog (see `addLinkedFolder`), one after another.
   * `folders` are the dropped items the drop said are folders.
   */
  addDropped(dropped: readonly File[], folders: readonly File[]): Promise<void>;
  /**
   * The same, given absolute paths: a path that turns out to be a folder is
   * offered for linking. For the smoke tests, which can't drop a folder.
   */
  addPaths(paths: readonly string[]): Promise<void>;
  /** "Add Documents": the system's open dialog (see `FilesBridge.pickDocuments`), then the files picked are added. */
  pickDocuments(): Promise<void>;
  /** While that dialog is open: the buttons that open it wait, so it isn't opened twice. */
  pickingDocuments: boolean;
  /**
   * "Locate file…" on a Document whose file moved: the system's open dialog,
   * then the file picked is added, which points the Document at it if it has
   * the Document's content. Says what happened: "cancelled", "located", or
   * "other" when the file was a different one (and became a Document of its own).
   */
  locateDocumentFile(id: string): Promise<"cancelled" | "located" | "other">;
  renameDocument(id: string, name: string): Promise<void>;
  /** To Recently deleted, at once, with one undo line for all of them. */
  deleteDocuments(ids: readonly string[]): Promise<void>;
  /** Processes a Document that failed again, from its file. */
  retryDocument(id: string): Promise<void>;
  dismissSkippedFiles(): void;
  dismissAlreadyAdded(): void;
  /** Downloads the embedding model again after a failure. */
  downloadEmbeddingModel(): Promise<void>;
  /** Downloads the built-in reranking model again after a failure. */
  downloadRerankingModel(): Promise<void>;
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
  /**
   * "Locate folder…": the system's folder picker, then the Linked folder is
   * looked for there, its Documents keeping their ids and Citations. Says why
   * if the folder picked isn't it.
   */
  locateLinkedFolder(linkedFolderId: string): Promise<void>;
  /** Unlinks a folder: its Documents leave the index; nothing on disk changes. */
  removeLinkedFolder(linkedFolderId: string): Promise<void>;
  /** Adds a Tag to the filter, or takes it out if it is in it. */
  toggleTagFilter(tagId: string): void;
  /** Shows the Documents with any of these Tags; none shows them whatever their Tags. */
  setTagFilter(tagIds: readonly string[]): void;
  addDocumentTag(documentId: string, tagId: string): Promise<void>;
  removeDocumentTag(documentId: string, tagId: string): Promise<void>;
  /** Puts a Tag on several Documents at once, or takes it off them. */
  addTagToDocuments(documentIds: readonly string[], tagId: string): Promise<void>;
  removeTagFromDocuments(documentIds: readonly string[], tagId: string): Promise<void>;
  /** Creates a Tag; the list follows the core's event. Throws what the core throws. */
  createTag(name: string, description?: string): Promise<Tag>;
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

const fileName = (path: string) => path.split(/[\\/]/).at(-1) ?? path;

/** Puts a Document in the list: in place if it's there, otherwise first. */
const upsert = (documents: Document[], item: Document) =>
  documents.some((each) => each.id === item.id)
    ? documents.map((each) => (each.id === item.id ? item : each))
    : [item, ...documents];

/** Puts a Linked folder in the list in place, if it's there. New ones arrive with the core's event. */
const replaceLinked = (linkedFolders: LinkedFolder[], linked: LinkedFolder) =>
  linkedFolders.map((each) => (each.id === linked.id ? linked : each));

/** The last list `selectVisibleDocuments` filtered, and what from. */
let visible: { documents: Document[]; tagFilter: readonly string[]; shown: Document[] } | null =
  null;

/**
 * The Documents the sidebar lists: all of them, or those with a Tag it
 * filters by. The same array until the Documents or the filter change, so it
 * needn't be compared item by item at each change to the store (#156).
 */
export const selectVisibleDocuments = (state: AppState): Document[] => {
  const { documents, tagFilter } = state;
  if (tagFilter.length === 0) return documents;
  if (visible?.documents !== documents || visible.tagFilter !== tagFilter) {
    visible = {
      documents,
      tagFilter,
      shown: documents.filter((item) => matchesTags(item, tagFilter)),
    };
  }
  return visible.shown;
};

/** What the Tag filter is, as a key: folds made while filtering belong to it. Null without one. */
export const selectTagFilterKey = (state: AppState): string | null =>
  state.tagFilter.length === 0 ? null : [...state.tagFilter].sort().join(" ");

/**
 * The core's per-Document events (status, Tags, moves, removals, Library
 * assignments), applied to the store together at most every 100ms (see
 * `createDocumentUpdates`). A store action that sets the Documents or the
 * Library itself flushes them first, so nothing older lands on top of it.
 */
const documentUpdates = createDocumentUpdates((changes) =>
  useAppStore.setState((state) => ({
    documents: applyDocumentChanges(state.documents, changes.upserted, changes.removed),
    // Before the Library is loaded there is nothing to change: loading reads it whole.
    ...(state.library && { library: applyAssignments(state.library, changes.assignments) }),
  })),
);

/**
 * Whole lists the core pushes by event, and `load` reads too. A list pushed
 * after `load` read its own is newer, so `load` must not put its copy back
 * over it: on a slow first run, the example Linked folder's Folder, made by
 * its first scan, was pushed while `load` still waited, and then vanished.
 * The core's events and replies arrive in the order it sent them.
 */
type PushedList =
  | "settings"
  | "minds"
  | "folders"
  | "linkedFolders"
  | "keptCitationTexts"
  | "tags"
  | "skills"
  | "chatReadiness"
  | "embeddingModel"
  | "embedding"
  | "rerank"
  | "privacy";
let pushes = 0;
const lastPushed = new Map<PushedList, number>();
const pushed = (list: PushedList) => lastPushed.set(list, ++pushes);

/** Reads a list for `load`, noting which pushes it already includes. */
function readList<T>(list: PushedList, request: Promise<T>, readAt: Map<PushedList, number>) {
  return request.then((value) => {
    readAt.set(list, pushes);
    return value;
  });
}

/** Whether the list was pushed after `load` read it. */
const pushedSince = (list: PushedList, readAt: Map<PushedList, number>) =>
  (lastPushed.get(list) ?? 0) > (readAt.get(list) ?? 0);

export const useAppStore = create<AppState>()((set, get) => {
  let libraryRequest = 0;
  /** The interface's words, outside React: for the lines the store writes (Undo's). */
  const t = (key: MessageKey, params?: MessageParams) =>
    translate(
      get().settings?.language ?? resolveLanguage("system", navigator.languages),
      key,
      params,
    );

  /**
   * Moves, shown at once and then sent to the core; if the core refuses, the
   * Minds and the Library are read again, as it has them. Returns what it did.
   */
  const move = async (items: MoveItems, folderId: string | null) => {
    documentUpdates.flush();
    set((state) => movedLocally(state, items, folderId));
    try {
      return await core.moveToFolder({ ...itemsOf(items), folderId });
    } catch (error) {
      set({ minds: await core.listMinds() });
      await get().refreshLibrary();
      throw error;
    }
  };
  /** Runs an action, reporting a failure instead of throwing. */
  const attempt = async (action: () => Promise<void>) => {
    try {
      await action();
    } catch (error) {
      useNotices.getState().showError(error);
    }
  };
  const push = (change: Undoable) => useUndo.getState().push(change);

  /**
   * Refuses to put something back in a Folder deleted since, as the core would,
   * offering to restore the Folder first (`names`: theirs when the change was made).
   */
  const refuseIfDeleted = (folderIds: readonly string[], names: ReadonlyMap<string, string>) => {
    const { library } = get();
    if (!library) return;
    const gone = folderIds.find((id) => !library.groups.some((group) => group.id === id));
    if (gone === undefined) return;
    const name = names.get(gone) ?? "";
    throw new UndoRefused(t("undo.folderGone", { name }), {
      label: t("undo.restoreFolder", { name }),
      run: async () => {
        await core.restore([{ kind: "folder", id: gone }]);
        await get().refreshLibrary();
      },
    });
  };

  /** Takes a move back: each Mind and conversation to where it was, each Document's assignment as it was. */
  const putBack = async (done: FolderMove, names: ReadonlyMap<string, string>) => {
    refuseIfDeleted(foldersMovedFrom(done), names);
    for (const back of undoMoves(done)) await move(back.items, back.folderId);
    for (const document of done.documents)
      await core.restoreAssignment(document.id, document.previous);
  };

  /**
   * The Folder made last, until it is named: its first name is part of making
   * it, so that replaces its step and one Undo takes both back.
   */
  let justMade: { id: string; step: Undoable; stepNamed(name: string): Undoable } | null = null;

  /** A Mind renamed from `before`, with Undo, if its title changed. */
  const offerRename = (mind: Mind, before: string) => {
    if (before === mind.title) return;
    const shown = mindName(mind) || t("mind.untitled");
    const rename = (name: string) => core.renameMind(mind.id, name);
    push(renamed(t, shown, rename, { before, after: mind.title }));
  };

  /** Puts a Folder's fields back as given. */
  const folderFields = (id: string, group: LibraryGroupInput) => async () => {
    await core.updateLibraryGroup(id, group);
    await get().refreshLibrary();
  };

  /** Each Document's name, as the undo line says it. */
  const documentNames = (ids: readonly string[]) =>
    ids.map((id) => {
      const document = get().documents.find((each) => each.id === id);
      return document ? displayName(document) : "";
    });

  /** Puts a Tag on Documents, or takes it off, with Undo: their links go back as they were. */
  const retag = async (added: boolean, documentIds: readonly string[], tagId: string) => {
    const ids = [...documentIds];
    const before = tagLinks(get().documents, ids, tagId);
    const names = documentNames(ids);
    const again = () =>
      added ? core.addTagToDocuments(ids, tagId) : core.removeTagFromDocuments(ids, tagId);
    await again();
    const tag = get().tags.find((each) => each.id === tagId)?.name ?? "";
    push(retagged(t, { added, tag, names }, before, again));
  };

  /** Closes a tab, then says so with Reopen. */
  const closeTabWith = async (tab: LayoutTab) => {
    if (await get().changeLayout({ type: "close", tabId: tab.id })) {
      get().showTabNotice({ kind: "closed", item: tab.item, count: 1 });
    }
  };

  /**
   * Opens the link dialog on a folder: at once, then with what linking it
   * would take once that's counted (unless it already is).
   */
  const showLinking = async (folder: LinkingFolder) => {
    set({ linking: folder });
    if (folder.preview || folder.error) return;
    const { path } = folder;
    try {
      const preview = await core.previewLinkedFolder(path);
      if (get().linking?.path === path) set({ linking: { path, preview, error: null } });
    } catch (error) {
      if (get().linking?.path === path) {
        set({ linking: { path, preview: null, error: errorMessage(error, t) } });
      }
    }
  };

  /** Asks about folders to link, one at a time: now if the dialog is free, otherwise after the others. */
  const offerLinks = (folders: readonly LinkingFolder[]) => {
    const [first, ...rest] = get().linking ? [] : folders;
    set((state) => ({ linkQueue: [...state.linkQueue, ...(first ? rest : folders)] }));
    if (first) void showLinking(first);
  };

  /** The dialog is done with a folder: it asks about the next one, if any. */
  const nextLink = () => {
    const [next, ...rest] = get().linkQueue;
    set({ linking: null, linkQueue: rest });
    if (next) void showLinking(next);
  };

  /**
   * Adds files by path, as Documents: names those IncarnaMind has already,
   * and those it can't take, once (in the footer), and offers any that turn
   * out to be folders for linking.
   */
  const addByPath = async (
    paths: readonly string[],
    folders: readonly string[],
    notOnDisk: readonly string[],
  ) => {
    // Before adding: what status events bring meanwhile is new, not something added before.
    documentUpdates.flush();
    const known = new Set(get().documents.map((each) => each.id));
    const result =
      paths.length > 0 ? await core.addDocuments([...paths]) : { documents: [], skipped: [] };
    const linkable: LinkingFolder[] = folders.map((path) => ({ path, preview: null, error: null }));
    const skipped = [...notOnDisk];
    for (const { path } of result.skipped) {
      // A folder (the drop didn't say) can't be added, but can be linked: only a folder has a preview.
      const preview = await core.previewLinkedFolder(path).catch(() => null);
      if (preview) linkable.push({ path, preview, error: null });
      else skipped.push(fileName(path));
    }
    documentUpdates.flush();
    set((state) => {
      // Status events may already have brought newer copies of these; keep those.
      const listed = new Set(state.documents.map((each) => each.id));
      const added = new Map<string, Document>();
      for (const item of result.documents) {
        if (!listed.has(item.id)) added.set(item.id, item);
      }
      return {
        documents: [...[...added.values()].reverse(), ...state.documents],
        skippedFiles: skipped,
        alreadyAdded: [
          ...new Set(
            result.documents.filter((item) => known.has(item.id)).map((item) => item.name),
          ),
        ],
      };
    });
    offerLinks(linkable);
  };

  /** Dropped or picked files' paths, and the names of those that aren't on disk. */
  const pathsOf = (picked: readonly File[]) => {
    const paths: string[] = [];
    const notOnDisk: string[] = [];
    for (const file of picked) {
      const path = files.pathForFile(file);
      if (path) paths.push(path);
      else notOnDisk.push(file.name);
    }
    return { paths, notOnDisk };
  };

  return {
    ...createLayoutSlice(set, get, (error) => useNotices.getState().showError(error)),
    status: { kind: "loading" },
    minds: [],
    titleToFocus: null,
    pickingDocuments: false,
    settings: null,
    chatReadiness: null,
    documents: [],
    keptCitationTexts: [],
    skippedFiles: [],
    alreadyAdded: [],
    embeddingModel: null,
    embedding: null,
    rerank: null,
    settingsOpen: false,
    settingsPage: "general",
    folders: [],
    linkedFolders: [],
    linking: null,
    linkQueue: [],
    tags: [],
    tagFilter: [],
    tagsDialogOpen: false,
    library: null,
    libraryFilter: "all",
    refreshLibrary: async () => {
      const ticket = ++libraryRequest;
      const library = await core.getLibrary();
      if (ticket !== libraryRequest) return;
      // The assignments that came before the snapshot are in it already: it replaces them.
      documentUpdates.flush();
      set({
        library,
        ...(!["all", "unsorted", "starters", LIBRARY_DELETED].includes(get().libraryFilter) &&
        !library.groups.some((g) => g.id === get().libraryFilter)
          ? { libraryFilter: "all" }
          : {}),
      });
    },
    openLibrary: (filter = "all", options) => {
      set({ libraryFilter: filter });
      void get().changeLayout({
        type: "open",
        item: { kind: "library" },
        where: options?.side ? { in: "side" } : { in: "place" },
      });
    },
    skills: [],
    examples: null,
    privacy: null,
    questionToStart: null,

    async load() {
      try {
        // A first run: the example Mind is made, once, and is what the app opens on.
        const offered = await core.offerExamples().catch(() => null);
        if (offered?.mindId) await startGettingStarted(offered.mindId);
        await get().loadLayout();
        const readAt = new Map<PushedList, number>();
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
          keptCitationTexts,
          examples,
          connectors,
          rerank,
          privacy,
        ] = await Promise.all([
          readList("minds", core.listMinds(), readAt),
          readList("settings", core.getSettings(), readAt),
          core.listDocuments(),
          readList("chatReadiness", core.getChatReadiness(), readAt),
          readList("folders", core.listFolders(), readAt),
          readList("embeddingModel", core.getEmbeddingModel(), readAt),
          readList("tags", core.listTags(), readAt),
          readList("skills", core.listSkills(), readAt),
          readList("embedding", core.getEmbeddingSettings(), readAt),
          readList("linkedFolders", core.listLinkedFolders(), readAt),
          readList("keptCitationTexts", core.listKeptCitationTexts(), readAt),
          core.getExamples(),
          core.listConnectors(),
          readList("rerank", core.getRerankSettings(), readAt),
          readList("privacy", core.getPrivacySettings(), readAt),
        ]);
        // What the core's events brought before this is in the lists just read.
        documentUpdates.flush();
        const read = {
          linkedFolders,
          minds,
          settings,
          keptCitationTexts,
          chatReadiness,
          folders,
          embeddingModel,
          tags,
          skills,
          embedding,
          rerank,
          privacy,
        } satisfies Pick<AppState, PushedList>;
        const fresh = Object.fromEntries(
          Object.entries(read).filter(([list]) => !pushedSince(list as PushedList, readAt)),
        ) as Partial<Pick<AppState, PushedList>>;
        set({ ...fresh, documents, examples, status: { kind: "ready" } });
        await get().refreshLibrary();
        tickIndexed(connectors.length);
      } catch (error) {
        set({ status: { kind: "failed", message: errorMessage(error, t) } });
      }
    },

    createMind: (options) =>
      attempt(async () => {
        const mind = await core.createMind({
          ...(options?.title && { title: options.title }),
          ...(options?.folderId && { folderId: options.folderId }),
        });
        set(
          options?.startQuestion
            ? { questionToStart: { mindId: mind.id, scope: options.scope ?? null } }
            : { titleToFocus: mind.id },
        );
        await get().showNewMind(mind, options?.groupId);
      }),

    showNewMind: async (mind, groupId) => {
      // The "minds.changed" event may have listed it already.
      set((state) => ({ minds: [mind, ...state.minds.filter((each) => each.id !== mind.id)] }));
      await get().changeLayout({
        type: "open",
        item: { kind: "mind", mindId: mind.id },
        where: { in: "group", ...(groupId && { groupId }) },
      });
    },

    titleFocused: () => set({ titleToFocus: null }),

    startQuestion() {
      const { layout, createMind } = get();
      const mindId = currentMindIdOf(layout);
      if (mindId) set({ questionToStart: { mindId, scope: null } });
      else void createMind({ startQuestion: true });
    },

    questionStarted: () => set({ questionToStart: null }),

    async askAbout({ scope }) {
      const { minds, layout, examples } = get();
      const mindId = mindToAskIn({
        minds,
        openMindId: currentMindIdOf(layout),
        exampleMindId: examples?.mindId ?? null,
      });
      if (!mindId) return get().createMind({ startQuestion: true, scope });
      get().openMind(mindId);
      set({ questionToStart: { mindId, scope } });
    },

    // From a Folder as it is, the new Mind is the Folder's.
    startMindFrom: ({ kind, scope, title }) =>
      get().createMind({
        startQuestion: true,
        title,
        scope,
        folderId: kind === "folder" ? (scope.folderIds[0] ?? null) : null,
      }),

    moveToFolder: (items, folderId) =>
      attempt(async () => {
        const { minds, documents, library } = get();
        // Only what isn't there already: dropping a Mind back on its own Folder moves nothing.
        const folderOf = new Map(
          library?.assignments.map((each) => [each.documentId, each.groupId]),
        );
        const mindIds = itemsOf(items).mindIds.filter(
          (id) => minds.find((mind) => mind.id === id)?.folderId !== folderId,
        );
        const documentIds = itemsOf(items).documentIds.filter(
          (id) => (folderOf.get(id) ?? null) !== folderId,
        );
        if (mindIds.length + documentIds.length === 0) return;
        const moving = { mindIds, documentIds };
        const folderName =
          folderId === null
            ? t("library.unsorted")
            : (library?.groups.find((group) => group.id === folderId)?.name ?? "");
        const message = moveMessage(t, moving, folderName, { minds, documents });
        const names = new Map(library?.groups.map((group) => [group.id, group.name]));
        let done = await move(moving, folderId);
        push({
          message,
          name: moveName(t, moving, { minds, documents }),
          undo: () => putBack(done, names),
          redo: async () => {
            if (folderId !== null) refuseIfDeleted([folderId], names);
            done = await move(moving, folderId);
          },
        });
      }),

    createFolder: async (items) => {
      let madeId: string | null = null;
      await attempt(async () => {
        const { minds, documents, library } = get();
        const name = untitledFolderName(
          t,
          (library?.groups ?? []).map((group) => group.name),
        );
        const folder = await core.createLibraryGroup({ name, description: "" });
        await get().refreshLibrary();
        const chosen = items ? itemsOf(items) : null;
        const going =
          chosen && chosen.mindIds.length + chosen.documentIds.length > 0 ? chosen : null;
        let done = going ? await move(going, folder.id) : null;
        const names = new Map(library?.groups.map((group) => [group.id, group.name]));
        const stepNamed = (shown: string): Undoable => ({
          message: madeMessage(t, shown, going, { minds, documents }),
          name: t("undo.name.newFolder"),
          undo: async () => {
            if (done) await putBack(done, names);
            await core.deleteLibraryGroup(folder.id);
            await get().refreshLibrary();
          },
          redo: async () => {
            await core.restore([{ kind: "folder", id: folder.id }]);
            await get().refreshLibrary();
            if (going) done = await move(going, folder.id);
          },
        });
        const step = stepNamed(name);
        justMade = { id: folder.id, step, stepNamed };
        push(step);
        useRequests.getState().ask({ nameFolder: folder.id });
        madeId = folder.id;
      });
      return madeId;
    },

    nameNewFolder: (folderId, name) =>
      attempt(async () => {
        const folder = get().library?.groups.find((group) => group.id === folderId);
        const trimmed = name.trim();
        if (!folder || !trimmed || trimmed === folder.name) return;
        const made = justMade;
        if (made?.id !== folderId || useUndo.getState().done.at(-1) !== made.step) {
          await get().editFolder(folderId, { name: trimmed, description: folder.description });
          return;
        }
        await core.updateLibraryGroup(folderId, { name: trimmed, description: folder.description });
        await get().refreshLibrary();
        justMade = null;
        useUndo.getState().withdraw(made.step);
        push(made.stepNamed(trimmed));
      }),

    renameFolder: (folderId, name) =>
      attempt(async () => {
        const folder = get().library?.groups.find((group) => group.id === folderId);
        const trimmed = name.trim();
        if (!folder || !trimmed || trimmed === folder.name) return;
        await get().editFolder(folderId, { name: trimmed, description: folder.description });
      }),

    async editFolder(folderId, input) {
      const folder = get().library?.groups.find((group) => group.id === folderId);
      const edited = await core.updateLibraryGroup(folderId, input);
      await get().refreshLibrary();
      if (!folder) return;
      const fields = (group: LibraryGroup) => ({
        name: group.name,
        description: group.description,
        countsOwnFiles: group.countsOwnFiles,
      });
      const before = fields(folder);
      const after = fields(edited);
      if (JSON.stringify(before) === JSON.stringify(after)) return;
      const rename = before.name !== after.name;
      push({
        message: t(rename ? "object.renamed" : "undo.folderEdited", { name: after.name }),
        name: t(rename ? "undo.name.rename" : "undo.name.editFolder"),
        undo: folderFields(folderId, before),
        redo: folderFields(folderId, after),
      });
    },

    // Its Minds' list follows the core's "minds.changed" event, the Library its "library.changed".
    deleteFolder: (folderId) =>
      attempt(async () => {
        const folder = get().library?.groups.find((group) => group.id === folderId);
        const remove = async () => {
          await core.deleteLibraryGroup(folderId);
          await get().refreshLibrary();
        };
        await remove();
        if (folder) push(deleted(t, folder.name, [{ kind: "folder", id: folderId }], remove));
      }),

    openExamples: () =>
      attempt(async () => {
        let { examples } = get();
        if (!examples?.mindId) {
          examples = await core.createExamples();
          set({ examples });
        }
        if (examples.mindId) get().openMind(examples.mindId);
      }),

    // The core closes the Mind's tab; its Documents go with their Linked folder.
    removeExamples: () =>
      attempt(async () => {
        const group = get().examples?.group;
        const remove = () => core.removeExamples(group);
        await remove();
        push({
          message: t("undo.examplesRemoved"),
          name: t("undo.name.removeExamples"),
          undo: async () => {
            await core.createExamples(group);
          },
          redo: remove,
        });
      }),

    updateGettingStarted(patch) {
      const { settings } = get();
      const current = settings?.device.gettingStarted;
      if (!settings || !current?.started) return;
      const next = { ...current, ...patch };
      if (
        Object.entries(next).every(([key, value]) => current[key as keyof GettingStarted] === value)
      ) {
        return;
      }
      // At once, so ticks made together don't undo each other; the core's event confirms it.
      set({ settings: { ...settings, device: { ...settings.device, gettingStarted: next } } });
      core.updateSettings({ device: { gettingStarted: next } }).catch(() => undefined);
    },

    openMind(id, options) {
      void get().changeLayout({
        type: "open",
        item: { kind: "mind", mindId: id },
        where: options?.side ? { in: "side" } : { in: "place" },
      });
    },

    closeTab(tabId) {
      const tab = get()
        .layout?.groups.flatMap((group) => group.tabs)
        .find((each) => each.id === tabId);
      if (tab) void closeTabWith(tab);
    },

    reopenTab() {
      if (!get().layout?.reopenable) return;
      get().showTabNotice(null);
      void get().changeLayout({ type: "reopen" });
    },

    // The list follows the core's "minds.changed" event, which arrives before these calls return.
    renameMind: (id, title, options) =>
      attempt(async () => {
        const before = get().minds.find((mind) => mind.id === id)?.title;
        const mind = await core.renameMind(id, title);
        if (!options?.typing && before !== undefined) offerRename(mind, before);
      }),

    titleLeft(id, before, after) {
      const mind = get().minds.find((each) => each.id === id);
      // Its last letters may still be on their way to the core: the title is what was typed.
      if (mind) offerRename({ ...mind, title: after.trim() }, before);
    },

    deleteMind: (id) =>
      attempt(async () => {
        const mind = get().minds.find((each) => each.id === id);
        const remove = () => core.deleteMind(id);
        await remove();
        const shown = (mind && mindName(mind)) || t("mind.untitled");
        push(deleted(t, shown, [{ kind: "mind", id }], remove));
      }),

    reportError(error) {
      useNotices.getState().showError(error);
    },

    openDocument(location, how = { as: "list" }) {
      const { documentId, ...place } = location;
      const item = { kind: "document", documentId } as const;
      const at = Object.keys(place).length > 0 ? { place } : {};
      // A merged window shows one group: what opens to the side comes to the front.
      const merged = groupsMerged();
      const keep = how.keep ? { keep: true } : {};
      const change: LayoutChange =
        how.as === "here"
          ? {
              type: "open",
              item,
              ...at,
              preview: false,
              where: { in: "group", ...(how.from && { groupId: how.from }) },
            }
          : how.as === "list"
            ? { type: "open", item, ...at, preview: true, where: { in: "beside" }, focus: merged }
            : {
                type: "open",
                item,
                ...at,
                ...keep,
                preview: true,
                where: { in: "side", ...(how.from && { from: how.from }) },
                focus: merged,
              };
      void get().changeLayout(change);
    },

    openSettings(page) {
      // Also a click handler: anything but a page name opens General.
      set({ settingsOpen: true, settingsPage: resolveSettingsPage(page) });
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

    addDocuments: (picked) =>
      attempt(async () => {
        const { paths, notOnDisk } = pathsOf(picked);
        await addByPath(paths, [], notOnDisk);
      }),

    addDropped: (dropped, folders) =>
      attempt(async () => {
        const folderPaths = pathsOf(folders).paths;
        const { paths, notOnDisk } = pathsOf(dropped);
        const filePaths = paths.filter((path) => !folderPaths.includes(path));
        await addByPath(filePaths, folderPaths, notOnDisk);
      }),

    addPaths: (paths) => attempt(() => addByPath(paths, [], [])),

    pickDocuments: () =>
      attempt(async () => {
        if (get().pickingDocuments) return;
        set({ pickingDocuments: true });
        try {
          const paths = await files.pickDocuments();
          if (paths.length > 0) await addByPath(paths, [], []);
        } finally {
          set({ pickingDocuments: false });
        }
      }),

    async locateDocumentFile(id) {
      const paths = await files.pickDocuments();
      if (paths.length === 0) return "cancelled";
      const result = await core.addDocuments(paths);
      documentUpdates.flush();
      set((state) => ({
        documents: result.documents.reduce(upsert, state.documents),
      }));
      return result.documents.some((each) => each.id === id && each.fileStatus === "available")
        ? "located"
        : "other";
    },

    renameDocument: (id, name) =>
      attempt(async () => {
        const before = get().documents.find((each) => each.id === id)?.name;
        const rename = async (to: string) => {
          const document = await core.renameDocument(id, to);
          documentUpdates.flush();
          set((state) => ({ documents: upsert(state.documents, document) }));
          return document;
        };
        const document = await rename(name);
        if (before === undefined || before === document.name) return;
        push(renamed(t, displayName(document), rename, { before, after: document.name }));
      }),

    deleteDocuments: (ids) =>
      attempt(async () => {
        const first = get().documents.find((each) => each.id === ids[0]);
        const remove = async () => {
          for (const id of ids) await core.deleteDocument(id);
          documentUpdates.flush();
          set((state) => ({ documents: state.documents.filter((each) => !ids.includes(each.id)) }));
        };
        await remove();
        if (ids.length === 0) return;
        const shown = first ? displayName(first) : "";
        push(
          deleted(
            t,
            shown,
            ids.map((id) => ({ kind: "document", id })),
            remove,
          ),
        );
      }),

    // Its status follows the core's "document.status" event too.
    retryDocument: (id) =>
      attempt(async () => {
        const retried = await core.retryDocument(id);
        documentUpdates.flush();
        set((state) => ({ documents: upsert(state.documents, retried) }));
      }),

    dismissSkippedFiles() {
      set({ skippedFiles: [] });
    },

    dismissAlreadyAdded() {
      set({ alreadyAdded: [] });
    },

    downloadEmbeddingModel: () =>
      attempt(async () => {
        set({ embeddingModel: await core.downloadEmbeddingModel() });
      }),

    downloadRerankingModel: () =>
      attempt(async () => {
        set({ rerank: await core.downloadRerankingModel() });
      }),

    retryEmbedding: () =>
      attempt(async () => {
        set({ embedding: await core.retryEmbedding() });
      }),

    addLinkedFolder: () =>
      attempt(async () => {
        const path = await files.pickLinkedFolder();
        // The dialog opens at once; a big folder takes a moment to count.
        if (path) offerLinks([{ path, preview: null, error: null }]);
      }),

    // Its Documents, Folders and progress arrive with the core's events.
    confirmLinkedFolder: (layout) =>
      attempt(async () => {
        const preview = get().linking?.preview;
        nextLink();
        if (!preview) return;
        const linked = await core.addLinkedFolder(preview.path);
        // Inside a Linked folder already: nothing was linked, so its layout stays as the User set it.
        if (preview.insideLinkedFolderId !== null) return;
        // Set now, before its first scan ends, so the scan's own suggestion doesn't replace it.
        const shown = await core.setLinkedFolderLayout(linked.id, layout);
        set((state) => ({ linkedFolders: replaceLinked(state.linkedFolders, shown) }));
      }),

    // The dialog's own close (Esc, or after linking) calls this too: only a folder still asked about counts.
    cancelLinkedFolder() {
      if (get().linking) nextLink();
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

    // Its Documents follow the core's events, as its scan finds them there.
    locateLinkedFolder: (linkedFolderId) =>
      attempt(async () => {
        const path = await files.pickLinkedFolder(linkedFolderId);
        if (!path) return;
        const linked = await core.locateLinkedFolder(linkedFolderId, path);
        set((state) => ({ linkedFolders: replaceLinked(state.linkedFolders, linked) }));
      }),

    // The list, its Folders and Documents follow the core's events.
    removeLinkedFolder: (linkedFolderId) =>
      attempt(async () => {
        await core.removeLinkedFolder(linkedFolderId);
      }),

    toggleTagFilter(tagId) {
      const current = get().tagFilter;
      set({
        tagFilter: current.includes(tagId)
          ? current.filter((each) => each !== tagId)
          : [...current, tagId],
      });
    },

    setTagFilter(tagIds) {
      set({ tagFilter: [...new Set(tagIds)] });
    },

    // The Documents follow the core's "documents.tagged" event, which arrives before these calls return.
    addDocumentTag: (documentId, tagId) => attempt(() => retag(true, [documentId], tagId)),

    removeDocumentTag: (documentId, tagId) => attempt(() => retag(false, [documentId], tagId)),

    addTagToDocuments: (documentIds, tagId) => attempt(() => retag(true, documentIds, tagId)),

    removeTagFromDocuments: (documentIds, tagId) => attempt(() => retag(false, documentIds, tagId)),

    async createTag(name, description = "") {
      const tag = await core.createTag({ name, description });
      // The list follows "tags.changed"; this makes the new Tag usable at once.
      set((state) => ({
        tags: state.tags.some((each) => each.id === tag.id)
          ? state.tags
          : [...state.tags, tag].sort((a, b) =>
              a.name.localeCompare(b.name, undefined, { sensitivity: "base" }),
            ),
      }));
      return tag;
    },

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

/**
 * The checklist starts with the example Mind on a first run, which opens as
 * the only tab, before the store has loaded the settings.
 */
async function startGettingStarted(exampleMindId: string): Promise<void> {
  try {
    const { device } = await core.getSettings();
    await core.updateSettings({
      device: { gettingStarted: { ...device.gettingStarted, started: true } },
    });
    await core.changeLayout(WINDOW_ID, {
      type: "open",
      item: { kind: "mind", mindId: exampleMindId },
    });
  } catch {
    // Then the app opens as usual, without the checklist.
  }
}

/** A Document of the User's own: not one of the examples. */
export const isOwnDocument = (document: Document, examples: Examples | null) =>
  document.linkedFolderId === null || document.linkedFolderId !== examples?.linkedFolderId;

/** Ticks "Index your Documents or connect apps" once there are Documents of the User's own, or Connectors. */
function tickIndexed(connectors: number): void {
  const { documents, examples, updateGettingStarted } = useAppStore.getState();
  if (connectors > 0 || documents.some((each) => isOwnDocument(each, examples)))
    updateGettingStarted({ indexed: true });
}

// Settings can change outside this window (another window, or the core itself), so follow the core's event.
core.on("settings.changed", (settings) => {
  pushed("settings");
  useAppStore.setState({ settings });
});

// The same for the list of Minds: created, renamed, deleted, or reordered by an edit.
// A deleted Mind's tab closes in the core, which says so with "layout.changed".
core.on("minds.changed", (minds) => {
  pushed("minds");
  const { examples } = useAppStore.getState();
  useAppStore.setState({ minds });
  // The example Mind deleted like any other: no longer the examples.
  if (examples?.mindId && !minds.some((mind) => mind.id === examples.mindId)) {
    core.getExamples().then(
      (current) => useAppStore.setState({ examples: current }),
      () => undefined,
    );
  }
});

followLayout((partial) => useAppStore.setState(partial));

// Processing happens in the background: follow each Document's status as the core reports it.
core.on("document.status", (changed) => {
  documentUpdates.upsert([changed]);
  const { examples, updateGettingStarted } = useAppStore.getState();
  if (isOwnDocument(changed, examples)) updateGettingStarted({ indexed: true });
});

// The examples made, opened again or removed: their Mind and Linked folder show "Example".
// The screen shows the tea example; the groups' examples open from the first run's picks.
core.on("examples.changed", (examples) => {
  if (examples.group === "tea") useAppStore.setState({ examples });
});

// Privacy choices change on the Privacy page, at the first run's question, or with local mode.
core.on("privacy.changed", (privacy) => {
  pushed("privacy");
  useAppStore.setState({ privacy });
});

/**
 * Whether the first run's question about usage data (in a test build, its
 * notice) is waiting: this copy can send usage data, the User hasn't
 * answered, and local mode isn't keeping it off.
 */
export const usageDataQuestionWaiting = (privacy: PrivacySettings | null): boolean =>
  privacy?.usageData.available === true && !privacy.usageData.asked && !privacy.usageData.localMode;

/**
 * Usage data from the interface (see src/core/usageEvents.ts): the core
 * sends it only while the User agrees, and drops it otherwise.
 */
export function recordUsage(event: UiUsageEvent): void {
  core.recordUsage(event).catch((error: unknown) => console.error(error));
}

// A Connector is one way to "Index your Documents or connect apps".
core.on("connectors.changed", (connectors) => tickIndexed(connectors.length));

// A Question of the User's own: anything but the example Answer, written again.
core.on("answer.started", ({ answerId }) => {
  const { examples, updateGettingStarted } = useAppStore.getState();
  if (answerId !== examples?.answerId) updateGettingStarted({ askedOwn: true });
});

core.on("chatReadiness.changed", (chatReadiness) => {
  pushed("chatReadiness");
  useAppStore.setState({ chatReadiness });
});

// An outside agent asked to show one of its Citations (#351): it opens beside the Mind, at its quote.
core.on("agents.citationShown", (place) => {
  const state = useAppStore.getState();
  if (state.settingsOpen) state.closeSettings();
  state.openDocument(place, { as: "list" });
});

// The embedding model downloads in the background: follow its state and progress.
core.on("embeddingModel.status", (embeddingModel) => {
  pushed("embeddingModel");
  useAppStore.setState({ embeddingModel });
});

// The embedding model can change in Settings or by local mode, and a rebuild reports its progress.
core.on("embedding.changed", (embedding) => {
  pushed("embedding");
  useAppStore.setState({ embedding });
});

// Reranking changes in Settings, and the built-in reranking model downloads in the background.
core.on("rerank.changed", (rerank) => {
  pushed("rerank");
  useAppStore.setState({ rerank });
});

// Folders change through this window or another: follow the list. The sidebar's tree follows it.
core.on("folders.changed", (folders) => {
  pushed("folders");
  useAppStore.setState({ folders });
});

// Linked folders added, removed, or scanning, paused, out of reach, or making progress.
core.on("linkedFolders.changed", (linkedFolders) => {
  pushed("linkedFolders");
  useAppStore.setState({ linkedFolders });
});

// A Linked folder was unlinked: the text its Citations quote is kept, so they can still be checked.
core.on("keptCitationTexts.changed", (keptCitationTexts) => {
  pushed("keptCitationTexts");
  useAppStore.setState({ keptCitationTexts });
});

// Documents removed from the index, e.g. with their Linked folder.
core.on("documents.removed", (removed) => documentUpdates.remove(removed));

// Documents whose files moved to another Folder on disk.
core.on("documents.moved", (moved) => documentUpdates.upsert(moved));

// Tags change through this window or another: follow the list, and drop a deleted Tag from the filter.
core.on("tags.changed", (tags) => {
  pushed("tags");
  const { tagFilter } = useAppStore.getState();
  const kept = tagFilter.filter(
    (value) => value === NEEDS_REVIEW || tags.some((tag) => tag.id === value),
  );
  useAppStore.setState({ tags, ...(kept.length !== tagFilter.length && { tagFilter: kept }) });
});

// Skills are imported, turned on or off, or removed, through this window or another.
core.on("skills.changed", (skills) => {
  pushed("skills");
  useAppStore.setState({ skills });
});

// Documents' Tags, or their tagging, changed: by the User, by automatic tagging, or with a deleted Tag.
core.on("documents.tagged", (tagged) => documentUpdates.upsert(tagged));

// Documents' reference records changed: the User edited one, put one back, or Look up filled one.
core.on("references.changed", (changed) => documentUpdates.upsert(changed));

// Organize queued, started, finished or failed on Documents, or the User put one in a Folder.
core.on("library.assignments", (assignments) => documentUpdates.assign(assignments));

// The Folders or the settings changed: read the Library again.
core.on("library.changed", () => {
  void useAppStore
    .getState()
    .refreshLibrary()
    .catch((error) => useAppStore.getState().reportError(error));
});
