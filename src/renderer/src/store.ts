import { create } from "zustand";
import type {
  ChatReadiness,
  DeviceSettings,
  Document,
  Mind,
  Settings,
  SettingsPatch,
} from "../../core/api";
import { core, files } from "./core";

type Status = { kind: "loading" } | { kind: "ready" } | { kind: "failed"; message: string };

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
  /** The Document viewer panel on the right. Closed on launch; Documents and Citations open it in later tickets. */
  viewerOpen: boolean;
  /** Most recently added first. */
  documents: Document[];
  /** Names of the files the last add couldn't take, shown until dismissed. */
  skippedFiles: string[];
  settingsOpen: boolean;

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
  updateSettings(patch: SettingsPatch): Promise<void>;
  /** Changes pane widths on screen only, e.g. while dragging a divider. */
  previewLayout(layout: Partial<DeviceSettings>): void;
  dismissActionError(): void;
  /** Adds dropped or picked files as Documents. */
  addDocuments(picked: readonly File[]): Promise<void>;
  renameDocument(id: string, name: string): Promise<void>;
  deleteDocument(id: string): Promise<void>;
  dismissSkippedFiles(): void;
}

const messageOf = (error: unknown) => (error instanceof Error ? error.message : String(error));

const fileName = (path: string) => path.split(/[\\/]/).at(-1) ?? path;

/** Puts a Document in the list: in place if it's there, otherwise first. */
const upsert = (documents: Document[], item: Document) =>
  documents.some((each) => each.id === item.id)
    ? documents.map((each) => (each.id === item.id ? item : each))
    : [item, ...documents];

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
    documents: [],
    skippedFiles: [],
    settingsOpen: false,

    async load() {
      try {
        const [minds, settings, documents, chatReadiness] = await Promise.all([
          core.listMinds(),
          core.getSettings(),
          core.listDocuments(),
          core.getChatReadiness(),
        ]);
        set({ minds, settings, documents, chatReadiness, status: { kind: "ready" } });
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

    closeViewer() {
      set({ viewerOpen: false });
    },

    toggleViewer() {
      set((state) => ({ viewerOpen: !state.viewerOpen }));
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
  };
});

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
