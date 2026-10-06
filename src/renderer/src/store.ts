import { create } from "zustand";
import type { DeviceSettings, Mind, Settings, SettingsPatch } from "../../core/api";
import { core } from "./core";

type Status = { kind: "loading" } | { kind: "ready" } | { kind: "failed"; message: string };

interface AppState {
  status: Status;
  minds: Mind[];
  openMindId: string | null;
  /** Set once loaded. */
  settings: Settings | null;
  /** The last action that failed, shown until dismissed. */
  actionError: string | null;
  /** The Document viewer panel on the right. Closed on launch; Documents and Citations open it in later tickets. */
  viewerOpen: boolean;

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
  updateSettings(patch: SettingsPatch): Promise<void>;
  /** Changes pane widths on screen only, e.g. while dragging a divider. */
  previewLayout(layout: Partial<DeviceSettings>): void;
  dismissActionError(): void;
}

const messageOf = (error: unknown) => (error instanceof Error ? error.message : String(error));

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
    actionError: null,
    viewerOpen: false,

    async load() {
      try {
        const [minds, settings] = await Promise.all([core.listMinds(), core.getSettings()]);
        set({ minds, settings, status: { kind: "ready" } });
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
