import { create } from "zustand";
import type { ChatReadiness, DeviceSettings, Mind, Settings, SettingsPatch } from "../../core/api";
import { core } from "./core";

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
  settingsOpen: boolean;

  load(): Promise<void>;
  createMind(): Promise<void>;
  openMind(id: string): void;
  openViewer(): void;
  closeViewer(): void;
  toggleViewer(): void;
  openSettings(): void;
  closeSettings(): void;
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
    chatReadiness: null,
    actionError: null,
    viewerOpen: false,
    settingsOpen: false,

    async load() {
      try {
        const [minds, settings, chatReadiness] = await Promise.all([
          core.listMinds(),
          core.getSettings(),
          core.getChatReadiness(),
        ]);
        set({ minds, settings, chatReadiness, status: { kind: "ready" } });
      } catch (error) {
        set({ status: { kind: "failed", message: messageOf(error) } });
      }
    },

    createMind: () =>
      attempt(async () => {
        const mind = await core.createMind();
        set((state) => ({ minds: [mind, ...state.minds], openMindId: mind.id }));
      }),

    openMind(id) {
      set({ openMindId: id });
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
  };
});

// Settings can change outside this window (another window, or the core itself), so follow the core's event.
core.on("settings.changed", (settings) => useAppStore.setState({ settings }));
core.on("chatReadiness.changed", (chatReadiness) => useAppStore.setState({ chatReadiness }));
