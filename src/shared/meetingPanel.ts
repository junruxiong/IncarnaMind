/**
 * The floating panel during a call (#839): a small window above the meeting
 * app, with the help buttons and the latest suggestion. It shares the app's
 * preload, which gives its page the core and this bridge for the argument.
 */
import type { GeneralHelpKind } from "./meetingHelp";

export const MEETING_PANEL_ARGUMENT = "--incarnamind-meeting-panel";
export const MEETING_PANEL_BRIDGE_KEY = "incarnamindMeetingPanel";

/** The help buttons' global shortcuts the main process holds now, by button. */
export type HeldKeys = Partial<Record<GeneralHelpKind, string>>;

export const MEETING_PANEL_CHANNELS = {
  /** Page → main: make the window this size, in CSS pixels, keeping its top right corner. */
  size: "meetingPanel:size",
  /** Page → main: the User closed it; it comes back with the next meeting, or from ⌘K. */
  close: "meetingPanel:close",
  /** Page → main: bring the app's window to the front, at Settings › Models to set up a model. */
  setUp: "meetingPanel:setUp",
  /** Page → main: the shortcuts held now (`HeldKeys`). */
  keys: "meetingPanel:keys",
  /** Main → page: the shortcuts held changed. */
  keysChanged: "meetingPanel:keysChanged",
  /** Main → page: a button's shortcut was pressed. */
  help: "meetingPanel:help",
} as const;

export interface MeetingPanelBridge {
  resize(width: number, height: number): void;
  close(): void;
  setUp(): void;
  keys(): Promise<HeldKeys>;
  onKeys(listener: (keys: HeldKeys) => void): () => void;
  onHelp(listener: (kind: GeneralHelpKind) => void): () => void;
}
