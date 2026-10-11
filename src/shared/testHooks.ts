/**
 * UI hooks for the smoke tests, installed as `window.incarnamindTestHooks` only
 * when the app is launched with INCARNAMIND_TEST_HOOKS=1.
 */
import type { DocumentLocation } from "./documentViewer";

export interface TestHooks {
  /**
   * What a Citation does: opens a Document at a page range, highlighting a
   * quote, as a preview in the group right of the Mind the User is in; with
   * no Mind in front, as a Document opened from the sidebar.
   */
  openDocument(location: DocumentLocation): void;
  /**
   * The next "Import a folder…" or "Import a zip…" in Settings → Skills gets
   * this path (null: cancelled) instead of showing the system's open dialog.
   */
  interceptSkillPicker(path: string | null): void;
  /**
   * What dropping these files and folders (absolute paths) on the window
   * does, once their paths are known: a test can't drop a folder.
   */
  addPaths(paths: string[]): Promise<void>;
  /** How many animations are running, leaving copies of closed menus included: settled at 0. */
  motionRunning(): number;
  /** Plays every animation at this speed (0.25: four times slower), so a test can act during one. */
  setMotionSpeed(speed: number): void;
  /**
   * How each leaving copy (a closed menu fading where it was) shows now, as
   * nothing else can see them: its box, opacity, transform and its origin.
   */
  leavingCopies(): {
    box: { left: number; top: number; width: number; height: number };
    opacity: number;
    transform: string;
    origin: string;
  }[];
}
