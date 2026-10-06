/**
 * UI hooks for the smoke tests, installed as `window.incarnamindTestHooks` only
 * when the app is launched with INCARNAMIND_TEST_HOOKS=1.
 */
import type { DocumentLocation } from "./documentViewer";

export interface TestHooks {
  /** Opens the Document viewer panel without a Document. */
  openViewer(): void;
  closeViewer(): void;
  /** What a Citation does: opens a Document in the viewer at a page range, highlighting a quote. */
  openDocument(location: DocumentLocation): void;
}
