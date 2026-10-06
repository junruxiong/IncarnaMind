/**
 * UI hooks for the smoke tests, installed as `window.incarnamindTestHooks` only
 * when the app is launched with INCARNAMIND_TEST_HOOKS=1.
 */
export interface TestHooks {
  openViewer(): void;
  closeViewer(): void;
}
