// Build-time configuration of the main process, read by electron-vite from the
// environment or a .env file when the app is built. Only `MAIN_VITE_`
// variables reach the main process.

interface ImportMetaEnv {
  /**
   * Where crash reports go: a Sentry DSN, set by the release workflow from a
   * repository secret. Unset or empty, the build can't send crash reports and
   * Settings doesn't offer them. Never written in the code.
   */
  readonly MAIN_VITE_SENTRY_DSN?: string;
}
