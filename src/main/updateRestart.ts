/**
 * Restarts to install an update. The Linux AppImage starts the new copy before this one has
 * quit, so this one lets go of the data folder first, or the new copy would find it taken and
 * quit at once (#1352).
 */
export function restartToInstall(
  updater: { quitAndInstall(): void },
  releaseDataFolder: () => void,
): void {
  releaseDataFolder();
  updater.quitAndInstall();
}
