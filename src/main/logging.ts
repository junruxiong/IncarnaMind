/**
 * Starts IncarnaMind's log (./log) for the desktop app: the app's start and
 * quit, errors written to the console (which is where the core and the main
 * process report problems), errors nothing caught in the main process, and
 * windows and processes that crashed. The core logs through the `log`
 * adapter (./platform); the window's uncaught errors come over the files
 * bridge (./files).
 */
import { arch, release } from "node:os";
import { app } from "electron";
import { createScrubber } from "./crashScrubber";
import { createFileLogger, createLogFile, type FileLogger, logConsole, logsFolder } from "./log";

/** Starts the log in the data folder's `logs/`. Call once, as early as the data folder is known. */
export function startLogging(dataDir: string): FileLogger {
  const logger = createFileLogger({
    file: createLogFile({ directory: logsFolder(dataDir) }),
    scrubber: createScrubber({
      homeDir: app.getPath("home"),
      dataDir,
      others: [app.getPath("temp")],
    }),
    appPath: app.getAppPath(),
  });
  const printError = console.error.bind(console);
  logConsole(logger);

  // A monitor, so Electron still handles the error as it would have.
  process.on("uncaughtExceptionMonitor", (error, origin) =>
    logger.exception("main.uncaught", error, { origin }),
  );
  // Electron's main process only warns about these; this logs them whole, and still warns.
  process.on("unhandledRejection", (reason) => {
    logger.exception("main.unhandledRejection", reason);
    printError("Unhandled promise rejection:", reason);
  });
  app.on("render-process-gone", (_event, _contents, details) => {
    const fields = { reason: details.reason, exitCode: details.exitCode };
    if (details.reason === "clean-exit") logger.info("window.exited", fields);
    else logger.error("window.crashed", fields);
  });
  app.on("child-process-gone", (_event, details) => {
    const fields = {
      type: details.type,
      service: details.serviceName,
      reason: details.reason,
      exitCode: details.exitCode,
    };
    if (details.reason === "clean-exit") logger.info("process.exited", fields);
    else logger.error("process.crashed", fields);
  });
  app.on("will-quit", () => logger.info("app.quit"));

  logger.info("app.start", {
    version: app.getVersion(),
    electron: process.versions.electron,
    os: process.platform,
    osVersion: release(),
    arch: arch(),
    packaged: app.isPackaged,
  });
  return logger;
}
