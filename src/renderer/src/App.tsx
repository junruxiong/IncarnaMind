import { useEffect, useSyncExternalStore } from "react";
import { AppShell, useRememberSidebarWidth } from "./components/AppShell";
import { ConsentDialog } from "./components/ConsentDialog";
import { DropOverlay, useFileDrop } from "./components/FileDrop";
import { LibraryPane } from "./components/LibraryPane";
import { MindPane } from "./components/MindPane";
import { ChatSetupDialog } from "./components/providers/ChatSetupDialog";
import { ResizeRod } from "./components/ResizeRod";
import { SettingsDialog } from "./components/SettingsDialog";
import { Sidebar } from "./components/Sidebar";
import { TagsDialog } from "./components/TagsDialog";
import { buttonStyle } from "./components/ui";
import { ViewerPanel } from "./components/ViewerPanel";
import { files } from "./core";
import { useLanguage, useT } from "./i18n";
import { useAppStore } from "./store";
import { useDevViewerShortcut } from "./viewerControls";

// Pane limits, from the old frontend's layout store. A divider is a 1px rule.
const ROD_WIDTH = 1;
const SIDEBAR = { min: 165, max: 480 };
const VIEWER = { min: 220, max: 900 };
const CENTRE_MIN = 300;

const subscribeToResize = (onChange: () => void) => {
  window.addEventListener("resize", onChange);
  return () => window.removeEventListener("resize", onChange);
};
const useWindowWidth = () => useSyncExternalStore(subscribeToResize, () => window.innerWidth);

export function App() {
  const status = useAppStore((state) => state.status);
  const load = useAppStore((state) => state.load);
  const language = useLanguage();
  const t = useT();

  useEffect(() => {
    void load();
  }, [load]);

  useEffect(() => {
    document.documentElement.lang = language;
  }, [language]);

  // The frame shows at once; the data fills it in as it arrives.
  if (status.kind === "loading") return <AppShell />;
  if (status.kind === "failed") {
    return (
      <p role="alert" className="p-6 text-ui text-danger">
        {t("error.load", { message: status.message })}
      </p>
    );
  }
  return <Workspace />;
}

/**
 * What the application menu's items do (src/main/menu.ts): New Mind, Close
 * Tab and Settings…. Nothing while a dialog is open, as for the tab shortcuts.
 */
function useMenuCommands(): void {
  useEffect(
    () =>
      files.onMenuCommand((command) => {
        if (document.querySelector("dialog[open]")) return;
        const { createMind, closeTab, openMindId, openSettings } = useAppStore.getState();
        if (command === "new-mind") void createMind();
        else if (command === "close-tab") {
          if (openMindId) closeTab(openMindId);
        } else openSettings();
      }),
    [],
  );
}

/**
 * Sidebar with Minds and Documents on the left, on the frame, and the open
 * Mind filling the rest, on the sheet; 1px rules divide them. The Document
 * viewer panel appears on the right only while open, narrowing the Mind area.
 * Files dropped anywhere are added as Documents; a folder dropped is offered
 * for linking.
 */
function Workspace() {
  const t = useT();
  const device = useAppStore((state) => state.settings?.device);
  const viewerOpen = useAppStore((state) => state.viewerOpen);
  const libraryOpen = useAppStore((state) => state.libraryOpen);
  const closeViewer = useAppStore((state) => state.closeViewer);
  const previewLayout = useAppStore((state) => state.previewLayout);
  const updateSettings = useAppStore((state) => state.updateSettings);
  const addDropped = useAppStore((state) => state.addDropped);
  const fileDrop = useFileDrop((files, folders) => void addDropped(files, folders));
  const windowWidth = useWindowWidth();
  const openSettings = useAppStore((state) => state.openSettings);
  useDevViewerShortcut();
  useMenuCommands();
  useRememberSidebarWidth(device?.sidebarWidth);

  if (!device) return null;
  const { sidebarWidth } = device;
  const room = windowWidth - ROD_WIDTH - CENTRE_MIN;
  // Until the User resizes it, the viewer opens at half the room beside the sidebar (DESIGN.md).
  const preferredViewerWidth =
    device.viewerWidth ?? Math.min(VIEWER.max, Math.round((windowWidth - sidebarWidth) / 2));
  // In a window too narrow for the saved width, the viewer gives way to the
  // Mind (down to its own minimum); the saved width comes back as the window grows.
  const viewerWidth = Math.max(
    VIEWER.min,
    Math.min(preferredViewerWidth, room - ROD_WIDTH - sidebarWidth),
  );
  const shownViewerWidth = viewerOpen ? viewerWidth + ROD_WIDTH : 0;

  return (
    <div className="relative flex h-screen overflow-hidden bg-sheet" {...fileDrop.handlers}>
      <Sidebar width={sidebarWidth} onOpenSettings={openSettings} />
      <ResizeRod
        label={t("sidebar.resize")}
        width={sidebarWidth}
        min={SIDEBAR.min}
        max={Math.min(SIDEBAR.max, room - shownViewerWidth)}
        direction={1}
        onPreview={(width) => previewLayout({ sidebarWidth: width })}
        onCommit={(width) => void updateSettings({ device: { sidebarWidth: width } })}
      />
      {libraryOpen ? <LibraryPane /> : <MindPane />}
      {viewerOpen && (
        <>
          <ResizeRod
            testId="viewer-resize"
            belowBand
            label={t("viewer.resize")}
            width={viewerWidth}
            min={VIEWER.min}
            max={Math.min(VIEWER.max, room - ROD_WIDTH - sidebarWidth)}
            direction={-1}
            onPreview={(width) => previewLayout({ viewerWidth: width })}
            onCommit={(width) => void updateSettings({ device: { viewerWidth: width } })}
          />
          <ViewerPanel width={viewerWidth} onClose={closeViewer} />
        </>
      )}
      <SettingsDialog />
      <TagsDialog />
      <ChatSetupDialog />
      {/* Opens after the dialog that triggered the request, so it shows on top of it. */}
      <ConsentDialog />
      <ActionError />
      {fileDrop.active && <DropOverlay />}
    </div>
  );
}

function ActionError() {
  const t = useT();
  const message = useAppStore((state) => state.actionError);
  const dismiss = useAppStore((state) => state.dismissActionError);
  if (!message) return null;
  return (
    <div
      role="alert"
      className="fixed right-4 bottom-4 z-20 flex max-w-sm items-start gap-3 rounded-lg bg-sheet py-2.5 pr-2.5 pl-4 text-[13px] leading-5 text-danger shadow-popover"
    >
      <p className="min-w-0 flex-1 py-1 break-words">{t("error.action", { message })}</p>
      <button type="button" onClick={dismiss} className={buttonStyle("ghost", "sm")}>
        {t("error.dismiss")}
      </button>
    </div>
  );
}
