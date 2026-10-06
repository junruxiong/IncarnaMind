import { useEffect, useState, useSyncExternalStore } from "react";
import { MindPane } from "./components/MindPane";
import { ResizeRod } from "./components/ResizeRod";
import { SettingsDialog } from "./components/SettingsDialog";
import { Sidebar } from "./components/Sidebar";
import { ViewerPanel } from "./components/ViewerPanel";
import { useLanguage, useT } from "./i18n";
import { useAppStore } from "./store";
import { useDevViewerShortcut } from "./viewerControls";

// Pane limits, from the old frontend's layout store.
const ROD_WIDTH = 3;
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

  if (status.kind === "loading") return null;
  if (status.kind === "failed") {
    return (
      <p role="alert" className="p-6 text-sm text-red-700">
        {t("error.load", { message: status.message })}
      </p>
    );
  }
  return <Workspace />;
}

/**
 * Sidebar with Minds on the left and the open Mind filling the rest. The
 * Document viewer panel appears on the right only while open, narrowing the Mind area.
 */
function Workspace() {
  const t = useT();
  const device = useAppStore((state) => state.settings?.device);
  const viewerOpen = useAppStore((state) => state.viewerOpen);
  const closeViewer = useAppStore((state) => state.closeViewer);
  const previewLayout = useAppStore((state) => state.previewLayout);
  const updateSettings = useAppStore((state) => state.updateSettings);
  const windowWidth = useWindowWidth();
  const [settingsOpen, setSettingsOpen] = useState(false);
  useDevViewerShortcut();

  if (!device) return null;
  const { sidebarWidth, viewerWidth } = device;
  const shownViewerWidth = viewerOpen ? viewerWidth + ROD_WIDTH : 0;
  const room = windowWidth - ROD_WIDTH - CENTRE_MIN;

  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar width={sidebarWidth} onOpenSettings={() => setSettingsOpen(true)} />
      <ResizeRod
        label={t("sidebar.resize")}
        width={sidebarWidth}
        min={SIDEBAR.min}
        max={Math.min(SIDEBAR.max, room - shownViewerWidth)}
        direction={1}
        onPreview={(width) => previewLayout({ sidebarWidth: width })}
        onCommit={(width) => void updateSettings({ device: { sidebarWidth: width } })}
      />
      <MindPane />
      {viewerOpen && (
        <>
          <ResizeRod
            testId="viewer-resize"
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
      <SettingsDialog open={settingsOpen} onClose={() => setSettingsOpen(false)} />
      <ActionError />
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
      className="fixed right-4 bottom-4 flex max-w-sm items-start gap-3 rounded-[9px] bg-white px-4 py-3 text-sm text-red-700 shadow-custom-focus"
    >
      <p>{t("error.action", { message })}</p>
      <button
        type="button"
        onClick={dismiss}
        className="shrink-0 rounded-[9px] px-2 text-gray-500 hover:bg-gray-100 hover:text-gray-700"
      >
        {t("error.dismiss")}
      </button>
    </div>
  );
}
