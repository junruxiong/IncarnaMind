import { useCallback, useEffect, useMemo, useRef, useState, useSyncExternalStore } from "react";
import { AppShell, useRememberSidebarWidth } from "./components/AppShell";
import { ConsentDialog } from "./components/ConsentDialog";
import { DropOverlay, useFileDrop } from "./components/FileDrop";
import { MindPane } from "./components/MindPane";
import { ChatSetupDialog } from "./components/providers/ChatSetupDialog";
import { ResizeRod } from "./components/ResizeRod";
import { SettingsDialog } from "./components/SettingsDialog";
import { Sidebar } from "./components/Sidebar";
import { SidebarControlContext } from "./components/SidebarToggle";
import { TagsDialog } from "./components/TagsDialog";
import { UsageDataDialog } from "./components/UsageDataDialog";
import { buttonStyle } from "./components/ui";
import { ViewerPanel } from "./components/ViewerPanel";
import { files } from "./core";
import { useLanguage, useT } from "./i18n";
import { isNarrow, sidebarIsHidden } from "./sidebarLayout";
import { useAppStore } from "./store";
import { useDevViewerShortcut } from "./viewerControls";

// Pane limits, from the old frontend's layout store. The divider inside the card is a 1px rule.
const ROD_WIDTH = 1;
/** The card's distance from the sidebar, and from the window's top, right and bottom edges. */
const CARD_GAP = 8;
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
        const { createMind, closeTab, openMindId, openSettings, libraryOpen, closeLibrary } =
          useAppStore.getState();
        if (command === "toggle-sidebar") return; // The sidebar's own hook handles it.
        if (command === "new-mind") void createMind();
        else if (command === "close-tab") {
          if (libraryOpen) closeLibrary();
          else if (openMindId) closeTab(openMindId);
        } else openSettings();
      }),
    [],
  );
}

const isMac = /Mac|iPhone|iPad/.test(navigator.platform);

/**
 * Hiding and showing the sidebar (#219): Cmd+\ / Ctrl+\ and View > Hide Sidebar
 * (the menu, whose label follows), the title bar's button, dragging the
 * divider shut, and a window too narrow for the viewer. The User's choice
 * is remembered; the narrow window's is not.
 */
function useSidebarControl(
  userHidden: boolean,
  viewerOpen: boolean,
  windowWidth: number,
  sidebarWidth: number,
) {
  const previewLayout = useAppStore((state) => state.previewLayout);
  const updateSettings = useAppStore((state) => state.updateSettings);
  const narrow = isNarrow(viewerOpen, windowWidth, sidebarWidth);
  // The User showed it in a window narrow enough to hide it: it stays until there is room again.
  const [forcedShown, setForcedShown] = useState(false);
  useEffect(() => {
    if (!narrow) setForcedShown(false);
  }, [narrow]);
  const hidden = sidebarIsHidden({ userHidden, narrow, forcedShown });
  const state = useRef({ hidden, narrow, userHidden });
  state.current = { hidden, narrow, userHidden };
  const refocus = useRef(false);

  const setHidden = useCallback(
    (next: boolean) => {
      const current = state.current;
      setForcedShown(!next && current.narrow);
      if (next === current.userHidden) return;
      // Updated here before the core confirms, so the sidebar answers at once.
      previewLayout({ sidebarHidden: next });
      void updateSettings({ device: { sidebarHidden: next } });
    },
    [previewLayout, updateSettings],
  );
  const toggle = useCallback(() => {
    const active = document.activeElement;
    // Keep the focus where it was: on the button the User used, which may move, or in the page.
    refocus.current = active instanceof HTMLElement && active.dataset.testid === "sidebar-toggle";
    if (!state.current.hidden && active instanceof HTMLElement && active.closest("#sidebar")) {
      // Hiding what holds the focus: it goes to the button that shows it again.
      refocus.current = true;
    }
    setHidden(!state.current.hidden);
  }, [setHidden]);
  const takeRefocus = useCallback(() => {
    const wanted = refocus.current;
    refocus.current = false;
    return wanted;
  }, []);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.defaultPrevented || event.altKey || event.shiftKey || event.code !== "Backslash") {
        return;
      }
      const command = isMac ? event.metaKey && !event.ctrlKey : event.ctrlKey && !event.metaKey;
      if (!command || document.querySelector("dialog[open]")) return;
      event.preventDefault();
      toggle();
    };
    window.addEventListener("keydown", onKey);
    const stopMenu = files.onMenuCommand((command) => {
      if (command === "toggle-sidebar" && !document.querySelector("dialog[open]")) toggle();
    });
    return () => {
      window.removeEventListener("keydown", onKey);
      stopMenu();
    };
  }, [toggle]);

  // The View menu says Hide or Show to match.
  useEffect(() => files.setSidebarHidden(hidden), [hidden]);

  return { hidden, setHidden, toggle, takeRefocus };
}

/**
 * While the sidebar is hidden: it slides out over the content when the
 * pointer touches the window's left edge or keyboard focus goes into it, and
 * slides back when the pointer leaves and focus is out.
 */
function useSidebarPeek(hidden: boolean) {
  const [pointer, setPointer] = useState(false);
  const [focus, setFocus] = useState(false);
  useEffect(() => {
    if (!hidden) {
      setPointer(false);
      setFocus(false);
    }
  }, [hidden]);
  const peeking = hidden && (pointer || focus);
  return {
    peeking,
    onEdge: () => setPointer(true),
    panel: {
      onPointerLeave: (event: React.PointerEvent<HTMLElement>) => {
        // A menu or dialog the sidebar opened keeps it out until it is closed.
        if (event.currentTarget.querySelector(":popover-open, dialog[open]")) return;
        setPointer(false);
      },
      // Focus the keyboard moved in (not a click's) shows it, and leaving hides it again.
      onFocus: (event: React.FocusEvent<HTMLElement>) => {
        if (event.target.matches(":focus-visible")) setFocus(true);
      },
      onBlur: (event: React.FocusEvent<HTMLElement>) => {
        if (!event.currentTarget.contains(event.relatedTarget)) setFocus(false);
      },
    },
  };
}

/**
 * Sidebar with Minds and Documents on the left, on the frame, and the open
 * Mind, or the Library in its tab, filling the rest in a card on the sheet, 8px from the sidebar and from
 * the window's other edges. The Document viewer panel appears on the right,
 * inside the card, only while open, narrowing the Mind area.
 * Files dropped anywhere are added as Documents; a folder dropped is offered
 * for linking.
 */
function Workspace() {
  const t = useT();
  const device = useAppStore((state) => state.settings?.device);
  const viewerOpen = useAppStore((state) => state.viewerOpen);
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
  const sidebar = useSidebarControl(
    device?.sidebarHidden ?? false,
    viewerOpen,
    windowWidth,
    device?.sidebarWidth ?? 248,
  );
  const peek = useSidebarPeek(sidebar.hidden);
  const [announcement, setAnnouncement] = useState("");
  const announced = useRef(sidebar.hidden);
  useEffect(() => {
    if (announced.current === sidebar.hidden) return;
    announced.current = sidebar.hidden;
    setAnnouncement(t(sidebar.hidden ? "sidebar.hidden.announce" : "sidebar.shown.announce"));
  }, [sidebar.hidden, t]);
  const control = useMemo(
    () => ({
      hidden: sidebar.hidden,
      peeking: peek.peeking,
      toggle: sidebar.toggle,
      takeRefocus: sidebar.takeRefocus,
    }),
    [sidebar.hidden, peek.peeking, sidebar.toggle, sidebar.takeRefocus],
  );

  if (!device) return null;
  const { sidebarWidth } = device;
  // What the sidebar takes of the window's width: nothing while it is hidden (it peeks over the content).
  const shownSidebarWidth = sidebar.hidden ? 0 : sidebarWidth;
  // What the sidebar and the viewer may take, with the card's gap and margin and the Mind's minimum kept.
  const room = windowWidth - 2 * CARD_GAP - CENTRE_MIN;
  // Until the User resizes it, the viewer opens at half the card (DESIGN.md).
  const preferredViewerWidth =
    device.viewerWidth ??
    Math.min(VIEWER.max, Math.round((windowWidth - shownSidebarWidth - 2 * CARD_GAP) / 2));
  // In a window too narrow for the saved width, the viewer gives way to the
  // Mind (down to its own minimum); the saved width comes back as the window grows.
  const viewerWidth = Math.max(
    VIEWER.min,
    Math.min(preferredViewerWidth, room - ROD_WIDTH - shownSidebarWidth),
  );
  const shownViewerWidth = viewerOpen ? viewerWidth + ROD_WIDTH : 0;

  return (
    <SidebarControlContext.Provider value={control}>
      <div className="relative flex h-screen overflow-hidden bg-frame" {...fileDrop.handlers}>
        {/* The 8px above the card moves the window like the band does. */}
        <div aria-hidden="true" className="title-bar absolute inset-x-0 top-0 h-2" />
        {/* Shown, this wrapper takes no box of its own; hidden, it is the panel that slides out. */}
        <div
          data-testid="sidebar-panel"
          data-hidden={sidebar.hidden || undefined}
          data-open={peek.peeking || undefined}
          className="sidebar-panel"
          style={{ "--sidebar-width": `${sidebarWidth}px` } as React.CSSProperties}
          {...peek.panel}
        >
          <Sidebar width={sidebarWidth} onOpenSettings={openSettings} />
        </div>
        {sidebar.hidden ? (
          <div
            aria-hidden="true"
            data-testid="sidebar-edge"
            className="sidebar-edge"
            onPointerEnter={peek.onEdge}
          />
        ) : (
          <ResizeRod
            label={t("sidebar.resize")}
            width={sidebarWidth}
            min={SIDEBAR.min}
            max={Math.min(SIDEBAR.max, room - shownViewerWidth)}
            direction={1}
            gap
            testId="sidebar-resize"
            onPreview={(width) => previewLayout({ sidebarWidth: width })}
            onCommit={(width) => void updateSettings({ device: { sidebarWidth: width } })}
            onCollapse={() => sidebar.setHidden(true)}
          />
        )}
        <div
          data-testid="card"
          data-sidebar-hidden={sidebar.hidden || undefined}
          className="app-card"
        >
          <MindPane />
          {viewerOpen && (
            <>
              <ResizeRod
                testId="viewer-resize"
                belowBand
                label={t("viewer.resize")}
                width={viewerWidth}
                min={VIEWER.min}
                max={Math.min(VIEWER.max, room - ROD_WIDTH - shownSidebarWidth)}
                direction={-1}
                onPreview={(width) => previewLayout({ viewerWidth: width })}
                onCommit={(width) => void updateSettings({ device: { viewerWidth: width } })}
              />
              <ViewerPanel width={viewerWidth} onClose={closeViewer} />
            </>
          )}
        </div>
        <SettingsDialog />
        <TagsDialog />
        <ChatSetupDialog />
        <UsageDataDialog />
        {/* Opens after the dialog that triggered the request, so it shows on top of it. */}
        <ConsentDialog />
        <ActionError />
        {fileDrop.active && <DropOverlay />}
        <p role="status" data-testid="sidebar-announcement" className="sr-only">
          {announcement}
        </p>
      </div>
    </SidebarControlContext.Provider>
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
