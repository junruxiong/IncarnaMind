import {
  type DragEvent,
  type KeyboardEvent,
  type RefObject,
  useEffect,
  useRef,
  useState,
} from "react";
import { useT } from "../i18n";
import { useMindStatus } from "../mindStatus";
import { useAppStore } from "../store";
import { CloseLineIcon, ExportLineIcon, MindLineIcon, PlusLineIcon } from "./lineIcons";

/*
 * The Mind tabs (DESIGN.md, Components: Mind tabs). This file is their
 * behaviour and structure; their whole look is the `.mind-tabs` block in
 * styles.css, driven by the classes and data-* attributes set here, so the
 * look can change there alone.
 */

const isMac = /Mac|iPhone|iPad/.test(navigator.platform);

/** Drag type for a tab, so file drops and sidebar drags ignore it. */
const TAB_DRAG_TYPE = "application/x-incarnamind-tab";

/** Which side of a tab a dragged one would land on: its left half, or its right. */
const sideOf = (event: DragEvent<HTMLDivElement>): "before" | "after" => {
  const box = event.currentTarget.getBoundingClientRect();
  return event.clientX < box.left + box.width / 2 ? "before" : "after";
};

/**
 * The Mind pane's header: a strip of open Minds as tabs, then "+" for a new
 * Mind, then Export at the right end. Clicking a tab shows its Mind; its ✕ or
 * a middle click closes it; dragging moves it. The arrow keys move between
 * tabs, and the window's shortcuts work too (see `useTabShortcuts`).
 */
export function MindTabs({ onExport }: { onExport(): void }) {
  const t = useT();
  const tabs = useAppStore((state) => state.tabs);
  const openMindId = useAppStore((state) => state.openMindId);
  const openMind = useAppStore((state) => state.openMind);
  const moveTab = useAppStore((state) => state.moveTab);
  const createMind = useAppStore((state) => state.createMind);
  const list = useRef<HTMLDivElement>(null);
  const [dragging, setDragging] = useState<string | null>(null);
  const [dropAt, setDropAt] = useState<{ id: string; side: "before" | "after" } | null>(null);
  useTabShortcuts();

  const overflow = useOverflow(list);

  // The shown tab stays in sight when there are more tabs than room.
  useEffect(() => {
    if (!openMindId) return;
    list.current
      ?.querySelector(`[data-mind-id="${CSS.escape(openMindId)}"]`)
      ?.scrollIntoView({ block: "nearest", inline: "nearest" });
  }, [openMindId]);

  /** Arrow keys, Home and End move to another tab and show it, as in any tab list. */
  const moveFocus = (event: KeyboardEvent<HTMLDivElement>) => {
    const index = openMindId === null ? -1 : tabs.indexOf(openMindId);
    const next =
      event.key === "ArrowRight"
        ? tabs[(index + 1) % tabs.length]
        : event.key === "ArrowLeft"
          ? tabs[(index - 1 + tabs.length) % tabs.length]
          : event.key === "Home"
            ? tabs[0]
            : event.key === "End"
              ? tabs.at(-1)
              : undefined;
    if (next === undefined) return;
    event.preventDefault();
    openMind(next);
    list.current?.querySelector<HTMLElement>(`[data-mind-id="${CSS.escape(next)}"]`)?.focus();
  };

  const drop = (event: DragEvent<HTMLDivElement>, targetId: string) => {
    event.preventDefault();
    const side = sideOf(event);
    if (dragging && dragging !== targetId) {
      const rest = tabs.filter((each) => each !== dragging);
      moveTab(dragging, rest.indexOf(targetId) + (side === "after" ? 1 : 0));
    }
    setDragging(null);
    setDropAt(null);
  };

  return (
    <header data-testid="mind-header" className="mind-tabs">
      <div
        ref={list}
        role="tablist"
        aria-label={t("tabs.label")}
        data-testid="mind-tabs"
        data-hidden-before={overflow.before || undefined}
        data-hidden-after={overflow.after || undefined}
        className="mind-tabs-list"
        onKeyDown={moveFocus}
        // A mouse wheel turns vertically; the strip scrolls sideways.
        onWheel={(event) => {
          if (Math.abs(event.deltaY) > Math.abs(event.deltaX) && list.current) {
            list.current.scrollLeft += event.deltaY;
          }
        }}
      >
        {tabs.map((id, index) => {
          const selected = id === openMindId;
          const next = tabs[index + 1];
          return (
            <MindTab
              key={id}
              mindId={id}
              selected={selected}
              // A divider on a tab's right edge: not on the last, nor next to the shown one.
              divider={!selected && next !== undefined && next !== openMindId}
              dragging={dragging === id}
              dropSide={dropAt?.id === id ? dropAt.side : undefined}
              onDragStart={(event) => {
                event.dataTransfer.setData(TAB_DRAG_TYPE, id);
                event.dataTransfer.effectAllowed = "move";
                setDragging(id);
              }}
              onDragOver={(event) => {
                if (!dragging || !event.dataTransfer.types.includes(TAB_DRAG_TYPE)) return;
                event.preventDefault();
                event.dataTransfer.dropEffect = "move";
                const side = sideOf(event);
                if (dropAt?.id !== id || dropAt.side !== side) setDropAt({ id, side });
              }}
              onDrop={(event) => drop(event, id)}
              onDragEnd={() => {
                setDragging(null);
                setDropAt(null);
              }}
            />
          );
        })}
      </div>
      <button
        type="button"
        data-testid="new-tab"
        aria-label={t("tabs.new", { shortcut: isMac ? "⌘T" : "Ctrl+T" })}
        title={t("tabs.new", { shortcut: isMac ? "⌘T" : "Ctrl+T" })}
        onClick={() => void createMind()}
        className="mind-tabs-new"
      >
        <PlusLineIcon />
      </button>
      <span className="mind-tabs-spacer" />
      {openMindId && (
        <button
          type="button"
          data-testid="export-mind"
          aria-label={t("export.action.label")}
          title={t("export.action.label")}
          onClick={onExport}
          className="mind-tabs-export"
        >
          <ExportLineIcon />
          <span>{t("export.action")}</span>
        </button>
      )}
    </header>
  );
}

interface MindTabProps {
  mindId: string;
  selected: boolean;
  divider: boolean;
  dragging: boolean;
  dropSide: "before" | "after" | undefined;
  onDragStart(event: DragEvent<HTMLDivElement>): void;
  onDragOver(event: DragEvent<HTMLDivElement>): void;
  onDrop(event: DragEvent<HTMLDivElement>): void;
  onDragEnd(): void;
}

/** One tab: the Mind's icon (or what it's busy with), its title, and ✕. */
function MindTab(props: MindTabProps) {
  const { mindId, selected, divider, dragging, dropSide } = props;
  const t = useT();
  const title = useAppStore(
    (state) => state.minds.find((mind) => mind.id === mindId)?.title || t("mind.untitled"),
  );
  const openMind = useAppStore((state) => state.openMind);
  const closeTab = useAppStore((state) => state.closeTab);
  const status = useMindStatus(mindId);

  return (
    <div
      role="tab"
      id={`mind-tab-${mindId}`}
      aria-selected={selected}
      aria-controls={selected ? "mind-tabpanel" : undefined}
      tabIndex={selected ? 0 : -1}
      title={title}
      draggable
      data-testid="mind-tab"
      data-mind-id={mindId}
      data-divider={divider || undefined}
      data-dragging={dragging || undefined}
      data-drop={dropSide}
      onClick={() => openMind(mindId)}
      onKeyDown={(event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          openMind(mindId);
        } else if (event.key === "Delete") {
          event.preventDefault();
          closeTab(mindId);
        }
      }}
      onMouseDown={(event) => {
        if (event.button === 1) event.preventDefault(); // no autoscroll
      }}
      onAuxClick={(event) => {
        if (event.button === 1) closeTab(mindId);
      }}
      onDragStart={props.onDragStart}
      onDragOver={props.onDragOver}
      onDrop={props.onDrop}
      onDragEnd={props.onDragEnd}
      className="mind-tab"
    >
      {/* The shown tab's feet, flaring into the Mind below; the others hide them. */}
      <span aria-hidden="true" data-side="left" className="mind-tab-foot" />
      <span className="mind-tab-inner">
        <span className="mind-tab-icon">
          {status === null ? (
            <MindLineIcon />
          ) : (
            <span
              role="img"
              data-testid="mind-tab-status"
              data-status={status}
              aria-label={status === "writing" ? t("tabs.writing") : t("approvals.answer.waiting")}
              className="mind-tab-dot"
            />
          )}
        </span>
        <span data-testid="mind-tab-title" className="mind-tab-title">
          {title}
        </span>
        <button
          type="button"
          // The tab is the keyboard stop; ⌘W or Delete closes it from there.
          tabIndex={-1}
          data-testid="mind-tab-close"
          aria-label={t("tabs.close", { title })}
          title={t("tabs.close", { title })}
          onClick={(event) => {
            event.stopPropagation();
            closeTab(mindId);
          }}
          className="mind-tab-close"
        >
          <CloseLineIcon />
        </button>
      </span>
      <span aria-hidden="true" data-side="right" className="mind-tab-foot" />
    </div>
  );
}

/**
 * The window's tab shortcuts, as in a browser: ⌘T (Ctrl+T) a new Mind in a
 * new tab, ⌘W closes the tab, ⌘1–⌘8 go to that tab and ⌘9 to the last, and
 * Ctrl+Tab or Ctrl+Shift+Tab move to the next or previous tab. Not while a
 * dialog is open, and not when something else already used the key.
 */
function useTabShortcuts() {
  useEffect(() => {
    const onKey = (event: globalThis.KeyboardEvent) => {
      if (event.defaultPrevented || event.altKey) return;
      if (document.querySelector("dialog[open]")) return;
      const { tabs, openMindId, openMind, closeTab, createMind } = useAppStore.getState();
      const current = openMindId === null ? -1 : tabs.indexOf(openMindId);

      if (event.key === "Tab" && event.ctrlKey && !event.metaKey) {
        event.preventDefault();
        if (tabs.length === 0) return;
        const step = event.shiftKey ? -1 : 1;
        const next = tabs[(current + step + tabs.length) % tabs.length];
        if (next) openMind(next);
        return;
      }

      const command = isMac ? event.metaKey && !event.ctrlKey : event.ctrlKey && !event.metaKey;
      if (!command || event.shiftKey) return;
      if (event.code === "KeyT") {
        event.preventDefault();
        void createMind();
      } else if (event.code === "KeyW") {
        event.preventDefault();
        if (openMindId) closeTab(openMindId);
      } else if (/^Digit[1-9]$/.test(event.code)) {
        event.preventDefault();
        const digit = Number(event.code.slice("Digit".length));
        const target = digit === 9 ? tabs.at(-1) : tabs[digit - 1];
        if (target) openMind(target);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);
}

/**
 * Whether tabs are scrolled out of sight before or after the strip's visible
 * part: the strip fades out on that side (see `.mind-tabs-list`).
 */
function useOverflow(list: RefObject<HTMLDivElement | null>) {
  const [overflow, setOverflow] = useState({ before: false, after: false });
  useEffect(() => {
    const element = list.current;
    if (!element) return;
    const measure = () => {
      const before = element.scrollLeft > 1;
      const after = element.scrollLeft + element.clientWidth < element.scrollWidth - 1;
      setOverflow((current) =>
        current.before === before && current.after === after ? current : { before, after },
      );
    };
    measure();
    element.addEventListener("scroll", measure, { passive: true });
    // The strip's size changes with the window; its width of tabs, as tabs open and close.
    const resized = new ResizeObserver(measure);
    resized.observe(element);
    const changed = new MutationObserver(measure);
    changed.observe(element, { childList: true });
    return () => {
      element.removeEventListener("scroll", measure);
      resized.disconnect();
      changed.disconnect();
    };
  }, [list]);
  return overflow;
}
