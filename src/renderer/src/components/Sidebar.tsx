import { useEffect, useRef, useState } from "react";
import type { Mind } from "../../../core/api";
import { useT } from "../i18n";
import { useMindStatus } from "../mindStatus";
import { useAppStore } from "../store";
import { DeleteMindDialog } from "./DeleteMindDialog";
import { DocumentsSection } from "./DocumentsSection";
import {
  AppMark,
  MindLineIcon,
  PencilLineIcon,
  PlusLineIcon,
  SettingsLineIcon,
  TrashLineIcon,
} from "./lineIcons";
import { SidebarStatus } from "./SidebarStatus";
import {
  rowActionButtonClass,
  rowActionsClass,
  rowButtonClass,
  rowClass,
  rowIconClass,
  rowInputClass,
  SectionLabel,
} from "./sidebarRows";

/**
 * The sidebar, on the frame: a 44px header like every pane's, then "New
 * Mind", the Minds and the Documents in 28px rows, and a footer with the
 * tagging and processing status above Settings. Every row puts its icon at
 * x 16 and its text at x 40; section labels share the icon column.
 */
export function Sidebar({ width, onOpenSettings }: { width: number; onOpenSettings(): void }) {
  const t = useT();
  const minds = useAppStore((state) => state.minds);
  const createMind = useAppStore((state) => state.createMind);
  const [confirmingDelete, setConfirmingDelete] = useState<Mind | null>(null);
  const [mindsFolded, setMindsFolded] = useRememberedFold("sidebar.minds.folded");

  return (
    <aside
      aria-label={t("sidebar.label")}
      data-testid="sidebar"
      className="flex min-w-[165px] shrink flex-col bg-frame"
      style={{ flexBasis: width }}
    >
      <header
        data-testid="sidebar-header"
        className="flex h-11 shrink-0 items-center gap-2 border-b border-rule px-4"
      >
        <AppMark />
        <span className="truncate text-ui font-semibold text-ink">{t("app.name")}</span>
      </header>

      <div
        data-testid="sidebar-tree"
        className="thin-scrollbar flex min-h-0 flex-1 flex-col overflow-y-auto px-2 pt-2 pb-3"
      >
        <div className={rowClass(false)}>
          <button
            type="button"
            data-testid="new-mind"
            onClick={() => void createMind()}
            className={rowButtonClass}
          >
            <PlusLineIcon className="size-4 shrink-0" />
            <span className="truncate">{t("sidebar.newMind")}</span>
          </button>
        </div>

        <SectionLabel
          id="minds-heading"
          folded={mindsFolded}
          onToggle={() => setMindsFolded(!mindsFolded)}
          toggleLabel={mindsFolded ? t("sidebar.minds.unfold") : t("sidebar.minds.fold")}
        >
          {mindsFolded && minds.length > 0
            ? t("sidebar.minds.count", { count: minds.length })
            : t("sidebar.minds")}
        </SectionLabel>
        {!mindsFolded && (
          <nav aria-labelledby="minds-heading">
            {minds.length === 0 ? (
              <p className="px-2 py-1 text-[13px] leading-5 text-ink-meta">
                {t("sidebar.noMinds")}
              </p>
            ) : (
              <ul>
                {minds.map((mind) => (
                  <MindRow key={mind.id} mind={mind} onDelete={() => setConfirmingDelete(mind)} />
                ))}
              </ul>
            )}
          </nav>
        )}
        <DocumentsSection />
      </div>

      <footer
        data-testid="sidebar-footer"
        className="flex shrink-0 flex-col border-t border-rule p-2"
      >
        <SidebarStatus />
        <div className={rowClass(false)}>
          <button type="button" onClick={onOpenSettings} className={rowButtonClass}>
            <SettingsLineIcon className={rowIconClass(false)} />
            <span className="truncate">{t("sidebar.settings")}</span>
          </button>
        </div>
      </footer>
      <DeleteMindDialog mind={confirmingDelete} onClose={() => setConfirmingDelete(null)} />
    </aside>
  );
}

/**
 * A Mind: its title, and an amber dot while one of its Answers waits for the
 * User's approval. Pointed at, it offers delete.
 */
function MindRow({ mind, onDelete }: { mind: Mind; onDelete(): void }) {
  const t = useT();
  const isOpen = useAppStore((state) => state.openMindId === mind.id);
  const openMind = useAppStore((state) => state.openMind);
  const waiting = useMindStatus(mind.id) === "waiting-for-approval";
  const [renaming, setRenaming] = useState(false);
  if (renaming) {
    return (
      <li className={rowClass(isOpen)}>
        <span className="flex h-full w-full min-w-0 items-center gap-2 pr-1 pl-2">
          <MindLineIcon className={rowIconClass(isOpen)} />
          <RenameMind mind={mind} onDone={() => setRenaming(false)} />
        </span>
      </li>
    );
  }
  return (
    <li className={rowClass(isOpen)}>
      <button
        type="button"
        data-testid="mind-list-item"
        data-mind-id={mind.id}
        aria-current={isOpen ? "page" : undefined}
        // ⌘-click (Ctrl-click) or a middle click opens it in a new tab, as in a browser.
        onClick={(event) => openMind(mind.id, { newTab: event.metaKey || event.ctrlKey })}
        onMouseDown={(event) => {
          if (event.button === 1) event.preventDefault(); // no autoscroll
        }}
        onAuxClick={(event) => {
          if (event.button === 1) openMind(mind.id, { newTab: true });
        }}
        onDoubleClick={() => setRenaming(true)}
        className={rowButtonClass}
      >
        <MindLineIcon className={rowIconClass(isOpen)} />
        <span
          data-testid="row-text"
          className={`min-w-0 flex-1 truncate ${mind.title ? "" : "text-ink-meta"}`}
        >
          {mind.title || t("mind.untitled")}
        </span>
        {waiting && (
          <span
            role="img"
            data-testid="mind-approval"
            aria-label={t("approvals.answer.waiting")}
            title={t("approvals.answer.waiting")}
            className="size-1.5 shrink-0 rounded-full bg-attention"
          />
        )}
      </button>
      <div className={rowActionsClass}>
        <button
          type="button"
          data-testid="rename-mind"
          aria-label={t("mind.rename")}
          title={t("mind.rename")}
          onClick={() => setRenaming(true)}
          className={rowActionButtonClass}
        >
          <PencilLineIcon className="size-[15px]" />
        </button>
        <button
          type="button"
          data-testid="delete-mind"
          aria-label={t("mind.delete")}
          title={t("mind.delete")}
          onClick={onDelete}
          className={rowActionButtonClass}
        >
          <TrashLineIcon className="size-[15px]" />
        </button>
      </div>
    </li>
  );
}

/** A Mind's title typed in its row: Enter or leaving the field saves, Esc cancels. */
function RenameMind({ mind, onDone }: { mind: Mind; onDone(): void }) {
  const t = useT();
  const renameMind = useAppStore((state) => state.renameMind);
  const [value, setValue] = useState(mind.title);
  const field = useRef<HTMLInputElement>(null);
  const finished = useRef(false);

  useEffect(() => {
    field.current?.focus();
    field.current?.select();
  }, []);

  const finish = (save: boolean) => {
    if (finished.current) return;
    finished.current = true;
    if (save && value.trim() !== mind.title) void renameMind(mind.id, value);
    onDone();
  };

  return (
    <input
      ref={field}
      value={value}
      data-testid="mind-rename"
      aria-label={t("mind.renameLabel", { title: mind.title || t("mind.untitled") })}
      placeholder={t("mind.untitled")}
      onChange={(event) => setValue(event.target.value)}
      onKeyDown={(event) => {
        if (event.nativeEvent.isComposing) return;
        if (event.key === "Enter" || event.key === "Escape") {
          event.preventDefault();
          finish(event.key === "Enter");
        }
      }}
      onBlur={() => finish(true)}
      className={rowInputClass}
    />
  );
}

/** A section folded away or not, remembered on this computer. */
function useRememberedFold(key: string): [boolean, (folded: boolean) => void] {
  const [folded, setFolded] = useState(() => {
    try {
      return localStorage.getItem(`incarnamind.${key}`) === "true";
    } catch {
      return false;
    }
  });
  const set = (next: boolean) => {
    setFolded(next);
    try {
      localStorage.setItem(`incarnamind.${key}`, String(next));
    } catch {
      // Not remembered, then: it unfolds again next time.
    }
  };
  return [folded, set];
}
