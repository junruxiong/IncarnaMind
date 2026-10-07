import { useState } from "react";
import type { Mind } from "../../../core/api";
import { useApprovals } from "../approvals";
import { useT } from "../i18n";
import { useAppStore } from "../store";
import { DeleteMindDialog } from "./DeleteMindDialog";
import { DocumentsSection } from "./DocumentsSection";
import { AppMark, MindLineIcon, PlusLineIcon, SettingsLineIcon, TrashLineIcon } from "./lineIcons";
import { SidebarStatus } from "./SidebarStatus";
import {
  rowActionButtonClass,
  rowActionsClass,
  rowButtonClass,
  rowClass,
  rowIconClass,
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
        className="hide-scrollbar flex min-h-0 flex-1 flex-col overflow-y-auto px-2 pt-2 pb-3"
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

        <SectionLabel id="minds-heading">{t("sidebar.minds")}</SectionLabel>
        <nav aria-labelledby="minds-heading">
          {minds.length === 0 ? (
            <p className="px-2 py-1 text-[13px] leading-5 text-ink-meta">{t("sidebar.noMinds")}</p>
          ) : (
            <ul>
              {minds.map((mind) => (
                <MindRow key={mind.id} mind={mind} onDelete={() => setConfirmingDelete(mind)} />
              ))}
            </ul>
          )}
        </nav>
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
  const waiting = useApprovals((state) =>
    Object.values(state.waiting).some((request) => request.mindId === mind.id),
  );
  return (
    <li className={rowClass(isOpen)}>
      <button
        type="button"
        data-testid="mind-list-item"
        data-mind-id={mind.id}
        aria-current={isOpen ? "page" : undefined}
        onClick={() => openMind(mind.id)}
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
