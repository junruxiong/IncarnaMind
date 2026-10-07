import { type KeyboardEvent, useState } from "react";
import type { Mind } from "../../../core/api";
import { useT } from "../i18n";
import { useAppStore } from "../store";
import { ExportDialog } from "./ExportDialog";
import { ExportIcon, MindIcon, PlusIcon } from "./icons";
import { MindEditor } from "./MindEditor";
import { ChatReadinessNotice } from "./providers/ChatReadinessNotice";

/** The centre: the open Mind, its title and its Blocks. */
export function MindPane() {
  const t = useT();
  const mind = useAppStore((state) => state.minds.find((each) => each.id === state.openMindId));
  const createMind = useAppStore((state) => state.createMind);
  /** The Mind whose export dialog is open: switching to another Mind closes it. */
  const [exportingId, setExportingId] = useState<string | null>(null);
  const title = mind ? mind.title || t("mind.untitled") : "";

  return (
    <main data-testid="mind-area" className="flex min-w-[300px] flex-1 flex-col overflow-hidden">
      <div className="flex h-10 shrink-0 items-end">
        {mind && (
          <>
            <div
              title={title}
              className="relative ml-2 flex h-8 max-w-[220px] items-center rounded-t-[9px] bg-white pr-4 pl-8 text-sm text-gray-700"
            >
              <MindIcon className="absolute left-[10px] size-4" />
              <span className="truncate">{title}</span>
            </div>
            <button
              type="button"
              data-testid="export-mind"
              aria-label={t("export.action.label")}
              title={t("export.action.label")}
              onClick={() => setExportingId(mind.id)}
              className="mr-3 mb-1 ml-auto flex items-center gap-1 rounded-[9px] px-2 py-1 text-sm text-gray-600 hover:bg-gray-200 hover:text-gray-800"
            >
              <ExportIcon className="size-4" />
              <span>{t("export.action")}</span>
            </button>
          </>
        )}
      </div>
      <ExportDialog
        mind={mind && mind.id === exportingId ? mind : null}
        onClose={() => setExportingId(null)}
      />

      <div className="flex-grow overflow-auto rounded-tl-[6px] bg-white">
        {mind ? (
          // Keyed, so switching Minds starts a fresh title field and editor. Every text
          // in it starts at one edge (styles.css, `.mind-column`).
          <article
            key={mind.id}
            data-testid="mind-pane"
            data-mind-id={mind.id}
            className="mind-column"
          >
            <div className="mind-measure">
              <MindTitle mind={mind} />
              <ChatReadinessNotice />
              <MindEditor mindId={mind.id} />
            </div>
          </article>
        ) : (
          <div className="flex h-full flex-col items-center justify-center gap-2 px-10 text-center">
            <h1 className="text-xl font-medium text-gray-700">{t("mind.noneOpen.title")}</h1>
            <p className="text-sm text-gray-500">{t("mind.noneOpen.body")}</p>
            <button
              type="button"
              onClick={() => void createMind()}
              className="group mt-2 flex items-center gap-2 rounded-[9px] border border-gray-300 px-4 py-2 text-sm text-gray-700 hover:shadow-custom-unfocus"
            >
              <PlusIcon className="size-4 text-gray-500" />
              <span className="group-hover:text-gradient-mind">{t("sidebar.newMind")}</span>
            </button>
          </div>
        )}
      </div>
    </main>
  );
}

/**
 * The Mind's title, edited in place. Every change is saved as it is typed; while
 * the field has focus it shows what was typed, not the saved (trimmed) title.
 */
function MindTitle({ mind }: { mind: Mind }) {
  const t = useT();
  const renameMind = useAppStore((state) => state.renameMind);
  const [draft, setDraft] = useState<string | null>(null);

  const moveIntoContent = (event: KeyboardEvent<HTMLInputElement>) => {
    if (event.key !== "Enter" || event.nativeEvent.isComposing) return;
    event.preventDefault();
    event.currentTarget.closest("article")?.querySelector<HTMLElement>(".mind-editor")?.focus();
  };

  return (
    <h1 className="mind-title">
      <input
        data-testid="mind-title"
        aria-label={t("mind.title.label")}
        placeholder={t("mind.untitled")}
        value={draft ?? mind.title}
        onFocus={() => setDraft(mind.title)}
        onBlur={() => setDraft(null)}
        onChange={(event) => {
          setDraft(event.target.value);
          void renameMind(mind.id, event.target.value);
        }}
        onKeyDown={moveIntoContent}
      />
    </h1>
  );
}
