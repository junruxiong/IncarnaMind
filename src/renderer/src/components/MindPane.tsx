import { type KeyboardEvent, useState } from "react";
import type { Mind } from "../../../core/api";
import { useT } from "../i18n";
import { useAppStore } from "../store";
import { ExportDialog } from "./ExportDialog";
import { PlusLineIcon } from "./lineIcons";
import { MindEditor } from "./MindEditor";
import { MindTabs } from "./MindTabs";
import { ChatReadinessNotice } from "./providers/ChatReadinessNotice";
import { buttonStyle } from "./ui";

/**
 * The centre: a 44px strip of the open Minds as tabs (with "+" and Export),
 * then the shown Mind, its title and its Blocks. With no tab open, a way to
 * start a Mind.
 */
export function MindPane() {
  const t = useT();
  const mind = useAppStore((state) => state.minds.find((each) => each.id === state.openMindId));
  const createMind = useAppStore((state) => state.createMind);
  /** The Mind whose export dialog is open: switching to another Mind closes it. */
  const [exportingId, setExportingId] = useState<string | null>(null);

  return (
    <main
      data-testid="mind-area"
      className="flex min-w-[300px] flex-1 flex-col overflow-hidden bg-sheet"
    >
      {/* 44px, like every pane header, so it lines up with the sidebar's. */}
      <MindTabs onExport={() => setExportingId(mind?.id ?? null)} />
      <ExportDialog
        mind={mind && mind.id === exportingId ? mind : null}
        onClose={() => setExportingId(null)}
      />

      <div
        className="min-h-0 flex-grow overflow-auto"
        {...(mind && {
          role: "tabpanel",
          id: "mind-tabpanel",
          "aria-labelledby": `mind-tab-${mind.id}`,
        })}
      >
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
          <div
            data-testid="mind-none-open"
            className="flex h-full flex-col items-center justify-center gap-2 px-10 pb-11 text-center"
          >
            <h1 className="font-serif text-heading font-semibold text-ink">
              {t("mind.noneOpen.title")}
            </h1>
            <p className="max-w-sm text-ui text-ink-secondary">{t("mind.noneOpen.body")}</p>
            <button
              type="button"
              onClick={() => void createMind()}
              className={`mt-3 ${buttonStyle("primary")}`}
            >
              <PlusLineIcon className="size-4" />
              {t("sidebar.newMind")}
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
