import { type KeyboardEvent, useEffect, useRef, useState } from "react";
import type { Mind } from "../../../core/api";
import { ComposerDockContext } from "../editor/mindContext";
import { useT } from "../i18n";
import { useAppStore } from "../store";
import { ExportDialog } from "./ExportDialog";
import { ExampleBanner, useIsExample } from "./GettingStarted";
import { LibraryPane } from "./LibraryPane";
import { PlusLineIcon } from "./lineIcons";
import { MindEditor } from "./MindEditor";
import { MindTabs } from "./MindTabs";
import { ChatReadinessNotice } from "./providers/ChatReadinessNotice";
import { buttonStyle } from "./ui";

/**
 * The card's centre: a 36px strip of the open Minds as tabs (with "+" and Export),
 * then the shown Mind, its title and its Blocks, and the composer pinned at
 * its foot. With no tab open, a way to start a Mind.
 */
export function MindPane() {
  const t = useT();
  const mind = useAppStore((state) => state.minds.find((each) => each.id === state.openMindId));
  const libraryOpen = useAppStore((state) => state.libraryOpen);
  const createMind = useAppStore((state) => state.createMind);
  const isExample = useIsExample(mind?.id);
  /** The Mind whose export dialog is open: switching to another Mind closes it. */
  const [exportingId, setExportingId] = useState<string | null>(null);
  const [dock, setDock] = useState<HTMLElement | null>(null);

  return (
    <main
      data-testid="mind-area"
      className="flex min-w-[300px] flex-1 flex-col overflow-hidden bg-sheet"
    >
      {/* The card's 36px band: its bottom lines up with the sidebar header's rule. */}
      <MindTabs onExport={() => setExportingId(mind?.id ?? null)} />
      <ExportDialog
        mind={mind && mind.id === exportingId ? mind : null}
        onClose={() => setExportingId(null)}
      />

      {libraryOpen ? (
        // The Library is a tab beside the Minds'; the Mind stays in its own.
        <LibraryPane />
      ) : (
        <div
          className="flex min-h-0 flex-grow flex-col"
          {...(mind && {
            role: "tabpanel",
            id: "mind-tabpanel",
            "aria-labelledby": `mind-tab-${mind.id}`,
          })}
        >
          {mind ? (
            <>
              <div className="min-h-0 flex-grow overflow-auto">
                {/* Keyed, so switching Minds starts a fresh title field and editor. Every text
                  in it starts at one edge (styles.css, `.mind-column`). */}
                <article
                  key={mind.id}
                  data-testid="mind-pane"
                  data-mind-id={mind.id}
                  className="mind-column"
                >
                  <div className="mind-measure">
                    {isExample && <ExampleBanner />}
                    <MindTitle mind={mind} />
                    {/* The example works without a chat model: it says so in its Answer instead. */}
                    {!isExample && <ChatReadinessNotice />}
                    <ComposerDockContext.Provider value={dock}>
                      <MindEditor mindId={mind.id} />
                    </ComposerDockContext.Provider>
                  </div>
                </article>
              </div>
              {/* The composer, pinned at the column's foot: the Mind's editor draws it here. */}
              <div ref={setDock} data-testid="composer-dock" className="composer-dock" />
            </>
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
      )}
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
  const field = useRef<HTMLTextAreaElement>(null);
  const focusNow = useAppStore((state) => state.titleToFocus === mind.id);

  // A Mind just created: its title takes the focus from the button that made it.
  useEffect(() => {
    if (!focusNow) return;
    field.current?.focus();
    useAppStore.getState().titleFocused();
  }, [focusNow]);

  const moveIntoContent = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key !== "Enter" || event.nativeEvent.isComposing) return;
    event.preventDefault();
    event.currentTarget.closest("article")?.querySelector<HTMLElement>(".mind-editor")?.focus();
  };

  return (
    <h1 className="mind-title">
      {/* A text area, so a long title wraps instead of being cut off; it is still one line of text. */}
      <textarea
        ref={field}
        rows={1}
        data-testid="mind-title"
        aria-label={t("mind.title.label")}
        placeholder={t("mind.untitled")}
        value={draft ?? mind.title}
        onFocus={() => setDraft(mind.title)}
        onBlur={() => setDraft(null)}
        onChange={(event) => {
          // A title has no line breaks, even pasted ones.
          const title = event.target.value.replace(/\s*[\r\n]+\s*/g, " ");
          setDraft(title);
          void renameMind(mind.id, title);
        }}
        onKeyDown={moveIntoContent}
      />
    </h1>
  );
}
