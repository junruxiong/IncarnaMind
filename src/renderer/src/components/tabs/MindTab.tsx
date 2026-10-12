import { type KeyboardEvent, useContext, useEffect, useRef, useState } from "react";
import type { Mind } from "../../../../core/api";
import {
  CompareSheetSlotContext,
  ComposerDockContext,
  CountSlotContext,
} from "../../editor/mindContext";
import { useT } from "../../i18n";
import { assistantTabOf } from "../../layoutSlice";
import { useAppStore } from "../../store";
import { useGroupsMerged } from "../../useWindowWidth";
import { ExportResultLine } from "../ExportResultLine";
import { ExampleBanner, useIsExample } from "../GettingStarted";
import { PlusLineIcon } from "../lineIcons";
import { MakeSkillCard } from "../MakeSkillCard";
import { MeetingInMind } from "../MeetingInMind";
import { MindEditor } from "../MindEditor";
import { ChatReadinessNotice } from "../providers/ChatReadinessNotice";
import { TypingTextarea } from "../TypingTextarea";
import { buttonStyle } from "../ui";
import { useTabPlace } from "./tabItems";

/**
 * A Mind's tab: its title and its Blocks. When it is the Mind the assistant
 * reads, its composer is drawn in the assistant (#253).
 */
export function MindTab({ mindId }: { mindId: string }) {
  const mind = useAppStore((state) => state.minds.find((each) => each.id === mindId));
  const isExample = useIsExample(mindId);
  const [countSlot, setCountSlot] = useState<HTMLElement | null>(null);
  const [sheetSlot, setSheetSlot] = useState<HTMLElement | null>(null);
  const assistantSlot = useContext(ComposerDockContext);
  const tabId = useTabPlace()?.tabId;
  const merged = useGroupsMerged();
  const read = useAppStore((state) => assistantTabOf(state.layout, merged)?.id === tabId);
  if (!mind) return null;

  return (
    <div className="relative flex min-h-0 flex-grow flex-col">
      <div className="mind-scroller min-h-0 flex-grow overflow-auto">
        {/* Every text in it starts at one edge (styles.css, `.mind-column`). */}
        <article data-testid="mind-pane" data-mind-id={mind.id} className="mind-column">
          <div className="mind-measure">
            {isExample && <ExampleBanner />}
            <MindTitle mind={mind} />
            <div ref={setCountSlot} data-testid="mind-count-slot" />
            <MeetingInMind mindId={mind.id} />
            {/* The example works without a chat model: it says so in its Answer instead. */}
            {!isExample && <ChatReadinessNotice />}
            <ComposerDockContext.Provider value={read ? assistantSlot : null}>
              <CountSlotContext.Provider value={countSlot}>
                <CompareSheetSlotContext.Provider value={sheetSlot}>
                  <MindEditor mindId={mind.id} />
                </CompareSheetSlotContext.Provider>
              </CountSlotContext.Provider>
            </ComposerDockContext.Provider>
            <MakeSkillCard mindId={mind.id} />
          </div>
        </article>
      </div>
      {/* The Compare Documents sheet docks here, in the column's measure, below the note. */}
      <div ref={setSheetSlot} className="compare-sheet-dock" />
      <ExportResultLine mindId={mind.id} />
    </div>
  );
}

/** A window with no tab open: a way to start a Mind. */
export function NoTabOpen() {
  const t = useT();
  const createMind = useAppStore((state) => state.createMind);
  return (
    <div
      data-testid="mind-none-open"
      className="flex h-full flex-col items-center justify-center gap-2 px-10 pb-11 text-center"
    >
      <h1 className="font-serif text-heading font-semibold text-ink">{t("mind.noneOpen.title")}</h1>
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
  /** The title as it was when the User started typing it, for Undo. */
  const titleBefore = useRef(mind.title);
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
    const title = event.currentTarget;
    const article = title.closest("article");
    const content = () => article?.querySelector<HTMLElement>(".mind-editor");
    const now = content();
    if (now || !article) {
      now?.focus();
      return;
    }
    // A new Mind's text comes a moment after its title (its document is still opening):
    // the cursor goes there once it does, unless it has left the title by then.
    const observer = new MutationObserver(() => {
      const editor = content();
      if (!editor) return;
      observer.disconnect();
      if (document.activeElement === title) editor.focus();
    });
    observer.observe(article, { childList: true, subtree: true });
    title.addEventListener("blur", () => observer.disconnect(), { once: true });
  };

  return (
    <h1 className="mind-title">
      {/* A text area, so a long title wraps instead of being cut off; it is still one line of text. */}
      <TypingTextarea
        ref={field}
        rows={1}
        data-testid="mind-title"
        aria-label={t("mind.title.label")}
        placeholder={mind.autoTitle || t("mind.untitled")}
        value={draft ?? mind.title}
        onFocus={() => {
          setDraft(mind.title);
          titleBefore.current = mind.title;
        }}
        onBlur={() => {
          setDraft(null);
          useAppStore.getState().titleLeft(mind.id, titleBefore.current, draft ?? mind.title);
        }}
        onChange={(event) => {
          // A title has no line breaks, even pasted ones.
          const title = event.target.value.replace(/\s*[\r\n]+\s*/g, " ");
          setDraft(title);
          void renameMind(mind.id, title, { typing: true });
        }}
        onKeyDown={moveIntoContent}
      />
    </h1>
  );
}
