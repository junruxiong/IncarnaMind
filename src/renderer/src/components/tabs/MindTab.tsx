import { useContext, useEffect, useState } from "react";
import { files } from "../../core";
import {
  CompareSheetSlotContext,
  ComposerDockContext,
  PageSlotsContext,
} from "../../editor/mindContext";
import { useT } from "../../i18n";
import { assistantTabOf } from "../../layoutSlice";
import { useWritingProperties } from "../../mindStyles";
import { useQuietAssistant } from "../../quietAssistant";
import { useAppStore } from "../../store";
import { useGroupsMerged } from "../../useWindowWidth";
import { isWordFile, useWordImport } from "../../wordImport";
import { ExportResultLine } from "../ExportResultLine";
import { ExampleIntro } from "../FirstOpenSteps";
import { useIsExample } from "../GettingStarted";
import { PlusLineIcon } from "../lineIcons";
import { MakeSkillCard } from "../MakeSkillCard";
import { MeetingInMind } from "../MeetingInMind";
import { MindEditor } from "../MindEditor";
import { ChatReadinessNotice } from "../providers/ChatReadinessNotice";
import { buttonStyle } from "../ui";
import { useTabPlace } from "./tabItems";

/**
 * A Mind's tab: its page, which holds only its text (#908), with the ⋯
 * button at its top right and the count while writing and the check badge at
 * its foot. When it is the Mind the assistant reads, its composer is drawn in
 * the assistant (#253).
 */
export function MindTab({ mindId }: { mindId: string }) {
  const mind = useAppStore((state) => state.minds.find((each) => each.id === mindId));
  const isExample = useIsExample(mindId);
  const [top, setTop] = useState<HTMLElement | null>(null);
  const [menu, setMenu] = useState<HTMLElement | null>(null);
  const [count, setCount] = useState<HTMLElement | null>(null);
  const [badge, setBadge] = useState<HTMLElement | null>(null);
  const [sheetSlot, setSheetSlot] = useState<HTMLElement | null>(null);
  const assistantSlot = useContext(ComposerDockContext);
  const tabId = useTabPlace()?.tabId;
  const merged = useGroupsMerged();
  const read = useAppStore((state) => assistantTabOf(state.layout, merged)?.id === tabId);
  const dropTarget = useWordFileDrop();
  const writing = useWritingProperties(mindId);
  const pill = useQuietAssistant();
  if (!mind) return null;

  return (
    <div className="relative flex min-h-0 flex-grow flex-col">
      <div className="relative flex min-h-0 flex-grow flex-col">
        <div ref={dropTarget} className="mind-scroller min-h-0 flex-grow overflow-auto">
          {/* Every text in it starts at one edge (styles.css, `.mind-column`). */}
          <article
            data-testid="mind-pane"
            data-mind-id={mind.id}
            className="mind-column"
            style={writing}
          >
            <div className="mind-measure">
              {isExample && <ExampleIntro mindId={mindId} />}
              <div ref={setTop} data-testid="mind-top-slot" />
              <MeetingInMind mindId={mind.id} />
              {/* The example works without a chat model: it says so in its Answer instead. */}
              {!isExample && <ChatReadinessNotice />}
              <ComposerDockContext.Provider value={read ? assistantSlot : null}>
                <PageSlotsContext.Provider value={{ top, menu, count, badge }}>
                  <CompareSheetSlotContext.Provider value={sheetSlot}>
                    <MindEditor mindId={mind.id} />
                  </CompareSheetSlotContext.Provider>
                </PageSlotsContext.Provider>
              </ComposerDockContext.Provider>
              <MakeSkillCard mindId={mind.id} />
            </div>
          </article>
          {pill && <div aria-hidden="true" data-testid="ask-pill-fade" className="ask-pill-fade" />}
        </div>
        <div ref={setMenu} className="mind-menu-slot" />
        <div className="mind-foot-slot">
          <div ref={setCount} className="mind-foot-count" />
          <div ref={setBadge} className="mind-foot-badge" />
        </div>
      </div>
      {/* The Compare Documents sheet docks here, in the column's measure, below the note. */}
      <div ref={setSheetSlot} className="compare-sheet-dock" />
      <ExportResultLine mindId={mind.id} />
    </div>
  );
}

/**
 * A Word file dropped on a Mind, alone, opens the import screen (#104), which
 * also offers to add it as a Document, as a drop anywhere else does. It is
 * taken before the window's own drop handling (`useFileDrop`) sees it.
 */
function useWordFileDrop() {
  const [element, setElement] = useState<HTMLElement | null>(null);
  useEffect(() => {
    if (!element) return;
    const drop = (event: DragEvent) => {
      if (event.defaultPrevented || !event.dataTransfer) return;
      const dropped = Array.from(event.dataTransfer.files);
      const file = dropped.length === 1 ? dropped[0] : undefined;
      const path = file && isWordFile(file.name) ? files.pathForFile(file) : "";
      if (!path) return;
      event.preventDefault();
      void useWordImport.getState().open(path, true);
    };
    element.addEventListener("drop", drop);
    return () => element.removeEventListener("drop", drop);
  }, [element]);
  return setElement;
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
        onClick={() => void createMind({ titled: true })}
        className={`mt-3 ${buttonStyle("primary")}`}
      >
        <PlusLineIcon className="size-4" />
        {t("sidebar.newMind")}
      </button>
    </div>
  );
}
