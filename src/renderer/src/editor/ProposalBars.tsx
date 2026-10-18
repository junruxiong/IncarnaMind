import { autoUpdate, computePosition, flip, offset, shift } from "@floating-ui/dom";
import { type Editor, isMacOS, posToDOMRect } from "@tiptap/core";
import { useEditorState } from "@tiptap/react";
import {
  type MouseEvent as ReactMouseEvent,
  type RefObject,
  useEffect,
  useLayoutEffect,
  useRef,
} from "react";
import { formatDate } from "../../../shared/dates";
import { actOnAppliedEdit } from "../appliedEdits";
import { PencilLineIcon, SparkLineIcon } from "../components/lineIcons";
import { regenerateChange, resolveProposalsIn, speak, useEdits } from "../editWithAi";
import { useLanguage, useT } from "../i18n";
import { useAppStore } from "../store";
import {
  appliedLineIn,
  isAppliedAction,
  redrawAppliedLine,
  toggleAppliedOriginal,
} from "./appliedEditLine";
import { type BarRoom, roomForBar } from "./barRoom";
import {
  changeReason,
  clearHovered,
  editorDoc,
  type PendingProposal,
  pendingAtSelection,
  pendingIn,
  shownProposal,
  sourcesRemoved,
  theirChange,
} from "./proposalReview";
import { redrawProposalRewrites, toggleProposalOriginal } from "./proposalRewrites";

/** How long the bar stays once the pointer leaves its change, so it can reach the bar. */
const LEAVE_MS = 300;

/** At most this much of a change's words is read out. */
const SPOKEN_CHARACTERS = 120;

const spokenText = (text: string) => {
  const flat = text.replace(/￼/g, " ").replace(/\s+/g, " ").trim();
  return flat.length > SPOKEN_CHARACTERS ? `${flat.slice(0, SPOKEN_CHARACTERS - 1)}…` : flat;
};

/**
 * Proposals waiting in the Mind (#93), reviewed in place:
 *
 * - at the top of the Mind, while any wait: "3 changes from the assistant ·
 *   Accept all · Reject all", which stays in view as the Mind scrolls, and a
 *   line of its own for each person's changes from a Word file (#108): "4
 *   changes from Dana Kim · Accept all · Reject all";
 * - over the change the pointer or the cursor is on, its own bar: "Change 1
 *   of 3 · Accept · Reject" (⌘↵, ⌘⌫), with who made someone's change and
 *   when, and when it would remove a Citation, that it does, or when a
 *   Document it cites has been deleted, "Source removed" (#94); why the
 *   assistant made it, as a tooltip, and as a line under the bar when the
 *   cursor is in it (#202);
 * - under a change out of date, instead of its bar, "Out of date ·
 *   Regenerate · Reject" (drawn by `ProposalReview`; its buttons are handled
 *   here), without Regenerate for someone's: it can't be accepted;
 * - for screen readers, the change the cursor moves into, what it does and
 *   the keys that accept and reject it.
 *
 * The words themselves show what each change does: removed words struck,
 * new words underlined (styles.css), never colour alone. The second click
 * of a double click does nothing, so a change is never applied twice, nor
 * the next one that slid under the pointer.
 */
export function ProposalBars({ editor, mindId }: { editor: Editor; mindId: string }) {
  // "Regenerate" and "Reject" under a change out of date, in the note's text.
  useEffect(() => {
    const dom = editor.view.dom;
    const buttonOf = (event: Event) =>
      event.target instanceof Element
        ? event.target.closest<HTMLElement>(
            "[data-proposal-regenerate], [data-proposal-reject], [data-applied-action], [data-change-original]",
          )
        : null;
    // The text keeps the focus and its selection.
    const press = (event: MouseEvent) => {
      if (buttonOf(event)) event.preventDefault();
    };
    const click = (event: MouseEvent) => {
      const button = buttonOf(event);
      if (!button || repeated(event)) return;
      event.preventDefault();
      const original = button.dataset.changeOriginal;
      if (original) {
        if (original.startsWith("applied:")) toggleAppliedOriginal(editor, original);
        else toggleProposalOriginal(editor, original);
        return;
      }
      const applied = button.dataset.appliedAction;
      if (applied && isAppliedAction(applied)) {
        actOnAppliedEdit(editor, applied);
        if (!editor.isDestroyed) editor.commands.focus();
        return;
      }
      const rejected = button.dataset.proposalReject;
      if (rejected) {
        resolveProposalsIn(editor, mindId, [rejected], "reject");
        if (!editor.isDestroyed) editor.commands.focus();
        return;
      }
      const proposal = pendingIn(editor.state).find(
        (each) => each.id === button.dataset.proposalRegenerate,
      );
      const doc = editorDoc(editor);
      if (proposal && doc) void regenerateChange(editor, mindId, doc, proposal);
    };
    dom.addEventListener("mousedown", press);
    dom.addEventListener("click", click);
    return () => {
      dom.removeEventListener("mousedown", press);
      dom.removeEventListener("click", click);
    };
  }, [editor, mindId]);
  // The applied edit's lines and the "Was: …" lines are drawn outside React: again once the language changes.
  const language = useLanguage();
  useEffect(() => {
    redrawAppliedLine(editor, language);
    redrawProposalRewrites(editor);
  }, [editor, language]);
  return (
    <>
      <ProposalsBar editor={editor} mindId={mindId} />
      <ChangeBar editor={editor} mindId={mindId} />
      <Spoken editor={editor} mindId={mindId} />
    </>
  );
}

function barRoom(editor: Editor): BarRoom {
  const card = editor.view.dom.closest(".app-card")?.getBoundingClientRect() ?? null;
  const dock = document.querySelector<HTMLElement>("[data-assistant]")?.dataset.dock;
  return roomForBar(card, dock, { width: window.innerWidth, height: window.innerHeight });
}

/** The second click of a double click: it does nothing more. */
const repeated = (event: MouseEvent) => event.detail > 1;

/** Changes waiting at the top of the Mind, in a line each: each person's from the file, and the assistant's. */
interface Group {
  /** Who made them, for someone's changes from a Word file (null when the file names no one); the assistant's: undefined. */
  author?: string | null;
  ids: string[];
  stale: boolean;
}

function ProposalsBar({ editor, mindId }: { editor: Editor; mindId: string }) {
  const t = useT();
  const groups = useEditorState({
    editor,
    selector: ({ editor: current }): Group[] => {
      const theirs = new Map<string | null, Group>();
      const assistant: Group = { ids: [], stale: false };
      const line = appliedLineIn(current.state);
      // An edit held back has its own line.
      const ownLine = line?.kind === "held" ? line.request : null;
      for (const each of pendingIn(current.state)) {
        if (each.request === ownLine) continue;
        let group = assistant;
        if (each.theirs) {
          const author = theirChange(current, each)?.author ?? null;
          group = theirs.get(author) ?? { author, ids: [], stale: false };
          theirs.set(author, group);
        }
        group.ids.push(each.id);
        group.stale ||= each.outOfDate;
      }
      return [...theirs.values(), ...(assistant.ids.length > 0 ? [assistant] : [])];
    },
  });
  if (groups.length === 0) return null;
  return (
    // biome-ignore lint/a11y/useSemanticElements: a region of the Mind, not a page landmark.
    <div
      role="region"
      aria-label={t(
        groups.some((group) => group.author !== undefined)
          ? "proposal.region.review"
          : "proposal.region",
      )}
      data-testid="proposals-bar"
      className="proposals-bar"
    >
      {groups.map((group) => (
        <GroupLine
          key={group.author === undefined ? "\u0000assistant" : (group.author ?? "\u0000unknown")}
          group={group}
          every={groups.length === 1}
          editor={editor}
          mindId={mindId}
        />
      ))}
    </div>
  );
}

/**
 * One line of the bar: how many changes, whose, and Accept all · Reject all
 * for them; for every change the Mind holds as it acts, when the line has
 * them all (`every`), those an Undo has just brought back too.
 */
function GroupLine({
  group,
  every,
  editor,
  mindId,
}: {
  group: Group;
  every: boolean;
  editor: Editor;
  mindId: string;
}) {
  const t = useT();
  const theirs = group.author !== undefined;
  const name = group.author ?? t("proposal.unknownAuthor");
  const prefix = theirs ? "their-changes" : "proposals";
  const count = group.ids.length;
  const resolveAll = (outcome: "accept" | "reject") => {
    resolveProposalsIn(editor, mindId, every ? "all" : group.ids, outcome);
  };
  return (
    <div className="proposals-bar-line" data-testid={`${prefix}-line`}>
      {theirs ? (
        <PencilLineIcon className="proposals-bar-icon" />
      ) : (
        <SparkLineIcon className="proposals-bar-icon" />
      )}
      <span data-testid={`${prefix}-count`} className="min-w-0 truncate">
        {theirs
          ? count === 1
            ? t("proposal.theirs.count.one", { name })
            : t("proposal.theirs.count", { count, name })
          : count === 1
            ? t("proposal.count.one")
            : t("proposal.count", { count })}
      </span>
      <span aria-hidden="true" className="proposal-dot">
        ·
      </span>
      <button
        type="button"
        data-testid={`${prefix}-accept-all`}
        title={group.stale ? t("proposal.acceptAll.outOfDate") : undefined}
        aria-label={theirs ? t("proposal.theirs.acceptAll", { name }) : undefined}
        onClick={(event) => {
          if (!repeated(event.nativeEvent)) resolveAll("accept");
        }}
        className="proposal-action"
      >
        {t("proposal.acceptAll")}
      </button>
      <span aria-hidden="true" className="proposal-dot">
        ·
      </span>
      <button
        type="button"
        data-testid={`${prefix}-reject-all`}
        aria-label={theirs ? t("proposal.theirs.rejectAll", { name }) : undefined}
        onClick={(event) => {
          if (!repeated(event.nativeEvent)) resolveAll("reject");
        }}
        className="proposal-action"
      >
        {t("proposal.rejectAll")}
      </button>
    </div>
  );
}

/** Who made someone's change and when, as the file says: "Dana Kim, 2 Oct 2026". */
function useMadeBy(made: { author: string | null; date: string | null } | null): string | null {
  const t = useT();
  const language = useLanguage();
  if (!made) return null;
  const author = made.author ?? t("proposal.unknownAuthor");
  const date = made.date ? new Date(made.date) : null;
  if (!date || Number.isNaN(date.getTime())) return author;
  // The day the file names, not the day it falls on here: Word writes its own clock's time.
  return t("proposal.madeBy", {
    author,
    date: formatDate(date, language, { timeZone: "UTC" }),
  });
}

interface Shown {
  id: string;
  index: number;
  count: number;
  from: number;
  to: number;
  dropped: number;
  citations: PendingProposal["citations"];
  /** Who made someone's change, and when; null for the assistant's. */
  made: { author: string | null; date: string | null } | null;
  /** Why the assistant made it; null when it didn't say. */
  reason: string | null;
  /** The cursor is in it, not only the pointer on it. */
  focused: boolean;
}

/** A change's bar sits over the first line of its change, following it as the Mind scrolls. */
export function useBarOver(
  editor: Editor,
  bar: RefObject<HTMLDivElement | null>,
  change: { from: number; to: number } | null,
): void {
  useLayoutEffect(() => {
    const element = bar.current;
    if (!change || !element || editor.isDestroyed) return;
    const reference = {
      getBoundingClientRect: () => posToDOMRect(editor.view, change.from, change.to),
      contextElement: editor.view.dom,
    };
    const place = () => {
      if (editor.isDestroyed) return;
      // Never over the Mind's bar of changes, which stays at the top as the Mind scrolls, nor an
      // applied edit's line above the change: under the change's line then.
      const covered = editor.view.dom
        .closest(".mind-editor-frame")
        ?.querySelector(".proposals-bar")
        ?.getBoundingClientRect().bottom;
      const changeTop = reference.getBoundingClientRect().top;
      const lines = [...editor.view.dom.querySelectorAll(".applied-line")]
        .map((line) => line.getBoundingClientRect().bottom)
        .filter((bottom) => bottom <= changeTop + 1);
      const top = Math.max(8, covered ?? 0, ...lines);
      void computePosition(reference, element, {
        placement: "top-start",
        strategy: "fixed",
        middleware: [
          offset(6),
          flip({ padding: { top } }),
          shift({ padding: 8, rootBoundary: barRoom(editor) }),
        ],
      }).then(({ x, y }) => {
        Object.assign(element.style, { left: `${x}px`, top: `${y}px`, visibility: "visible" });
      });
    };
    return autoUpdate(reference, element, place);
  }, [editor, bar, change]);
}

function ChangeBar({ editor, mindId }: { editor: Editor; mindId: string }) {
  const t = useT();
  const bar = useRef<HTMLDivElement>(null);
  const over = useRef(false);
  const shown = useEditorState({
    editor,
    selector: ({ editor: current }): Shown | null => {
      const proposal = shownProposal(current.state);
      // One out of date has its line under it instead.
      if (!proposal || proposal.outOfDate) return null;
      const all = pendingIn(current.state);
      const first = proposal.ranges[0] ?? proposal;
      return {
        id: proposal.id,
        index: all.findIndex((each) => each.id === proposal.id) + 1,
        count: all.length,
        from: first.from,
        to: first.to,
        dropped: proposal.droppedCitations,
        citations: proposal.citations,
        made: proposal.theirs ? theirChange(current, proposal) : null,
        reason: proposal.theirs ? null : changeReason(current, proposal),
        focused: pendingAtSelection(current.state).some((each) => each.id === proposal.id),
      };
    },
  });
  const madeBy = useMadeBy(shown?.made ?? null);
  const documents = useAppStore((state) =>
    state.status.kind === "ready" ? state.documents : null,
  );
  const kept = useAppStore((state) => state.keptCitationTexts);

  // The pointer leaving a change takes its bar away, unless it went onto the bar.
  useEffect(() => {
    const dom = editor.view.dom;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const leave = () => {
      clearTimeout(timer);
      timer = setTimeout(() => {
        if (!over.current) clearHovered(editor);
      }, LEAVE_MS);
    };
    const move = (event: MouseEvent) => {
      const target = event.target instanceof Element ? event.target : null;
      if (target?.closest("[data-proposal-id]")) clearTimeout(timer);
      else leave();
    };
    dom.addEventListener("mousemove", move);
    dom.addEventListener("mouseleave", leave);
    return () => {
      clearTimeout(timer);
      dom.removeEventListener("mousemove", move);
      dom.removeEventListener("mouseleave", leave);
    };
  }, [editor]);

  useBarOver(editor, bar, shown);

  if (!shown) return null;
  const removedSources = sourcesRemoved(shown, documents, kept);
  const resolve = (event: ReactMouseEvent, outcome: "accept" | "reject") => {
    if (repeated(event.nativeEvent)) return;
    over.current = false;
    resolveProposalsIn(editor, mindId, [shown.id], outcome);
    if (!editor.isDestroyed) editor.commands.focus();
  };
  const keys = isMacOS()
    ? { accept: "⌘↩", reject: "⌘⌫" }
    : { accept: "Ctrl+Enter", reject: "Ctrl+Backspace" };
  const why = shown.reason ? t("proposal.why", { reason: shown.reason }) : null;
  return (
    // biome-ignore lint/a11y/useSemanticElements: a toolbar of one change's actions, beside the text.
    <div
      ref={bar}
      role="group"
      aria-label={t("proposal.position", { index: shown.index, count: shown.count })}
      data-testid="change-bar"
      data-proposal-id={shown.id}
      onMouseEnter={() => {
        over.current = true;
      }}
      onMouseLeave={() => {
        over.current = false;
        clearHovered(editor);
      }}
      // The text keeps the focus and its selection.
      onMouseDown={(event) => event.preventDefault()}
      title={why ?? undefined}
      className="change-bar"
      style={{ visibility: "hidden" }}
    >
      <div className="change-bar-row">
        <span className="change-bar-position">
          {t("proposal.position", { index: shown.index, count: shown.count })}
        </span>
        {madeBy && (
          <span data-testid="change-bar-author" className="change-bar-author">
            {madeBy}
          </span>
        )}
        {shown.dropped > 0 && (
          <span data-testid="change-bar-drops" className="change-bar-drops">
            {shown.dropped === 1
              ? t("proposal.dropsCitations.one")
              : t("proposal.dropsCitations", { count: shown.dropped })}
          </span>
        )}
        {removedSources > 0 && (
          <span
            data-testid="change-bar-source-removed"
            title={t("proposal.sourceRemoved.why")}
            className="change-bar-drops"
          >
            {t("proposal.sourceRemoved")}
          </span>
        )}
        <button
          type="button"
          data-testid="change-accept"
          title={`${t("proposal.accept")} (${keys.accept})`}
          aria-keyshortcuts={isMacOS() ? "Meta+Enter" : "Control+Enter"}
          onClick={(event) => resolve(event, "accept")}
          className="proposal-action"
        >
          {t("proposal.accept")}
        </button>
        <span aria-hidden="true" className="proposal-dot">
          ·
        </span>
        <button
          type="button"
          data-testid="change-reject"
          title={`${t("proposal.reject")} (${keys.reject})`}
          aria-keyshortcuts={isMacOS() ? "Meta+Backspace" : "Control+Backspace"}
          onClick={(event) => resolve(event, "reject")}
          className="proposal-action"
        >
          {t("proposal.reject")}
        </button>
      </div>
      {why && shown.focused && (
        <div data-testid="change-bar-why" className="change-bar-why">
          {why}
        </div>
      )}
    </div>
  );
}

/** What a change does, in words, for a screen reader; whose, for someone's (`author`). */
function describe(
  proposal: PendingProposal,
  t: ReturnType<typeof useT>,
  author: string | null,
): string {
  const removed = spokenText(proposal.removed);
  const added = spokenText(proposal.added);
  const what =
    removed && added
      ? t("proposal.announce.replace", { removed, added })
      : added
        ? t("proposal.announce.add", { added })
        : t("proposal.announce.remove", { removed });
  const platform = isMacOS() ? "mac" : "other";
  const accept = t(`proposal.keys.accept.${platform}`);
  const reject = t(`proposal.keys.reject.${platform}`);
  if (author !== null) {
    return proposal.outOfDate
      ? t("proposal.announce.theirs.outOfDate", { author, what, reject })
      : t("proposal.announce.theirs", { author, what, accept, reject });
  }
  if (proposal.outOfDate) return t("proposal.announce.outOfDate", { what, reject });
  return t("proposal.announce", { what, accept, reject });
}

/** The Mind's polite live region: the change the cursor moves into, and what was accepted or rejected. */
function Spoken({ editor, mindId }: { editor: Editor; mindId: string }) {
  const t = useT();
  const spoken = useEdits((state) => state.spoken[mindId]);
  const focused = useEditorState({
    editor,
    selector: ({ editor: current }) => pendingAtSelection(current.state)[0]?.id ?? null,
  });
  useEffect(() => {
    if (!focused || editor.isDestroyed) return;
    const proposal = pendingIn(editor.state).find((each) => each.id === focused);
    if (!proposal) return;
    const made = proposal.theirs ? theirChange(editor, proposal) : null;
    speak(
      mindId,
      describe(proposal, t, made ? (made.author ?? t("proposal.unknownAuthor")) : null),
    );
  }, [focused, editor, mindId, t]);
  return (
    <div role="status" aria-live="polite" data-testid="proposal-spoken" className="sr-only">
      {spoken?.text}
    </div>
  );
}
