import { autoUpdate, computePosition, flip, offset, shift } from "@floating-ui/dom";
import { type Editor, isMacOS, posToDOMRect } from "@tiptap/core";
import { useEditorState } from "@tiptap/react";
import { useEffect, useLayoutEffect, useRef } from "react";
import { SparkLineIcon } from "../components/lineIcons";
import { resolveProposalsIn, speak, useEdits } from "../editWithAi";
import { useT } from "../i18n";
import {
  clearHovered,
  type PendingProposal,
  pendingAtSelection,
  pendingIn,
  shownProposal,
} from "./proposalReview";

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
 *   Accept all · Reject all", which stays in view as the Mind scrolls;
 * - over the change the pointer or the cursor is on, its own bar: "Change 1
 *   of 3 · Accept · Reject" (⌘↵, ⌘⌫), and when it would remove a Citation,
 *   that it does;
 * - for screen readers, the change the cursor moves into, what it does and
 *   the keys that accept and reject it.
 *
 * The words themselves show what each change does: removed words struck,
 * new words underlined (styles.css), never colour alone.
 */
export function ProposalBars({ editor, mindId }: { editor: Editor; mindId: string }) {
  return (
    <>
      <ProposalsBar editor={editor} mindId={mindId} />
      <ChangeBar editor={editor} mindId={mindId} />
      <Spoken editor={editor} mindId={mindId} />
    </>
  );
}

function ProposalsBar({ editor, mindId }: { editor: Editor; mindId: string }) {
  const t = useT();
  const ids = useEditorState({
    editor,
    selector: ({ editor: current }) => pendingIn(current.state).map((each) => each.id),
  });
  if (ids.length === 0) return null;
  const resolveAll = (outcome: "accept" | "reject") => {
    resolveProposalsIn(
      editor,
      mindId,
      pendingIn(editor.state).map((each) => each.id),
      outcome,
    );
  };
  return (
    // biome-ignore lint/a11y/useSemanticElements: a region of the Mind, not a page landmark.
    <div
      role="region"
      aria-label={t("proposal.region")}
      data-testid="proposals-bar"
      className="proposals-bar"
    >
      <SparkLineIcon className="proposals-bar-icon" />
      <span data-testid="proposals-count" className="min-w-0 truncate">
        {ids.length === 1 ? t("proposal.count.one") : t("proposal.count", { count: ids.length })}
      </span>
      <span aria-hidden="true" className="proposal-dot">
        ·
      </span>
      <button
        type="button"
        data-testid="proposals-accept-all"
        onClick={() => resolveAll("accept")}
        className="proposal-action"
      >
        {t("proposal.acceptAll")}
      </button>
      <span aria-hidden="true" className="proposal-dot">
        ·
      </span>
      <button
        type="button"
        data-testid="proposals-reject-all"
        onClick={() => resolveAll("reject")}
        className="proposal-action"
      >
        {t("proposal.rejectAll")}
      </button>
    </div>
  );
}

interface Shown {
  id: string;
  index: number;
  count: number;
  from: number;
  to: number;
  dropped: number;
}

function ChangeBar({ editor, mindId }: { editor: Editor; mindId: string }) {
  const t = useT();
  const bar = useRef<HTMLDivElement>(null);
  const over = useRef(false);
  const shown = useEditorState({
    editor,
    selector: ({ editor: current }): Shown | null => {
      const proposal = shownProposal(current.state);
      if (!proposal) return null;
      const all = pendingIn(current.state);
      const first = proposal.ranges[0] ?? proposal;
      return {
        id: proposal.id,
        index: all.findIndex((each) => each.id === proposal.id) + 1,
        count: all.length,
        from: first.from,
        to: first.to,
        dropped: proposal.droppedCitations,
      };
    },
  });

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

  // Sits over the first line of its change, following it as the Mind scrolls.
  useLayoutEffect(() => {
    const element = bar.current;
    if (!shown || !element || editor.isDestroyed) return;
    const reference = {
      getBoundingClientRect: () => posToDOMRect(editor.view, shown.from, shown.to),
      contextElement: editor.view.dom,
    };
    const place = () => {
      if (editor.isDestroyed) return;
      // Never over the Mind's bar of changes, which stays at the top as the Mind scrolls: under the line then.
      const covered = editor.view.dom
        .closest(".mind-editor-frame")
        ?.querySelector(".proposals-bar")
        ?.getBoundingClientRect().bottom;
      const top = Math.max(8, covered ?? 0);
      void computePosition(reference, element, {
        placement: "top-start",
        strategy: "fixed",
        middleware: [offset(6), flip({ padding: { top } }), shift({ padding: 8 })],
      }).then(({ x, y }) => {
        Object.assign(element.style, { left: `${x}px`, top: `${y}px`, visibility: "visible" });
      });
    };
    return autoUpdate(reference, element, place);
  }, [editor, shown]);

  if (!shown) return null;
  const resolve = (outcome: "accept" | "reject") => {
    over.current = false;
    resolveProposalsIn(editor, mindId, [shown.id], outcome);
    if (!editor.isDestroyed) editor.commands.focus();
  };
  const keys = isMacOS()
    ? { accept: "⌘↩", reject: "⌘⌫" }
    : { accept: "Ctrl+Enter", reject: "Ctrl+Backspace" };
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
      className="change-bar"
      style={{ visibility: "hidden" }}
    >
      <span className="change-bar-position">
        {t("proposal.position", { index: shown.index, count: shown.count })}
      </span>
      {shown.dropped > 0 && (
        <span data-testid="change-bar-drops" className="change-bar-drops">
          {shown.dropped === 1
            ? t("proposal.dropsCitations.one")
            : t("proposal.dropsCitations", { count: shown.dropped })}
        </span>
      )}
      <button
        type="button"
        data-testid="change-accept"
        title={`${t("proposal.accept")} (${keys.accept})`}
        aria-keyshortcuts={isMacOS() ? "Meta+Enter" : "Control+Enter"}
        onClick={() => resolve("accept")}
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
        onClick={() => resolve("reject")}
        className="proposal-action"
      >
        {t("proposal.reject")}
      </button>
    </div>
  );
}

/** What a change does, in words, for a screen reader. */
function describe(proposal: PendingProposal, t: ReturnType<typeof useT>): string {
  const removed = spokenText(proposal.removed);
  const added = spokenText(proposal.added);
  const what =
    removed && added
      ? t("proposal.announce.replace", { removed, added })
      : added
        ? t("proposal.announce.add", { added })
        : t("proposal.announce.remove", { removed });
  const platform = isMacOS() ? "mac" : "other";
  return t("proposal.announce", {
    what,
    accept: t(`proposal.keys.accept.${platform}`),
    reject: t(`proposal.keys.reject.${platform}`),
  });
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
    if (proposal) speak(mindId, describe(proposal, t));
  }, [focused, editor, mindId, t]);
  return (
    <div role="status" aria-live="polite" data-testid="proposal-spoken" className="sr-only">
      {spoken?.text}
    </div>
  );
}
