import { type Editor, isMacOS } from "@tiptap/core";
import { useEditorState } from "@tiptap/react";
import { useEffect, useRef, useState } from "react";
import { restoreChange, reviewedChangeAt } from "../appliedEdits";
import { useT } from "../i18n";
import { appliedLineIn, placeOfChange } from "./appliedEditLine";
import { useBarOver } from "./ProposalBars";

/** How long the bar stays once the pointer leaves its change, so it can reach the bar. */
const LEAVE_MS = 300;
/** How long the pointer rests on a change before the bar moves to it, so passing over one on the way doesn't. */
const REST_MS = 150;

/** "Change 1 of 3 · Restore original" over an applied edit's change shown again (#880). */
export function AppliedReviewBar({ editor }: { editor: Editor }) {
  const t = useT();
  const bar = useRef<HTMLDivElement>(null);
  const over = useRef(false);
  const [hovered, setHovered] = useState<number | null>(null);
  const shown = useEditorState({
    editor,
    selector: ({ editor: current }) => {
      const line = appliedLineIn(current.state);
      if (line?.kind !== "review") return null;
      const index = hovered ?? reviewedChangeAt(current.state);
      const change = index === null ? undefined : line.changes[index];
      const place = change ? placeOfChange(current.state, change) : null;
      return index !== null && place
        ? { index, count: line.changes.length, from: place.from, to: place.to }
        : null;
    },
  });

  useEffect(() => {
    const dom = editor.view.dom;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const move = (event: MouseEvent) => {
      const target = event.target instanceof Element ? event.target : null;
      const index = target?.closest("[data-applied-change]")?.getAttribute("data-applied-change");
      clearTimeout(timer);
      if (over.current) return;
      timer = index
        ? setTimeout(() => setHovered(Number(index)), REST_MS)
        : setTimeout(() => !over.current && setHovered(null), LEAVE_MS);
    };
    dom.addEventListener("mousemove", move);
    return () => {
      clearTimeout(timer);
      dom.removeEventListener("mousemove", move);
    };
  }, [editor]);

  useBarOver(editor, bar, shown);

  if (!shown) return null;
  const keys = isMacOS() ? "⌘⌫" : "Ctrl+Backspace";
  return (
    // biome-ignore lint/a11y/useSemanticElements: a toolbar of one change's actions, beside the text.
    <div
      ref={bar}
      role="group"
      aria-label={t("proposal.position", { index: shown.index + 1, count: shown.count })}
      data-testid="applied-change-bar"
      onMouseEnter={() => {
        over.current = true;
      }}
      onMouseLeave={() => {
        over.current = false;
        setHovered(null);
      }}
      // The text keeps the focus and its selection.
      onMouseDown={(event) => event.preventDefault()}
      className="change-bar"
      style={{ visibility: "hidden" }}
    >
      <div className="change-bar-row">
        <span className="change-bar-position">
          {t("proposal.position", { index: shown.index + 1, count: shown.count })}
        </span>
        <button
          type="button"
          data-testid="change-restore"
          title={`${t("edit.review.restore")} (${keys})`}
          aria-keyshortcuts={isMacOS() ? "Meta+Backspace" : "Control+Backspace"}
          onClick={(event) => {
            if (event.detail > 1) return;
            over.current = false;
            setHovered(null);
            restoreChange(editor, shown.index);
            if (!editor.isDestroyed) editor.commands.focus();
          }}
          className="proposal-action"
        >
          {t("edit.review.restore")}
        </button>
      </div>
    </div>
  );
}
