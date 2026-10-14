import type { Editor } from "@tiptap/core";
import type { Node as ProseMirrorNode } from "@tiptap/pm/model";
import { TextSelection } from "@tiptap/pm/state";
import { useEditorState } from "@tiptap/react";
import { useState } from "react";
import type { ReferenceCheck } from "../../../core/api";
import { type ListBlock, referenceEntries } from "../../../shared/referenceList";
import { ProblemLine } from "../components/ProblemLine";
import { buttonStyle } from "../components/ui";
import { useLanguage, useT } from "../i18n";
import { turnOffLocalMode } from "../localMode";
import { checkReferences, stopReferenceCheck, useReferenceChecks } from "../referenceChecks";
import { useAppStore } from "../store";

/** The Mind's text Blocks, as the reference list is read from them. */
function blocksOf(doc: ProseMirrorNode): ListBlock[] {
  const blocks: ListBlock[] = [];
  doc.descendants((node) => {
    if (!node.isTextblock) return true;
    const id: unknown = node.attrs.id;
    if (typeof id === "string") {
      blocks.push({ id, heading: node.type.name === "heading", text: node.textContent });
    }
    return false;
  });
  return blocks;
}

const NEEDS_A_LOOK: ReadonlySet<ReferenceCheck["mark"]> = new Set([
  "not-found",
  "mismatch",
  "not-checked",
]);

/** The dot each mark shows, in DESIGN.md's status colours. */
const MARK_DOT: Record<ReferenceCheck["mark"], string> = {
  found: "bg-success",
  mismatch: "bg-attention",
  "not-found": "bg-danger",
  "not-checked": "bg-ink-placeholder",
};

/**
 * The line under a verified document's claims (#711) when its Mind has a
 * reference list: what looking its references up sends, by the button that
 * does it, so clicking it is the consent (ADR-0022); how far a check is,
 * with Stop; then what it found, each reference that needs a look listed
 * below with why. In local mode it says why it can't run, with the way out.
 */
export function ReferencesLine({ editor, mindId }: { editor: Editor; mindId: string }) {
  const t = useT();
  const language = useLanguage();
  const count = useEditorState({
    editor,
    selector: ({ editor: current }) => referenceEntries(blocksOf(current.state.doc)).length,
  });
  const state = useReferenceChecks((current) => current.byMind[mindId]);
  const localOnly = useAppStore((current) => current.embedding?.localOnly ?? false);
  const [open, setOpen] = useState(true);
  if (count === 0 && !state) return null;
  const format = (value: number) => new Intl.NumberFormat(language).format(value);
  const dot = (
    <span className="mind-health-dot" aria-hidden="true">
      ·
    </span>
  );

  /** Moves to the Block a reference is in, as Verify's "need a look" moves to a claim. */
  const show = (blockId: string) => {
    if (editor.isDestroyed) return;
    let at: number | null = null;
    editor.state.doc.descendants((node, pos) => {
      if (at !== null) return false;
      if (node.attrs.id === blockId) at = pos + 1;
      return at === null;
    });
    if (at === null) return;
    const place = editor.state.doc.resolve(at);
    editor.view.dispatch(editor.state.tr.setSelection(TextSelection.near(place)).scrollIntoView());
    editor.view.focus();
  };

  if (state?.kind === "checking") {
    return (
      <div className="mind-verify" data-testid="references-line">
        <span role="status" data-testid="references-progress" className="tabular-nums">
          {t("references.checking", { done: format(state.done), total: format(state.total) })}
        </span>
        {dot}
        <button
          type="button"
          data-testid="references-stop"
          className="mind-verify-link"
          onClick={() => stopReferenceCheck(mindId)}
        >
          {t("checks.stop")}
        </button>
      </div>
    );
  }

  if (state?.kind === "checked") {
    const tally = (mark: ReferenceCheck["mark"]) =>
      state.references.filter((reference) => reference.mark === mark).length;
    const parts = (
      [
        ["not-found", "references.notFound"],
        ["mismatch", "references.mismatch"],
        ["not-checked", "references.notChecked"],
      ] as const
    ).flatMap(([mark, key]) => {
      const n = tally(mark);
      return n === 0 ? [] : [t(n === 1 ? `${key}.one` : key, { count: format(n) })];
    });
    const needLook = state.references.filter((reference) => NEEDS_A_LOOK.has(reference.mark));
    const total = state.references.length;
    return (
      <>
        <div className="mind-verify" data-testid="references-line">
          <span data-testid="references-done" className="mind-verify-done">
            <span>
              {t(total === 1 ? "references.checked.one" : "references.checked", {
                count: format(total),
              })}
            </span>
            {parts.map((part) => (
              <span key={part} className="contents">
                {dot}
                <span>{part}</span>
              </span>
            ))}
          </span>
          {needLook.length > 0 && (
            <>
              {dot}
              <button
                type="button"
                data-testid="references-toggle"
                aria-expanded={open}
                className="mind-verify-link"
                onClick={() => setOpen(!open)}
              >
                {t(open ? "references.hide" : "references.show")}
              </button>
            </>
          )}
          {dot}
          <button
            type="button"
            data-testid="references-again"
            className="mind-verify-link"
            onClick={() => void checkReferences(mindId, count)}
          >
            {t("claim.checkAgain")}
          </button>
        </div>
        {open && needLook.length > 0 && (
          <ul
            data-testid="references-list"
            aria-label={t("references.listLabel")}
            className="mind-references"
          >
            {needLook.map((reference) => (
              <li key={`${reference.blockId}\n${reference.reference}`}>
                <button
                  type="button"
                  data-testid="reference-check"
                  data-mark={reference.mark}
                  className="mind-reference"
                  onClick={() => show(reference.blockId)}
                >
                  <span className="mind-reference-mark">
                    <span
                      aria-hidden="true"
                      className={`size-1.5 shrink-0 rounded-full ${MARK_DOT[reference.mark]}`}
                    />
                    {reference.mark === "not-checked" && reference.reason
                      ? t(`references.reason.${reference.reason}`)
                      : t(`references.mark.${reference.mark}`)}
                  </span>
                  <span className="mind-reference-text">{reference.reference}</span>
                  {reference.differences.map((difference) => (
                    <span key={difference.field} className="mind-reference-detail">
                      {difference.field === "year"
                        ? t("references.difference.year.cited", {
                            cited: difference.cited ?? "",
                            found: difference.found,
                          })
                        : t(
                            difference.field === "title"
                              ? "references.difference.title"
                              : "references.difference.authors",
                            { found: difference.found },
                          )}
                    </span>
                  ))}
                </button>
              </li>
            ))}
          </ul>
        )}
      </>
    );
  }

  if (localOnly) {
    return (
      <ProblemLine
        testId="references-local-mode"
        role="status"
        className="mind-verify-problem"
        action={{
          label: t("references.localMode.turnOff"),
          testId: "references-local-mode-off",
          onClick: () => void turnOffLocalMode(),
        }}
      >
        {t("references.localMode", { count: format(count) })}
      </ProblemLine>
    );
  }

  return (
    <>
      <div className="mind-verify" data-testid="references-line">
        <span data-testid="references-sends">{t("references.ask", { count: format(count) })}</span>
        <button
          type="button"
          data-testid="references-check"
          className={buttonStyle("secondary", "sm")}
          onClick={() => void checkReferences(mindId, count)}
        >
          {t("references.check")}
        </button>
      </div>
      {state?.kind === "error" && (
        <ProblemLine
          testId="references-problem"
          className="mind-verify-problem"
          action={{
            label: t("verify.tryAgain"),
            onClick: () => void checkReferences(mindId, count),
          }}
        >
          {t("references.problem", { message: state.message })}
        </ProblemLine>
      )}
    </>
  );
}
