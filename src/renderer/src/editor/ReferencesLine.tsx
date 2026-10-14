import type { Editor } from "@tiptap/core";
import type { Node as ProseMirrorNode } from "@tiptap/pm/model";
import { TextSelection } from "@tiptap/pm/state";
import { useEditorState } from "@tiptap/react";
import { useState } from "react";
import type { ReferenceCheckReason } from "../../../core/api";
import { checkableReferences, type ListBlock } from "../../../shared/referenceList";
import { ProblemLine } from "../components/ProblemLine";
import { buttonStyle } from "../components/ui";
import { useLanguage, useT } from "../i18n";
import { turnOffLocalMode } from "../localMode";
import { checkReferences, stopReferenceCheck, useReferenceChecks } from "../referenceChecks";
import { useAppStore } from "../store";
import { NEEDS_A_LOOK, ReferenceCheckList, SERVICE_NAMES } from "./ReferenceCheckList";

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

/** A part of the line with the dot before it, so a wrapped line never ends on a dot. */
const AFTER_DOT = "inline-flex items-center gap-2";

/** Why a check stopped before asking about the rest: offline, or asked to slow down. */
const STOPPED: ReadonlySet<ReferenceCheckReason | null> = new Set(["unreachable", "rate-limited"]);

/**
 * The line under a verified document's claims (#711) when its Mind has a
 * reference list: what looking its references up sends, by the button that
 * does it, so clicking it is the consent (ADR-0022); how far a check is,
 * with Stop; then what it found, its references listed below
 * (ReferenceCheckList), or, offline or slowed down, one problem with Try
 * again. In local mode it says why it can't run, with the way out.
 */
export function ReferencesLine({ editor, mindId }: { editor: Editor; mindId: string }) {
  const t = useT();
  const language = useLanguage();
  const count = useEditorState({
    editor,
    selector: ({ editor: current }) => checkableReferences(blocksOf(current.state.doc)).length,
  });
  const state = useReferenceChecks((current) => current.byMind[mindId]);
  const localOnly = useAppStore((current) => current.embedding?.localOnly ?? false);
  const [shown, setShown] = useState<boolean | null>(null);
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

  const looked =
    state?.kind === "checked"
      ? state.references.filter((reference) => !STOPPED.has(reference.reason))
      : [];
  // Offline or slowed down, the entries a service wasn't asked about are one problem, not a row each.
  const stoppedBy =
    state?.kind === "checked"
      ? [
          ...new Map(
            state.references
              .filter((reference) => STOPPED.has(reference.reason) && reference.service)
              .map((reference) => [reference.service, reference.reason] as const),
          ),
        ]
      : [];
  const stopped = stoppedBy.map(([service, reason]) => (
    <ProblemLine
      key={service}
      testId="references-stopped"
      className="mind-verify-problem"
      action={{
        label: t("verify.tryAgain"),
        testId: "references-try-again",
        onClick: () => void checkReferences(mindId, count),
      }}
    >
      {t(
        reason === "rate-limited"
          ? "references.stopped.rate-limited"
          : "references.stopped.unreachable",
        { service: service ? SERVICE_NAMES[service] : "" },
      )}
    </ProblemLine>
  ));

  if (state?.kind === "checked" && looked.length > 0) {
    const total = state.references.length;
    const parts = (
      [
        ["not-found", "references.notFound"],
        ["not-listed", "references.notListed"],
        ["mismatch", "references.mismatch"],
        ["not-checked", "references.notChecked"],
      ] as const
    ).flatMap(([mark, key]) => {
      const n = looked.filter((reference) => reference.mark === mark).length;
      return n === 0 ? [] : [t(n === 1 ? `${key}.one` : key, { count: format(n) })];
    });
    const open = shown ?? looked.some((reference) => NEEDS_A_LOOK.has(reference.mark));
    return (
      <>
        <div className="mind-verify" data-testid="references-line">
          <span data-testid="references-done" className="mind-verify-done">
            <span>
              {looked.length < total
                ? t("references.checkedOf", { done: format(looked.length), total: format(total) })
                : t(total === 1 ? "references.checked.one" : "references.checked", {
                    count: format(total),
                  })}
            </span>
            {parts.map((part) => (
              <span key={part} className={AFTER_DOT}>
                {dot}
                <span>{part}</span>
              </span>
            ))}
          </span>
          <span className={AFTER_DOT}>
            {dot}
            <button
              type="button"
              data-testid="references-toggle"
              aria-expanded={open}
              className="mind-verify-link"
              onClick={() => setShown(!open)}
            >
              {t(open ? "references.hide" : "references.show")}
            </button>
          </span>
          <span className={AFTER_DOT}>
            {dot}
            <button
              type="button"
              data-testid="references-again"
              className="mind-verify-link"
              onClick={() => void checkReferences(mindId, count)}
            >
              {t("claim.checkAgain")}
            </button>
          </span>
        </div>
        {stopped}
        {open && <ReferenceCheckList references={looked} onShow={show} />}
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
      {stopped}
    </>
  );
}
