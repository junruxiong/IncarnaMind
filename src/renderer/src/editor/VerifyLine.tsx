import type { Editor } from "@tiptap/core";
import { TextSelection } from "@tiptap/pm/state";
import { useEditorState } from "@tiptap/react";
import { type ReactNode, useEffect, useLayoutEffect, useRef, useState } from "react";
import type * as Y from "yjs";
import {
  observeVerification,
  type Verification,
  verificationOf,
} from "../../../shared/verification";
import { DocumentLineIcon } from "../components/lineIcons";
import { ProblemLine } from "../components/ProblemLine";
import { ReadinessExplanation, readinessAction } from "../components/providers/ChatReadinessNotice";
import { UnreadFilesLine } from "../components/UnreadFilesLine";
import { buttonStyle } from "../components/ui";
import { core } from "../core";
import { estimateText, runStopText } from "../cost";
import { useLanguage, useT } from "../i18n";
import { enter } from "../motion";
import { useAppStore } from "../store";
import { useCheckUnread } from "../unreadFiles";
import {
  checkClaims,
  dismissVerifyOutcome,
  findClaims,
  openDocumentBeside,
  useVerify,
} from "../verify";
import { AnswerError } from "./AnswerView";
import { coverageOf, sameCoverage, shownClaims } from "./claimChecks";
import { ClaimMarkIcon, type ClaimMarkState } from "./icons";
import { useMindModel, useRunModel } from "./ModelChip";
import { Counted } from "./NoteHealth";
import { UncoveredLine } from "./UncoveredLine";

/** What a Mind made by "Verify a document…" verifies, as it changes. */
export function useVerification(doc: Y.Doc): Verification | null {
  const [verification, setVerification] = useState(() => verificationOf(doc));
  useEffect(() => {
    setVerification(verificationOf(doc));
    return observeVerification(doc, () => setVerification(verificationOf(doc)));
  }, [doc]);
  return verification;
}

type VerifyRow = "contradicted" | "unsupported" | "notChecked" | "recheck" | "supported";

/** The checks of the document's sentences: how many there are and are going, and where each state's are. */
function claimMarks(editor: Editor, pages: Verification["pages"]) {
  const { doc } = editor.state;
  let marked = 0;
  let checking = 0;
  const rows: Record<VerifyRow, number[]> = {
    contradicted: [],
    unsupported: [],
    notChecked: [],
    recheck: [],
    supported: [],
  };
  for (const claim of shownClaims(editor.state)) {
    const id: unknown = doc.resolve(claim.from).parent.attrs.id;
    if (typeof id !== "string" || !(id in pages)) continue;
    marked++;
    const state = claim.shown.state;
    if (state === "checking") checking++;
    else if (state === "supported") rows.supported.push(claim.from);
    else if (state === "contradicted") rows.contradicted.push(claim.from);
    else if (state === "unsupported") rows.unsupported.push(claim.from);
    else if (state === "needs-recheck") rows.recheck.push(claim.from);
    else rows.notChecked.push(claim.from);
  }
  return { marked, checking, rows, coverage: coverageOf(shownClaims(editor.state)) };
}

const sameMarks = (a: ReturnType<typeof claimMarks>, b: ReturnType<typeof claimMarks> | null) =>
  b !== null &&
  a.marked === b.marked &&
  a.checking === b.checking &&
  sameCoverage(a.coverage, b.coverage) &&
  (Object.keys(a.rows) as VerifyRow[]).every(
    (row) =>
      a.rows[row].length === b.rows[row].length &&
      a.rows[row].every((at, index) => at === b.rows[row][index]),
  );

/** Where a Mind made by "Verify a document…" (#97) is: its claims' checks, its run and its question. */
export function useVerifyState(
  editor: Editor,
  doc: Y.Doc,
  mindId: string,
  verification: Verification,
) {
  const t = useT();
  const language = useLanguage();
  const model = useMindModel(doc);
  const running = useVerify((state) =>
    state.running?.mindId === mindId ? state.running.phase : null,
  );
  const outcome = useVerify((state) => state.outcomes[mindId]);
  const marks = useEditorState({
    editor,
    selector: ({ editor: current }) => claimMarks(current, verification.pages),
    equalityFn: sameMarks,
  });
  const claims = verification.claims;
  const total = claims?.length ?? 0;
  const asking = claims !== null && total > 0 && marks.marked === 0 && running === null;

  const [estimate, setEstimate] = useState<string | null>(null);
  useEffect(() => {
    if (!asking) return;
    let current = true;
    core
      .estimateRun({ items: total, ...(model && { model }) })
      .then((result) => {
        if (current) setEstimate(estimateText(result, t, language));
      })
      .catch(() => {
        // No model to estimate for: checking says why.
        if (current) setEstimate(null);
      });
    return () => {
      current = false;
    };
  }, [asking, total, model, t, language]);

  const problem =
    running === null &&
    (outcome?.kind === "not-ready" ||
      outcome?.kind === "local-mode" ||
      outcome?.kind === "failed" ||
      outcome?.kind === "error");
  return { running, outcome, marks, claims, total, asking, estimate, problem };
}

export type VerifyState = ReturnType<typeof useVerifyState>;

/** Whether a verify's claims have their checks, so its card says what they found. */
export const verifyDone = ({ running, outcome, claims, total, marks }: VerifyState) =>
  running === null &&
  outcome?.kind !== "no-documents" &&
  outcome?.kind !== "nothing" &&
  claims !== null &&
  total > 0 &&
  marks.marked > 0;

/**
 * Where a Mind made by "Verify a document…" took its text from, under its
 * title in place of the word count: "From {name}", which opens it beside.
 */
export function VerifyFrom({ verification }: { verification: Verification }) {
  const t = useT();
  const name = verification.documentName;
  return (
    <div className="mind-verify" data-testid="verify-line">
      <button
        type="button"
        data-testid="verify-from"
        className="mind-verify-from"
        title={t("verify.from.title", { name })}
        onClick={() => openDocumentBeside(verification.documentId)}
      >
        <DocumentLineIcon kind="pdf" className="size-3.5 shrink-0" />
        <span className="min-w-0 truncate">{t("verify.from", { name })}</span>
      </button>
    </div>
  );
}

const VERIFY_ROWS: readonly VerifyRow[] = [
  "contradicted",
  "unsupported",
  "notChecked",
  "recheck",
  "supported",
];
const VERIFY_MARKS: Record<VerifyRow, ClaimMarkState> = {
  contradicted: "contradicted",
  unsupported: "unsupported",
  notChecked: "not-checked",
  recheck: "needs-recheck",
  supported: "supported",
};

/**
 * The check badge's card for a Mind made by "Verify a document…" (#97,
 * #909; the boards Verify-Badge-*): "Check it against this Folder's
 * Documents? [Check 24 claims]" with the estimate on a cloud model, then
 * "Checked 24 claims · 2 need a look" and a row a verdict, each moving to
 * the next such claim with the PDF beside turned to its page; or why there
 * is nothing to check.
 */
export function VerifyCard({
  editor,
  doc,
  mindId,
  verification,
  state,
  onNotNow,
}: {
  editor: Editor;
  doc: Y.Doc;
  mindId: string;
  verification: Verification;
  state: VerifyState;
  onNotNow(): void;
}) {
  const t = useT();
  const language = useLanguage();
  const openSettings = useAppStore((app) => app.openSettings);
  const runModel = useRunModel(doc);
  const inFolder = useAppStore(
    (app) => (app.minds.find((each) => each.id === mindId)?.folderId ?? null) !== null,
  );
  const unread = useCheckUnread(editor, doc);
  const { running, outcome, marks, claims, total, estimate } = state;

  // The counts fade in once, as the run that made them finishes.
  const doneRef = useRef<HTMLDivElement>(null);
  const checked = useRef(false);
  if (running === "checking") checked.current = true;
  useLayoutEffect(() => {
    if (running !== null || !checked.current || !doneRef.current) return;
    checked.current = false;
    enter(doneRef.current, {});
  }, [running]);

  const format = (value: number) => new Intl.NumberFormat(language).format(value);
  const goToNext = (row: VerifyRow) => {
    if (editor.isDestroyed) return;
    const places = marks.rows[row];
    const head = editor.state.selection.head;
    const next = places.find((at) => at > head) ?? places[0];
    if (next === undefined) return;
    const place = editor.state.doc.resolve(next);
    editor.view.dispatch(editor.state.tr.setSelection(TextSelection.near(place)).scrollIntoView());
    editor.view.focus();
    // The PDF beside it turns to the claim's page.
    const id: unknown = place.parent.attrs.id;
    const page = typeof id === "string" ? verification.pages[id] : undefined;
    if (page !== undefined) {
      openDocumentBeside(verification.documentId, { pageFrom: page, pageTo: page });
    }
  };

  let body: ReactNode;
  if (running === "finding") {
    body = <p role="status">{t("verify.finding")}</p>;
  } else if (running === "checking") {
    body = (
      <p role="status" data-testid="verify-progress" className="tabular-nums">
        {t("verify.checking", {
          done: format(marks.marked - marks.checking),
          total: format(marks.marked),
        })}
      </p>
    );
  } else if (outcome?.kind === "no-documents") {
    body = (
      <div className="check-card-ask">
        <p data-testid="verify-empty">
          {t(inFolder ? "verify.noDocuments.folder" : "verify.noDocuments.all")}
        </p>
        <div className="check-card-ask-actions">
          <button
            type="button"
            data-testid="verify-add-files"
            className={buttonStyle("secondary", "sm")}
            onClick={() => void useAppStore.getState().pickDocuments()}
          >
            {t("checks.addFiles")}
          </button>
        </div>
      </div>
    );
  } else if (claims === null) {
    body = (
      <div className="check-card-ask">
        <p>{t("verify.notFound")}</p>
        <div className="check-card-ask-actions">
          <button
            type="button"
            data-testid="verify-find"
            className={buttonStyle("secondary", "sm")}
            onClick={() => void findClaims(mindId, doc)}
          >
            {t("verify.find")}
          </button>
        </div>
      </div>
    );
  } else if (total === 0 || outcome?.kind === "nothing") {
    body = <p data-testid="verify-empty">{t("verify.none")}</p>;
  } else if (marks.marked === 0) {
    body = (
      <div className="check-card-ask" data-testid="verify-ask">
        <p className="check-card-ask-title">
          {t(inFolder ? "verify.ask.folder" : "verify.ask.all")}
        </p>
        {estimate && <p data-testid="verify-estimate">{estimate}</p>}
        <div className="check-card-ask-actions">
          <button
            type="button"
            data-testid="verify-check"
            className={buttonStyle("primary", "sm")}
            onClick={() => void checkClaims(mindId, doc)}
          >
            {t(total === 1 ? "verify.check.one" : "verify.check", { count: format(total) })}
          </button>
          <button
            type="button"
            data-testid="verify-not-now"
            className={buttonStyle("ghost", "sm")}
            onClick={onNotNow}
          >
            {t("mindCheck.cancel")}
          </button>
        </div>
      </div>
    );
  } else {
    const look = marks.marked - marks.checking - marks.rows.supported.length;
    const stopped = outcome?.kind === "checked" ? outcome.stop : null;
    const summary = [
      t(marks.marked === 1 ? "verify.checked.one" : "verify.checked", {
        count: format(marks.marked),
      }),
      ...(look > 0
        ? [t(look === 1 ? "verify.needLook.one" : "verify.needLook", { count: format(look) })]
        : []),
      ...(stopped ? [runStopText(stopped, t, language)] : []),
    ].join(" · ");
    body = (
      <div ref={doneRef} data-testid="verify-done" className="check-rows">
        <p className="check-card-summary">{summary}</p>
        {VERIFY_ROWS.filter((row) => marks.rows[row].length > 0).map((row) => {
          const count = marks.rows[row].length;
          const text = t(
            row === "recheck" && count === 1 ? "verify.row.recheck.one" : `verify.row.${row}`,
            { count: format(count) },
          );
          return (
            <button
              key={row}
              type="button"
              data-testid={`verify-${row}`}
              data-count={count}
              data-part={row === "supported" ? "found" : row}
              className="check-row"
              aria-label={t("verify.needLook.next", { part: text })}
              onMouseDown={(event) => event.preventDefault()}
              onClick={() => goToNext(row)}
            >
              <span className="check-row-mark">
                <ClaimMarkIcon state={VERIFY_MARKS[row]} className="mind-health-icon" />
              </span>
              <span className="check-row-text">
                <Counted text={text} count={format(count)} />
              </span>
              <span className="check-row-next" aria-hidden="true">
                {t("badge.next")}
              </span>
            </button>
          );
        })}
      </div>
    );
  }

  return (
    <>
      <p className="check-card-title">{t("verify.card.title")}</p>
      {body}
      <UnreadFilesLine files={unread} testId="verify-unread" className="mind-unread" />
      {running === null && marks.marked > 0 && marks.coverage.mostlyUncovered && (
        <UncoveredLine
          className="mind-verify-problem"
          onCheckAll={
            marks.coverage.canWiden ? () => void checkClaims(mindId, doc, true) : undefined
          }
        />
      )}
      {running === null && outcome?.kind === "not-ready" && (
        <ProblemLine
          testId="verify-problem"
          className="mind-verify-problem"
          action={readinessAction(outcome.readiness, t, openSettings, t("question.setUp"), () =>
            dismissVerifyOutcome(mindId),
          )}
        >
          <ReadinessExplanation readiness={outcome.readiness} inLine />
        </ProblemLine>
      )}
      {running === null && outcome?.kind === "local-mode" && (
        <ProblemLine
          testId="verify-problem"
          className="mind-verify-problem"
          action={{ label: t("checks.problem.chooseModel"), onClick: () => openSettings("models") }}
        >
          {t("checks.problem.local-mode")}
        </ProblemLine>
      )}
      {running === null && outcome?.kind === "failed" && (
        // The Answer errors' own wordings: the company or model named, and what to do next.
        <AnswerError
          kind={outcome.error.kind}
          modelId={runModel.modelId}
          providerId={runModel.providerId}
          resetsAt={outcome.error.resetsAt ?? null}
          details={outcome.error.message}
          onRetry={() =>
            void (claims === null ? findClaims(mindId, doc) : checkClaims(mindId, doc))
          }
          onOpenSettings={openSettings}
        />
      )}
      {running === null && outcome?.kind === "error" && (
        <ProblemLine
          testId="verify-problem"
          className="mind-verify-problem"
          action={{
            label: t("verify.tryAgain"),
            testId: "verify-try-again",
            onClick: () =>
              void (claims === null ? findClaims(mindId, doc) : checkClaims(mindId, doc)),
          }}
        >
          {t("verify.problem.error", { message: outcome.message })}
        </ProblemLine>
      )}
    </>
  );
}
