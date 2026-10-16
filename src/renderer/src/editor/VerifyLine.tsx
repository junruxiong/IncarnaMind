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
import { DocumentLineIcon, VerifyLineIcon } from "../components/lineIcons";
import { MetaLine } from "../components/MetaLine";
import { ProblemLine } from "../components/ProblemLine";
import { ReadinessExplanation, readinessAction } from "../components/providers/ChatReadinessNotice";
import { UnreadFilesLine } from "../components/UnreadFilesLine";
import { buttonStyle } from "../components/ui";
import { core } from "../core";
import { type EstimateLine, estimateLine, runStopParts } from "../cost";
import { useLanguage, useT } from "../i18n";
import { enter } from "../motion";
import { useAppStore } from "../store";
import { useCheckUnread } from "../unreadFiles";
import {
  checkClaims,
  dismissVerifyOutcome,
  findClaims,
  openDocumentBeside,
  stopVerify,
  useVerify,
} from "../verify";
import { AnswerError } from "./AnswerView";
import { coverageOf, sameCoverage, shownClaims } from "./claimChecks";
import { useMindModel, useRunModel } from "./ModelChip";
import { UncoveredLine } from "./UncoveredLine";

const partsOf = (...parts: ReactNode[]): ReactNode[] => parts;

/** What a Mind made by "Verify a document…" verifies, as it changes. */
export function useVerification(doc: Y.Doc): Verification | null {
  const [verification, setVerification] = useState(() => verificationOf(doc));
  useEffect(() => {
    setVerification(verificationOf(doc));
    return observeVerification(doc, () => setVerification(verificationOf(doc)));
  }, [doc]);
  return verification;
}

/** The checks of the document's sentences: how many there are, are going, and need a look, and where. */
function claimMarks(editor: Editor, pages: Verification["pages"]) {
  const { doc } = editor.state;
  let marked = 0;
  let checking = 0;
  const needLook: number[] = [];
  for (const claim of shownClaims(editor.state)) {
    const id: unknown = doc.resolve(claim.from).parent.attrs.id;
    if (typeof id !== "string" || !(id in pages)) continue;
    marked++;
    if (claim.shown.state === "checking") checking++;
    else if (claim.shown.state !== "supported") needLook.push(claim.from);
  }
  return { marked, checking, needLook, coverage: coverageOf(shownClaims(editor.state)) };
}

/**
 * The line under the title of a Mind made by "Verify a document…" (#97; the
 * boards Final-35-Verify-Start and Final-08-Verify), in place of the word
 * count: where its text came from, then where its verify is. "Check it
 * against this Folder's Documents? [Check 24 claims]" with the estimate on a
 * cloud model, "Checking 12 of 24…" with Stop, then "Checked 24 claims · 2
 * need a look", which moves to the next such claim and fades in once as the
 * run finishes; or why there is nothing to check. Last on the line, the files
 * the checks couldn't read, if any.
 */
export function VerifyLine({
  editor,
  doc,
  mindId,
  verification,
}: {
  editor: Editor;
  doc: Y.Doc;
  mindId: string;
  verification: Verification;
}) {
  const t = useT();
  const language = useLanguage();
  const model = useMindModel(doc);
  const running = useVerify((state) =>
    state.running?.mindId === mindId ? state.running.phase : null,
  );
  const outcome = useVerify((state) => state.outcomes[mindId]);
  const openSettings = useAppStore((state) => state.openSettings);
  const runModel = useRunModel(doc);
  const inFolder = useAppStore(
    (state) => (state.minds.find((each) => each.id === mindId)?.folderId ?? null) !== null,
  );
  const marks = useEditorState({
    editor,
    selector: ({ editor: current }) => claimMarks(current, verification.pages),
    equalityFn: (a, b) =>
      b !== null &&
      a.marked === b.marked &&
      a.checking === b.checking &&
      sameCoverage(a.coverage, b.coverage) &&
      a.needLook.length === b.needLook.length &&
      a.needLook.every((at, index) => at === b.needLook[index]),
  });
  const unread = useCheckUnread(editor, doc);
  const claims = verification.claims;
  const total = claims?.length ?? 0;
  const asking = claims !== null && total > 0 && marks.marked === 0 && running === null;

  const [estimate, setEstimate] = useState<EstimateLine | null>(null);
  useEffect(() => {
    if (!asking) return;
    let current = true;
    core
      .estimateRun({ items: total, ...(model && { model }) })
      .then((result) => {
        if (current) setEstimate(estimateLine(result, t, language));
      })
      .catch(() => {
        // No model to estimate for: checking says why.
        if (current) setEstimate(null);
      });
    return () => {
      current = false;
    };
  }, [asking, total, model, t, language]);

  // The counts fade in once, as the run that made them finishes.
  const lineRef = useRef<HTMLDivElement>(null);
  const checked = useRef(false);
  if (running === "checking") checked.current = true;
  useLayoutEffect(() => {
    if (running !== null || !checked.current || !lineRef.current) return;
    checked.current = false;
    enter(lineRef.current, {});
  }, [running]);

  const format = (value: number) => new Intl.NumberFormat(language).format(value);
  const goToNext = () => {
    if (editor.isDestroyed) return;
    const head = editor.state.selection.head;
    const next = marks.needLook.find((at) => at > head) ?? marks.needLook[0];
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

  const stop = (
    <button
      type="button"
      data-testid="verify-stop"
      className="mind-verify-link"
      onClick={() => stopVerify(mindId)}
    >
      {t("checks.stop")}
    </button>
  );
  let state: ReactNode[];
  if (running === "finding") {
    state = partsOf(<span role="status">{t("verify.finding")}</span>, stop);
  } else if (running === "checking") {
    const progress = t("verify.checking", {
      done: format(marks.marked - marks.checking),
      total: format(marks.marked),
    });
    state = partsOf(
      <span role="status" data-testid="verify-progress" className="tabular-nums">
        {progress}
      </span>,
      stop,
    );
  } else if (outcome?.kind === "no-documents") {
    state = partsOf(
      <span data-testid="verify-empty">
        {t(inFolder ? "verify.noDocuments.folder" : "verify.noDocuments.all")}
      </span>,
      <button
        type="button"
        data-testid="verify-add-files"
        className={buttonStyle("secondary", "sm")}
        onClick={() => void useAppStore.getState().pickDocuments()}
      >
        {t("checks.addFiles")}
      </button>,
    );
  } else if (claims === null) {
    state = partsOf(
      <span>{t("verify.notFound")}</span>,
      <button
        type="button"
        data-testid="verify-find"
        className={buttonStyle("secondary", "sm")}
        onClick={() => void findClaims(mindId, doc)}
      >
        {t("verify.find")}
      </button>,
    );
  } else if (total === 0 || outcome?.kind === "nothing") {
    state = partsOf(<span data-testid="verify-empty">{t("verify.none")}</span>);
  } else if (marks.marked === 0) {
    state = partsOf(
      <span>{t(inFolder ? "verify.ask.folder" : "verify.ask.all")}</span>,
      <button
        type="button"
        data-testid="verify-check"
        title={estimate?.title}
        className={buttonStyle("secondary", "sm")}
        onClick={() => void checkClaims(mindId, doc)}
      >
        <VerifyLineIcon className="size-3.5 text-ink-secondary" />
        {t(total === 1 ? "verify.check.one" : "verify.check", { count: format(total) })}
      </button>,
      ...(estimate?.parts ?? []).map((part) => (
        <span key={part} data-testid="verify-estimate">
          {part}
        </span>
      )),
    );
  } else {
    const look = marks.needLook.length;
    const lookText = t(look === 1 ? "verify.needLook.one" : "verify.needLook", {
      count: format(look),
    });
    const stopped = outcome?.kind === "checked" ? outcome.stop : null;
    state = partsOf(
      <span>
        {t(marks.marked === 1 ? "verify.checked.one" : "verify.checked", {
          count: format(marks.marked),
        })}
      </span>,
      look > 0 && (
        <button
          type="button"
          data-testid="verify-need-look"
          className="mind-health-part"
          aria-label={t("verify.needLook.next", { part: lookText })}
          title={t("verify.needLook.next", { part: lookText })}
          onMouseDown={(event) => event.preventDefault()}
          onClick={goToNext}
        >
          {lookText}
        </button>
      ),
      ...(stopped ? runStopParts(stopped, t, language) : []).map((part) => (
        <span key={part} data-testid="verify-stopped">
          {part}
        </span>
      )),
      <button
        type="button"
        data-testid="verify-again"
        className="mind-verify-link"
        onClick={() => void checkClaims(mindId, doc)}
      >
        {t("claim.checkAgain")}
      </button>,
    );
  }

  const name = verification.documentName;
  return (
    <>
      <div ref={lineRef} className="mind-verify" data-testid="verify-line">
        <MetaLine
          as="div"
          parts={partsOf(
            <button
              type="button"
              data-testid="verify-from"
              className="mind-verify-from"
              title={t("verify.from.title", { name })}
              onClick={() => openDocumentBeside(verification.documentId)}
            >
              <DocumentLineIcon kind="pdf" className="size-3.5 shrink-0" />
              <span className="min-w-0 truncate">{t("verify.from", { name })}</span>
            </button>,
            ...state,
          )}
        >
          <UnreadFilesLine files={unread} testId="verify-unread" className="mind-unread" />
        </MetaLine>
      </div>
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
