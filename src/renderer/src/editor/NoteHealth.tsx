import type { Editor } from "@tiptap/core";
import { TextSelection } from "@tiptap/pm/state";
import { useEditorState } from "@tiptap/react";
import { useEffect, useState } from "react";
import type * as Y from "yjs";
import type { CitationAttributes } from "../../../core/api";
import { citationState } from "../../../shared/citations";
import {
  acceptAndCheck,
  checkSentences,
  dismissCheckProblem,
  stopCheck,
  useChecks,
} from "../checks";
import { ProblemLine } from "../components/ProblemLine";
import { ReadinessExplanation, readinessAction } from "../components/providers/ChatReadinessNotice";
import { UnreadFilesLine } from "../components/UnreadFilesLine";
import { useContradictionLine } from "../contradictions";
import { errorMessage } from "../errors";
import { useLanguage, useT } from "../i18n";
import { checkMind, startMindCheck, useMindCheck, useMindCheckRun } from "../mindCheck";
import { useAppStore } from "../store";
import { useCheckUnread } from "../unreadFiles";
import { AnswerError } from "./AnswerView";
import { coverageOf, sameCoverage, shownClaims } from "./claimChecks";
import { type NoteHealth as Health, type HealthPart, noteHealth } from "./healthCounts";
import { CheckMarkIcon, ClaimMarkIcon } from "./icons";
import { CheckedWhen, MindCheckAction, MindCheckProgress } from "./MindCheckLine";
import { useRunModel } from "./ModelChip";
import { UncoveredLine } from "./UncoveredLine";
import { COUNT_IDLE_MS } from "./WordCount";

export const NO_HEALTH: Health = {
  found: [],
  notFound: [],
  unsupported: [],
  contradicted: [],
  noCitation: [],
  recheck: [],
  notChecked: [],
  unchecked: [],
};

/** Each part's small mark, as the margin draws its state; none for sentences with no Citation. */
export function PartMark({ part }: { part: HealthPart }) {
  switch (part) {
    case "found":
      return <CheckMarkIcon check="found" className="mind-health-icon" />;
    case "notFound":
      return <CheckMarkIcon check="not-found" className="mind-health-icon" />;
    case "unsupported":
      return <ClaimMarkIcon state="unsupported" className="mind-health-icon" />;
    case "contradicted":
      return <ClaimMarkIcon state="contradicted" className="mind-health-icon" />;
    case "recheck":
      return <ClaimMarkIcon state="needs-recheck" className="mind-health-icon" />;
    case "notChecked":
      return <ClaimMarkIcon state="not-checked" className="mind-health-icon" />;
    case "noCitation":
      return null;
  }
}

/** The Mind's health, worked out a moment after it last changed, as the word count is. */
export function useHealth(editor: Editor, skipped: ReadonlySet<string>): Health {
  const documents = useAppStore((state) =>
    state.status.kind === "ready" ? state.documents : null,
  );
  const kept = useAppStore((state) => state.keptCitationTexts);
  const [health, setHealth] = useState<Health>(NO_HEALTH);
  useEffect(() => {
    let timer: ReturnType<typeof setTimeout> | undefined;
    const count = () => {
      timer = undefined;
      if (editor.isDestroyed) return;
      setHealth(
        noteHealth(
          editor.state.doc,
          shownClaims(editor.state),
          (attributes: CitationAttributes) => citationState(attributes, documents, kept).check,
          skipped,
        ),
      );
    };
    const later = () => {
      clearTimeout(timer);
      timer = setTimeout(count, COUNT_IDLE_MS);
    };
    count();
    editor.on("transaction", later);
    return () => {
      clearTimeout(timer);
      editor.off("transaction", later);
    };
  }, [editor, documents, kept, skipped]);
  return health;
}

/** The rows, after the problems' first, in the badge's card. */
const ROWS: readonly HealthPart[] = [
  "contradicted",
  "unsupported",
  "notFound",
  "notChecked",
  "recheck",
  "found",
];

/**
 * The check badge's card for a Mind (#96, #909): one row a part of its
 * health, each moving to the next such sentence or Citation, round to the
 * first; the claims no check has looked at, with Check all; what is
 * running, with Stop, or why a check gave nothing, with the one thing that
 * helps; and the files the checks couldn't read, if any.
 */
export function HealthCard({
  editor,
  doc,
  mindId,
  health,
}: {
  editor: Editor;
  doc: Y.Doc;
  mindId: string;
  health: Health;
}) {
  const t = useT();
  const language = useLanguage();
  const checkedAll = useMindCheckRun(doc);
  const unread = useCheckUnread(editor, doc);
  const asking = useMindCheck((state) => state.asking[mindId] !== undefined);
  const coverage = useEditorState({
    editor,
    selector: ({ editor: current }) => coverageOf(shownClaims(current.state)),
    equalityFn: sameCoverage,
  });
  const runModel = useRunModel(doc);
  const running = useChecks((state) => state.running[mindId]);
  const problem = useChecks((state) => state.problems[mindId]);
  const smallModel = useChecks((state) => state.smallModel[mindId] === true);
  const contradictions = useContradictionLine(doc, mindId);
  const openSettings = useAppStore((state) => state.openSettings);
  const folderName = useAppStore((state) => {
    const folderId =
      problem?.kind === "no-documents" || problem?.kind === "only-cited" ? problem.folderId : null;
    if (folderId === null) return null;
    return state.library?.groups.find((group) => group.id === folderId)?.name ?? null;
  });
  const format = (value: number) => new Intl.NumberFormat(language).format(value);

  const label = (part: HealthPart, count: number) => {
    const shown = { count: format(count) };
    switch (part) {
      case "found":
        return count === 1 ? t("health.found.one") : t("health.found", shown);
      case "notFound":
        return count === 1 ? t("health.notFound.one") : t("health.notFound", shown);
      case "recheck":
        return count === 1 ? t("health.recheck", shown) : t("health.recheck.other", shown);
      default:
        return t(`health.${part}`, shown);
    }
  };

  /** Puts the cursor at the next such sentence or Citation after it, round to the first. */
  const goToNext = (part: HealthPart) => {
    if (editor.isDestroyed) return;
    const places = health[part];
    const head = editor.state.selection.head;
    const next = places.find((pos) => pos > head) ?? places[0];
    if (next === undefined) return;
    const { doc } = editor.state;
    const at = Math.min(next, doc.content.size);
    editor.view.dispatch(
      editor.state.tr.setSelection(TextSelection.near(doc.resolve(at))).scrollIntoView(),
    );
    editor.view.focus();
  };

  const rows = ROWS.filter((part) => health[part].length > 0);
  const canCheck =
    !running && !asking && (health.unchecked.length > 0 || health.recheck.length > 0);
  const close = {
    label: t("checks.close"),
    testId: "checks-problem-close",
    onClick: () => dismissCheckProblem(mindId),
  };
  const row = (part: HealthPart) => {
    const text = label(part, health[part].length);
    return (
      <button
        key={part}
        type="button"
        data-testid={`health-${part}`}
        data-count={health[part].length}
        data-part={part}
        className="check-row"
        aria-label={t("health.next", { part: text })}
        onMouseDown={(event) => event.preventDefault()}
        onClick={() => goToNext(part)}
      >
        <span className="check-row-mark">
          <PartMark part={part} />
        </span>
        <span className="check-row-text">{text}</span>
        <span className="check-row-next" aria-hidden="true">
          {t("badge.next")}
        </span>
      </button>
    );
  };

  return (
    <>
      <div className="check-rows" data-testid="mind-health">
        {rows.map(row)}
        {health.unchecked.length > 0 && (
          <div className="check-row" data-testid="health-unchecked">
            <span className="check-row-mark" />
            <span className="check-row-text">
              {t(health.unchecked.length === 1 ? "health.unchecked.one" : "health.unchecked", {
                count: format(health.unchecked.length),
              })}
            </span>
            {canCheck && (
              <MindCheckAction
                unchecked={health.unchecked.length}
                recheck={health.recheck.length}
                onCheck={() => void startMindCheck(mindId, "changed")}
              />
            )}
          </div>
        )}
        {health.unchecked.length === 0 && canCheck && (
          <div className="check-card-actions">
            <MindCheckAction
              unchecked={0}
              recheck={health.recheck.length}
              onCheck={() => void startMindCheck(mindId, "changed")}
            />
          </div>
        )}
        {health.noCitation.length > 0 && row("noCitation")}
      </div>
      {checkedAll && !running && (
        <p className="check-card-note">
          <CheckedWhen run={checkedAll} />
        </p>
      )}
      {((smallModel && rows.length > 0) ||
        (contradictions?.kind !== "asking" && contradictions?.run.smallModel)) && (
        <p data-testid="checks-small-model" className="mind-health-line mind-health-note">
          {t("checks.smallModel")}
        </p>
      )}
      {running && (
        <ProblemLine
          role="status"
          testId="checks-running"
          className="mind-health-line"
          action={{
            label: t("checks.stop"),
            testId: "checks-stop",
            onClick: () => stopCheck(mindId),
          }}
        >
          {running.action === "whole-mind" ? (
            <MindCheckProgress editor={editor} requestId={running.requestId} />
          ) : (
            t(`checks.running.${running.action}`)
          )}
        </ProblemLine>
      )}
      {!running && problem?.kind === "not-ready" && (
        <ProblemLine
          testId="checks-problem"
          className="mind-health-line"
          action={readinessAction(problem.readiness, t, openSettings, t("question.setUp"), () =>
            dismissCheckProblem(mindId),
          )}
        >
          <ReadinessExplanation readiness={problem.readiness} inLine />
        </ProblemLine>
      )}
      {!running && problem?.kind === "local-mode" && (
        <ProblemLine
          testId="checks-problem"
          className="mind-health-line"
          action={{
            label: t("checks.problem.chooseModel"),
            onClick: () => {
              dismissCheckProblem(mindId);
              openSettings("models");
            },
          }}
        >
          {t("checks.problem.local-mode")}
        </ProblemLine>
      )}
      {!running && !asking && !problem && coverage.mostlyUncovered && (
        <UncoveredLine
          className="mind-health-line"
          onCheckAll={
            !coverage.canWiden
              ? undefined
              : checkedAll
                ? () => void startMindCheck(mindId, "all", { allDocuments: true })
                : coverage.range && coverage.action
                  ? () =>
                      void checkSentences(editor, mindId, doc, coverage.action ?? "find-support", {
                        ...(coverage.range && { ranges: [coverage.range] }),
                        allDocuments: true,
                      })
                  : undefined
          }
        />
      )}
      {!running && problem?.kind === "no-documents" && (
        <ProblemLine
          testId="checks-problem"
          className="mind-health-line"
          action={{
            label: t("checks.addFiles"),
            testId: "checks-add-files",
            onClick: () => void useAppStore.getState().pickDocuments(),
          }}
          also={close}
        >
          {folderName
            ? t("checks.problem.no-documents.folder", { folder: folderName })
            : t("checks.problem.no-documents.all")}
        </ProblemLine>
      )}
      {!running && problem?.kind === "waiting" && (
        <ProblemLine
          testId="checks-waiting"
          className="mind-health-line"
          action={{
            label: t("checks.acceptAndCheck"),
            testId: "checks-accept-and-check",
            onClick: () => {
              dismissCheckProblem(mindId);
              void acceptAndCheck(editor, mindId, doc, problem);
            },
          }}
          also={close}
        >
          {problem.count === 1
            ? t("checks.waiting.one")
            : t("checks.waiting", { count: format(problem.count) })}
        </ProblemLine>
      )}
      {!running && problem?.kind === "find-failed" && (
        <AnswerError
          kind={problem.error.kind}
          modelId={runModel.modelId}
          providerId={runModel.providerId}
          resetsAt={problem.error.resetsAt ?? null}
          details={problem.error.message}
          onRetry={() =>
            void checkMind(mindId, problem.mode, { allDocuments: problem.allDocuments })
          }
          onOpenSettings={openSettings}
        />
      )}
      {!running &&
        problem &&
        problem.kind !== "not-ready" &&
        problem.kind !== "local-mode" &&
        problem.kind !== "no-documents" &&
        problem.kind !== "waiting" &&
        problem.kind !== "find-failed" && (
          <ProblemLine
            testId="checks-problem"
            className="mind-health-line"
            action={
              problem.kind === "failed" && problem.retry
                ? {
                    label: t("checks.tryAgain"),
                    testId: "checks-try-again",
                    onClick: () => {
                      dismissCheckProblem(mindId);
                      problem.retry?.();
                    },
                  }
                : close
            }
            also={problem.kind === "failed" && problem.retry ? close : undefined}
          >
            {problem.kind === "only-cited"
              ? t("checks.problem.only-cited.folder", { folder: folderName ?? "" })
              : problem.kind === "failed"
                ? t("checks.problem.failed", { message: errorMessage(problem.error, t) })
                : t(`checks.problem.${problem.kind}`)}
          </ProblemLine>
        )}
      <UnreadFilesLine files={unread} testId="checks-unread" className="mind-unread" />
    </>
  );
}
