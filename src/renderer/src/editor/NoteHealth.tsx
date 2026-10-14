import type { Editor } from "@tiptap/core";
import { TextSelection } from "@tiptap/pm/state";
import { useEditorState } from "@tiptap/react";
import { Fragment, useEffect, useState } from "react";
import type * as Y from "yjs";
import type { CitationAttributes } from "../../../core/api";
import { citationState } from "../../../shared/citations";
import { checkSentences, dismissCheckProblem, stopCheck, useChecks } from "../checks";
import { ProblemLine } from "../components/ProblemLine";
import { ReadinessExplanation, readinessAction } from "../components/providers/ChatReadinessNotice";
import { useLanguage, useT } from "../i18n";
import { useKeptScreens } from "../meetings";
import { useAppStore } from "../store";
import { coverageOf, sameCoverage, shownClaims } from "./claimChecks";
import {
  HEALTH_PARTS,
  type NoteHealth as Health,
  type HealthPart,
  noteHealth,
} from "./healthCounts";
import { CheckMarkIcon, ClaimMarkIcon } from "./icons";
import { UncoveredLine } from "./UncoveredLine";
import { COUNT_IDLE_MS } from "./WordCount";

const NONE: Health = {
  found: [],
  notFound: [],
  unsupported: [],
  contradicted: [],
  noCitation: [],
  recheck: [],
  notChecked: [],
};

/** Each part's small mark, as the margin draws its state; none for sentences with no Citation. */
function PartMark({ part }: { part: HealthPart }) {
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
function useHealth(editor: Editor): Health {
  const documents = useAppStore((state) =>
    state.status.kind === "ready" ? state.documents : null,
  );
  const kept = useAppStore((state) => state.keptCitationTexts);
  const screens = useKeptScreens();
  const [health, setHealth] = useState<Health>(NONE);
  useEffect(() => {
    let timer: ReturnType<typeof setTimeout> | undefined;
    const count = () => {
      timer = undefined;
      if (editor.isDestroyed) return;
      setHealth(
        noteHealth(
          editor.state.doc,
          shownClaims(editor.state),
          (attributes: CitationAttributes) =>
            citationState(attributes, documents, kept, screens).check,
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
  }, [editor, documents, kept, screens]);
  return health;
}

/**
 * The Mind's health under its title (#96): "✓ 1 Citation found · ! 1
 * unsupported · ≠ 1 contradicted · 3 with no Citation", then those that need
 * a re-check or weren't checked. Each part is a button that moves to the
 * next such sentence or Citation, round to the first. Under it, while
 * sentences are checked, what is happening with Stop; or why a check gave
 * nothing, with the one thing that helps.
 */
export function NoteHealth({
  editor,
  mindId,
  doc,
}: {
  editor: Editor;
  mindId: string;
  doc: Y.Doc;
}) {
  const t = useT();
  const language = useLanguage();
  const health = useHealth(editor);
  const coverage = useEditorState({
    editor,
    selector: ({ editor: current }) => coverageOf(shownClaims(current.state)),
    equalityFn: sameCoverage,
  });
  const running = useChecks((state) => state.running[mindId]);
  const problem = useChecks((state) => state.problems[mindId]);
  const smallModel = useChecks((state) => state.smallModel[mindId] === true);
  const openSettings = useAppStore((state) => state.openSettings);
  const folderName = useAppStore((state) => {
    if (problem?.kind !== "no-documents" || problem.folderId === null) return null;
    return state.library?.groups.find((group) => group.id === problem.folderId)?.name ?? null;
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

  const parts = HEALTH_PARTS.filter((part) => health[part].length > 0);
  const close = {
    label: t("checks.close"),
    testId: "checks-problem-close",
    onClick: () => dismissCheckProblem(mindId),
  };

  return (
    <>
      {parts.length > 0 && (
        <fieldset className="mind-health" data-testid="mind-health" aria-label={t("health.label")}>
          {parts.map((part) => {
            const text = label(part, health[part].length);
            return (
              <Fragment key={part}>
                <span className="mind-health-dot" aria-hidden="true">
                  ·
                </span>
                <button
                  type="button"
                  data-testid={`health-${part}`}
                  data-count={health[part].length}
                  data-part={part}
                  className="mind-health-part"
                  aria-label={t("health.next", { part: text })}
                  title={t("health.next", { part: text })}
                  onMouseDown={(event) => event.preventDefault()}
                  onClick={() => goToNext(part)}
                >
                  <PartMark part={part} />
                  {text}
                </button>
              </Fragment>
            );
          })}
        </fieldset>
      )}
      {smallModel && parts.length > 0 && (
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
          {t(`checks.running.${running.action}`)}
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
      {!running && !problem && coverage.mostlyUncovered && (
        <UncoveredLine
          className="mind-health-line"
          onCheckAll={
            coverage.canWiden && coverage.range && coverage.action
              ? () =>
                  void checkSentences(editor, mindId, doc, coverage.action ?? "find-support", {
                    ...(coverage.range && { range: coverage.range }),
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
      {!running &&
        problem &&
        problem.kind !== "not-ready" &&
        problem.kind !== "local-mode" &&
        problem.kind !== "no-documents" && (
          <ProblemLine testId="checks-problem" className="mind-health-line" action={close}>
            {problem.kind === "failed"
              ? t("checks.problem.failed", { message: problem.message })
              : t(`checks.problem.${problem.kind}`)}
          </ProblemLine>
        )}
    </>
  );
}
