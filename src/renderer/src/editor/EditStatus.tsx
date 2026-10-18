import { ProblemLine } from "../components/ProblemLine";
import {
  ReadinessExplanation,
  readinessAction,
  readinessAlso,
} from "../components/providers/ChatReadinessNotice";
import {
  dismissEditProblem,
  type EditProblem,
  stopEdit,
  TRANSLATE_LANGUAGES,
  useEdits,
} from "../editWithAi";
import { providerErrorText } from "../errors";
import { useT } from "../i18n";
import { useAppStore } from "../store";

/**
 * Edit with AI in the composer's row (#93): what the AI is doing with the
 * selected text, with Stop, while it works; or, when it gave no changes, why,
 * as one line with the one thing that helps (DESIGN.md, Problems).
 */
export function EditStatus({ mindId }: { mindId: string }) {
  const t = useT();
  const pending = useEdits((state) => state.pending[mindId]);
  const problem = useEdits((state) => state.problems[mindId]);

  if (pending) {
    const language =
      TRANSLATE_LANGUAGES.find((each) => each.prompt === pending.language)?.name ??
      pending.language ??
      "";
    return (
      <ProblemLine
        role="status"
        testId="edit-pending"
        className="composer-problem"
        action={{ label: t("edit.stop"), testId: "edit-stop", onClick: () => stopEdit(mindId) }}
      >
        <span className="edit-pending-dot" aria-hidden="true" />
        {t(`edit.pending.${pending.action}`, { language })}
      </ProblemLine>
    );
  }
  return problem ? <EditProblemLine mindId={mindId} problem={problem} /> : null;
}

/**
 * The one line for why an Edit with AI gave no changes, with the one thing
 * that helps. `onRetry` is told as Try again asks for the edit again.
 */
export function EditProblemLine({
  mindId,
  problem,
  className = "composer-problem",
  onRetry,
}: {
  mindId: string;
  problem: EditProblem;
  className?: string;
  onRetry?(): void;
}) {
  const t = useT();
  const openSettings = useAppStore((state) => state.openSettings);
  const close = {
    label: t("edit.close"),
    testId: "edit-problem-close",
    onClick: () => dismissEditProblem(mindId),
  };
  switch (problem.kind) {
    case "not-ready":
      return (
        <ProblemLine
          testId="edit-problem"
          className={className}
          action={readinessAction(problem.readiness, t, openSettings, t("question.setUp"), () =>
            dismissEditProblem(mindId),
          )}
          also={readinessAlso(problem.readiness, t, openSettings, () => dismissEditProblem(mindId))}
        >
          <ReadinessExplanation readiness={problem.readiness} inLine />
        </ProblemLine>
      );
    case "local-mode":
      return (
        <ProblemLine
          testId="edit-problem"
          className={className}
          action={{
            label: t("edit.problem.chooseModel"),
            onClick: () => {
              dismissEditProblem(mindId);
              openSettings("models");
            },
          }}
        >
          {t("edit.problem.local-mode")}
        </ProblemLine>
      );
    case "failed":
      return (
        <ProblemLine
          testId="edit-problem"
          className={className}
          action={{
            label: t("edit.problem.retry"),
            onClick: () => {
              dismissEditProblem(mindId);
              onRetry?.();
              problem.retry();
            },
          }}
        >
          {t("edit.problem.failed", { message: problem.message })}
        </ProblemLine>
      );
    case "consent-revoked":
      return (
        <ProblemLine testId="edit-problem" className={className} action={close}>
          {providerErrorText(problem.error, t)}
        </ProblemLine>
      );
    default:
      return (
        <ProblemLine
          role={problem.kind === "unchanged" ? "status" : "alert"}
          testId="edit-problem"
          className={className}
          action={close}
        >
          {t(`edit.problem.${problem.kind}`)}
        </ProblemLine>
      );
  }
}
