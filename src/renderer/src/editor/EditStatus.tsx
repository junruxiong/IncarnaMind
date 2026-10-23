import type { Language } from "../../../core/language";
import { ProblemLine } from "../components/ProblemLine";
import {
  readinessAction,
  readinessAlso,
  readinessInLine,
} from "../components/providers/ChatReadinessNotice";
import type { EditProblemWords } from "../editProblems";
import {
  dismissEditProblem,
  type EditProblem,
  stopEdit,
  TRANSLATE_LANGUAGES,
  useEdits,
} from "../editWithAi";
import { providerErrorText } from "../errors";
import { useLanguage, useT } from "../i18n";
import type { SettingsPage } from "../settingsPages";
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

  if (pending?.menu || problem?.menu) return null;
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
 * Why an Edit with AI gave no changes, with the one thing that helps: the
 * words of its line in the composer, in the selection menu and in the page.
 * `retrying` is told as Try again asks for the edit again.
 */
export function editProblemWords(
  problem: EditProblem,
  mindId: string,
  say: {
    t: ReturnType<typeof useT>;
    language: Language;
    openSettings(page: SettingsPage): void;
  },
  retrying?: () => void,
): EditProblemWords {
  const { t, language, openSettings } = say;
  const close = () => dismissEditProblem(mindId);
  switch (problem.kind) {
    case "not-ready":
      return {
        text: readinessInLine(problem.readiness, t, language),
        role: "alert",
        action: readinessAction(problem.readiness, t, openSettings, t("question.setUp"), close),
        also: readinessAlso(problem.readiness, t, openSettings, close),
      };
    case "local-mode":
      return {
        text: t("edit.problem.local-mode"),
        role: "alert",
        action: {
          label: t("edit.problem.chooseModel"),
          onClick: () => {
            close();
            openSettings("models");
          },
        },
      };
    case "failed":
      return {
        text: t("edit.problem.failed", { message: problem.message }),
        role: "alert",
        action: {
          label: t("edit.problem.retry"),
          onClick: () => {
            close();
            retrying?.();
            problem.retry();
          },
        },
      };
    case "consent-revoked":
      return { text: providerErrorText(problem.error, t), role: "alert" };
    default:
      return {
        text: t(`edit.problem.${problem.kind}`),
        role: problem.kind === "unchanged" ? "status" : "alert",
      };
  }
}

/** The one line for why an Edit with AI gave no changes, with the one thing that helps. */
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
  const language = useLanguage();
  const openSettings = useAppStore((state) => state.openSettings);
  const words = editProblemWords(problem, mindId, { t, language, openSettings }, onRetry);
  return (
    <ProblemLine
      role={words.role}
      testId="edit-problem"
      className={className}
      action={
        words.action ?? {
          label: t("edit.close"),
          testId: "edit-problem-close",
          onClick: () => dismissEditProblem(mindId),
        }
      }
      also={words.also}
    >
      {words.text}
    </ProblemLine>
  );
}
