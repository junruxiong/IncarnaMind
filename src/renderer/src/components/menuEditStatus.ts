import { useShallow } from "zustand/react/shallow";
import { dismissEditProblem, type EditProblem, stopEdit, useEdits } from "../editWithAi";
import { providerErrorText } from "../errors";
import { useLanguage, useT } from "../i18n";
import { useAppStore } from "../store";
import { readinessAction } from "./providers/ChatReadinessNotice";
import { readinessKey, serviceName } from "./providers/shared";

interface Words {
  text: string;
  action?: { label: string; run(): void };
}

/**
 * An edit asked in the selection menu, once the menu has closed on it: the
 * sidebar footer's status row says what it is doing, with Stop, or why it gave
 * nothing. While the menu is open it says so itself.
 */
export function useMenuEditStatus() {
  const t = useT();
  const language = useLanguage();
  const openSettings = useAppStore((state) => state.openSettings);
  const { mindId, working, problem } = useEdits(
    useShallow((state) => {
      for (const id of new Set([...Object.keys(state.pending), ...Object.keys(state.problems)])) {
        if (state.menuShows[id]) continue;
        const pending = state.pending[id];
        const found = state.problems[id];
        if (pending?.inMenu && !pending.stopped)
          return { mindId: id, working: true, problem: null };
        if (!pending && found?.inMenu) return { mindId: id, working: false, problem: found };
      }
      return { mindId: null, working: false, problem: null };
    }),
  );
  if (mindId === null) return null;
  if (working) {
    const text = t("edit.pending.instruction");
    return {
      testId: "edit-status",
      tone: "progress" as const,
      short: text,
      full: text,
      action: { label: t("edit.stop"), testId: "edit-status-stop", run: () => stopEdit(mindId) },
    };
  }
  if (!problem) return null;
  const words = wordsOf(problem, t, language, openSettings, mindId);
  return {
    testId: "edit-status-problem",
    tone: "error" as const,
    alerts: true as const,
    wraps: true as const,
    short: words.text,
    full: words.text,
    ...(words.action && { action: { ...words.action, testId: "edit-status-action" } }),
    dismiss: () => dismissEditProblem(mindId),
  };
}

function wordsOf(
  problem: EditProblem,
  t: ReturnType<typeof useT>,
  language: ReturnType<typeof useLanguage>,
  openSettings: ReturnType<typeof useAppStore.getState>["openSettings"],
  mindId: string,
): Words {
  switch (problem.kind) {
    case "not-ready": {
      const { readiness } = problem;
      const service = "provider" in readiness ? serviceName(readiness.provider, t, language) : "";
      const way = readinessAction(readiness, t, openSettings, t("question.setUp"), () =>
        dismissEditProblem(mindId),
      );
      return {
        text: t(readinessKey(readiness), { service }).replace(/[.。]$/, ""),
        action: { label: way.label, run: way.onClick },
      };
    }
    case "local-mode":
      return {
        text: t("edit.problem.local-mode"),
        action: {
          label: t("edit.problem.chooseModel"),
          run: () => {
            dismissEditProblem(mindId);
            openSettings("models");
          },
        },
      };
    case "failed":
      return {
        text: t("edit.problem.failed", { message: problem.message }),
        action: {
          label: t("edit.problem.retry"),
          run: () => {
            dismissEditProblem(mindId);
            problem.retry();
          },
        },
      };
    case "consent-revoked":
      return { text: providerErrorText(problem.error, t) };
    default:
      return { text: t(`edit.problem.${problem.kind}`) };
  }
}
