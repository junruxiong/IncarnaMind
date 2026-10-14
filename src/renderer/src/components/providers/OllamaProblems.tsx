import type { ReactNode } from "react";
import type { ProviderErrorKind } from "../../../../core/api";
import { useT } from "../../i18n";
import { ollamaLineAction } from "../../ollamaProblem";
import { useRequests } from "../../requests";
import { type ProblemAction, ProblemLine } from "../ProblemLine";
import { getOllama } from "./GetOllama";
import { useOllamaStatus } from "./OllamaCard";

/** A failed Answer's one line under it, as `AnswerError` draws it. */
function AnswerProblem({
  kind,
  details,
  action,
  children,
}: {
  kind: ProviderErrorKind;
  details: string | null;
  action: ProblemAction | undefined;
  children: ReactNode;
}) {
  return (
    <div contentEditable={false} data-testid="answer-error" data-error-kind={kind}>
      <ProblemLine
        action={action && { ...action, testId: "answer-error-action" }}
        // The provider's own words, for whoever wants them.
        title={details ?? undefined}
        className="mt-1 select-none"
      >
        {children}
      </ProblemLine>
    </div>
  );
}

/**
 * "Keep everything on this computer" stopped an Answer whose model isn't on this
 * computer: Use a local model opens the choice in place, never a company's settings.
 */
export function LocalModelRefusedLine({
  model,
  details,
}: {
  model: string;
  details: string | null;
}) {
  const t = useT();
  return (
    <AnswerProblem
      kind="local-mode"
      details={details}
      action={{
        label: t("answer.error.useLocal"),
        onClick: () => useRequests.getState().ask({ chooseLocalModel: true }),
      }}
    >
      {t("answer.error.local-mode", { model })}
    </AnswerProblem>
  );
}

/**
 * The Answer's model is in Ollama, which isn't running. Looks again as it shows:
 * Open Ollama when its app is here, which asks the Question again once Ollama
 * answers; Get Ollama when it isn't; Try again when it runs by now.
 */
export function OllamaNotRunningLine({
  details,
  onRetry,
}: {
  details: string | null;
  onRetry?: (() => void) | undefined;
}) {
  const t = useT();
  const { status, opening, open } = useOllamaStatus();
  const next = ollamaLineAction(status, opening);
  let action: ProblemAction | undefined;
  if (next === "try-again" && onRetry) {
    action = { label: t("answer.error.retry"), onClick: onRetry };
  } else if (next === "open") {
    action = {
      label: t("localModel.open"),
      onClick: () =>
        void open().then((opened) => {
          if (opened?.running) onRetry?.();
        }),
    };
  } else if (next === "get") {
    action = { label: `${t("providers.ollama.get")} ↗`, onClick: getOllama };
  }
  return (
    <AnswerProblem kind="ollama-not-running" details={details} action={action}>
      {opening ? t("localModel.opening") : t("answer.error.ollama-not-running")}
    </AnswerProblem>
  );
}
