import { NodeViewContent, NodeViewWrapper, type ReactNodeViewProps } from "@tiptap/react";
import { BLOCK_ID_ATTRIBUTE, type ProviderErrorKind } from "../../../core/api";
import { useAnswers } from "../answers";
import { RegenerateIcon, StopIcon } from "../components/icons";
import { useT } from "../i18n";
import { useAppStore } from "../store";
import { useMindId } from "./mindContext";

const text = (value: unknown) => (typeof value === "string" ? value : null);

const errorKinds: Record<ProviderErrorKind, { fixInSettings: boolean }> = {
  auth: { fixInSettings: true },
  model: { fixInSettings: true },
  "consent-declined": { fixInSettings: true },
  "rate-limit": { fixInSettings: false },
  network: { fixInSettings: false },
  provider: { fixInSettings: false },
  unknown: { fixInSettings: false },
};

const isErrorKind = (value: unknown): value is ProviderErrorKind =>
  typeof value === "string" && Object.hasOwn(errorKinds, value);

/**
 * An Answer: which model wrote it, whether it is still being written (with a
 * stop button) or was stopped, a way to write it again, and, when it failed,
 * what went wrong and how to fix it. Its text is ordinary, editable Blocks.
 */
export function AnswerView({ node }: ReactNodeViewProps) {
  const t = useT();
  const mindId = useMindId();
  const answerId = text(node.attrs[BLOCK_ID_ATTRIBUTE]);
  const questionId = text(node.attrs.questionId);
  const modelId = text(node.attrs.modelId);
  const status = text(node.attrs.status) ?? "done";
  const writing = useAnswers((state) => (answerId ? state.writing.has(answerId) : false));
  // The event says so first; the status says so in windows that opened since.
  const streaming = writing || status === "streaming";
  const confirming = useAnswers(
    (state) => questionId !== null && state.blocked[questionId]?.kind === "edited",
  );
  const { regenerate, stop, dismiss } = useAnswers.getState();
  const openSettings = useAppStore((state) => state.openSettings);

  const writeAgain = (discardEdits = false) => {
    if (answerId && questionId) void regenerate(mindId, answerId, questionId, discardEdits);
  };

  return (
    <NodeViewWrapper
      className="answer-block"
      data-testid="answer"
      data-status={streaming ? "streaming" : status}
      data-answer-id={answerId ?? undefined}
    >
      <div contentEditable={false} className="answer-header">
        <span className="font-medium text-gray-500">{t("answer.label")}</span>
        {modelId && (
          <span data-testid="answer-model" className="truncate">
            · {modelId}
          </span>
        )}
        {streaming && <span className="answer-writing">{t("answer.status.streaming")}</span>}
        {!streaming && status === "stopped" && (
          <span data-testid="answer-stopped">· {t("answer.status.stopped")}</span>
        )}
        <span className="flex-1" />
        {streaming && answerId && (
          <button
            type="button"
            data-testid="answer-stop"
            onClick={() => stop(mindId, answerId)}
            className="answer-action"
          >
            <StopIcon className="size-3.5" />
            {t("answer.stop")}
          </button>
        )}
        {!streaming && questionId && (
          <button
            type="button"
            data-testid="answer-regenerate"
            onClick={() => writeAgain()}
            className="answer-action answer-action--on-hover"
          >
            <RegenerateIcon className="size-3.5" />
            {t("answer.regenerate")}
          </button>
        )}
      </div>

      {confirming && questionId && (
        <div
          contentEditable={false}
          role="alertdialog"
          data-testid="answer-confirm"
          className="answer-notice"
        >
          <span className="flex-1">{t("answer.edited.body")}</span>
          <button
            type="button"
            data-testid="answer-confirm-replace"
            onClick={() => writeAgain(true)}
            className="answer-notice-action"
          >
            {t("answer.edited.replace")}
          </button>
          <button
            type="button"
            onClick={() => dismiss(questionId)}
            className="answer-notice-action"
          >
            {t("answer.edited.keep")}
          </button>
        </div>
      )}

      {!streaming && status === "failed" && (
        <AnswerError
          kind={isErrorKind(node.attrs.errorKind) ? node.attrs.errorKind : "unknown"}
          details={text(node.attrs.errorMessage)}
          onRetry={questionId ? () => writeAgain() : undefined}
          onOpenSettings={openSettings}
        />
      )}

      <NodeViewContent className="answer-content" />
    </NodeViewWrapper>
  );
}

function AnswerError({
  kind,
  details,
  onRetry,
  onOpenSettings,
}: {
  kind: ProviderErrorKind;
  details: string | null;
  onRetry?: () => void;
  onOpenSettings(): void;
}) {
  const t = useT();
  return (
    <div
      contentEditable={false}
      role="alert"
      data-testid="answer-error"
      data-error-kind={kind}
      className="answer-error"
    >
      <p>{t(`answer.error.${kind}`)}</p>
      <div className="mt-1 flex flex-wrap items-center gap-2">
        {errorKinds[kind].fixInSettings && (
          <button type="button" onClick={onOpenSettings} className="answer-notice-action">
            {t("answer.error.openSettings")}
          </button>
        )}
        {onRetry && (
          <button type="button" onClick={onRetry} className="answer-notice-action">
            {t("answer.error.retry")}
          </button>
        )}
        {details && (
          <details className="text-custom-xs text-red-800/80">
            <summary className="cursor-pointer">{t("answer.error.details")}</summary>
            <p className="mt-1 break-words">{details}</p>
          </details>
        )}
      </div>
    </div>
  );
}
