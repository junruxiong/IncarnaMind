import { NodeViewContent, NodeViewWrapper, type ReactNodeViewProps } from "@tiptap/react";
import { useState } from "react";
import { type AnswerToolCall, BLOCK_ID_ATTRIBUTE, type ProviderErrorKind } from "../../../core/api";
import { useAnswers } from "../answers";
import { PlugIcon, RegenerateIcon, SearchIcon, StopIcon } from "../components/icons";
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
  // The experimental ChatGPT plan provider.
  "not-signed-in": { fixInSettings: true },
  "plan-limit": { fixInSettings: false },
  blocked: { fixInSettings: true },
};

const isErrorKind = (value: unknown): value is ProviderErrorKind =>
  typeof value === "string" && Object.hasOwn(errorKinds, value);

/** The Tool calls the core stored on the Answer (a JSON array), or none if unreadable. */
function toolCallsOf(value: unknown): AnswerToolCall[] {
  if (typeof value !== "string") return [];
  try {
    const parsed: unknown = JSON.parse(value);
    return Array.isArray(parsed)
      ? parsed.filter(
          (call): call is AnswerToolCall =>
            typeof call === "object" && call !== null && typeof call.id === "string",
        )
      : [];
  } catch {
    return [];
  }
}

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
  const toolCalls = toolCallsOf(node.attrs.toolCalls);

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

      {toolCalls.length > 0 && <ToolCalls calls={toolCalls} />}

      {text(node.attrs.citationSupport) === "none" && (
        <p contentEditable={false} data-testid="answer-no-citations" className="answer-info">
          {t("citation.unsupported")}
        </p>
      )}

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

/** The Tools an Answer called: its searches as one line, then a card for each Connector call. */
function ToolCalls({ calls }: { calls: AnswerToolCall[] }) {
  const searches = calls.filter((call) => call.source !== "connector");
  const connectorCalls = calls.filter((call) => call.source === "connector");
  return (
    <>
      {searches.length > 0 && <Searches searches={searches} />}
      {connectorCalls.map((call) => (
        <ConnectorCall key={call.id} call={call} />
      ))}
    </>
  );
}

/**
 * A call an Answer made through a Connector: which Connector and Tool, and
 * the arguments it sent, on one line that opens to show them in full. What
 * came back went to the model; it isn't a Passage, so it is never a Citation.
 */
function ConnectorCall({ call }: { call: AnswerToolCall }) {
  const t = useT();
  const [expanded, setExpanded] = useState(false);
  const params = { connector: call.connector?.name ?? "", tool: call.tool };
  const running = call.status === "running";
  const summary = running
    ? t("connectors.call.running", params)
    : call.status === "failed"
      ? t("connectors.call.failed", params)
      : t("connectors.call.done", params);
  const compact = JSON.stringify(call.input);
  return (
    <div
      contentEditable={false}
      data-testid="answer-connector-call"
      data-status={call.status}
      className="answer-tools"
    >
      <button
        type="button"
        aria-expanded={expanded}
        aria-label={`${summary}. ${t("connectors.call.details")}`}
        onClick={() => setExpanded((open) => !open)}
        className="answer-tools-summary max-w-full"
      >
        <PlugIcon className={`size-3.5 shrink-0 ${running ? "animate-pulse" : ""}`} />
        <span className="shrink-0">{summary}</span>
        {!expanded && compact !== "{}" && <code className="answer-tool-args">{compact}</code>}
      </button>
      {expanded && (
        <div className="answer-tool-args-full">
          <div className="mb-0.5 font-sans text-gray-400">{t("connectors.call.arguments")}</div>
          <pre data-testid="answer-connector-arguments">{JSON.stringify(call.input, null, 2)}</pre>
        </div>
      )}
    </div>
  );
}

/**
 * The searches an Answer ran, small and out of the way: one line saying it
 * searched the User's Documents, which opens to show what it searched for.
 */
function Searches({ searches }: { searches: AnswerToolCall[] }) {
  const t = useT();
  const [expanded, setExpanded] = useState(false);
  const running = searches.some((call) => call.status === "running");
  const failed = searches.every((call) => call.status === "failed");
  const summary = running ? t("search.running") : failed ? t("search.failed") : t("search.done");
  const results = (call: AnswerToolCall) =>
    call.status === "failed"
      ? t("search.failed")
      : call.resultCount === null
        ? null
        : call.resultCount === 0
          ? t("search.noResults")
          : call.resultCount === 1
            ? t("search.results.one")
            : t("search.results", { count: call.resultCount });
  return (
    <div contentEditable={false} data-testid="answer-tools" className="answer-tools">
      <button
        type="button"
        data-testid="answer-tools-toggle"
        aria-expanded={expanded}
        aria-label={`${summary}. ${t("search.details")}`}
        onClick={() => setExpanded((open) => !open)}
        className="answer-tools-summary"
      >
        <SearchIcon className={`size-3.5 ${running ? "animate-pulse" : ""}`} />
        <span>{summary}</span>
        {searches.length > 1 && <span>· {t("search.count", { count: searches.length })}</span>}
      </button>
      {expanded && (
        <ul className="answer-tools-list">
          {searches.map((call) => {
            const query = typeof call.input.query === "string" ? call.input.query : "";
            const found = results(call);
            return (
              <li key={call.id} data-testid="answer-tool-call" data-status={call.status}>
                {t("search.query", { query })}
                {found && <span className="text-gray-400"> · {found}</span>}
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}
