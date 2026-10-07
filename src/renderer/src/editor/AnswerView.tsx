import { NodeViewContent, NodeViewWrapper, type ReactNodeViewProps } from "@tiptap/react";
import { useState } from "react";
import { type AnswerToolCall, BLOCK_ID_ATTRIBUTE, type ProviderErrorKind } from "../../../core/api";
import { useAnswers } from "../answers";
import { RegenerateIcon, SearchIcon, SkillIcon, StopIcon } from "../components/icons";
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

      {toolCalls.length > 0 && <SkillCalls calls={toolCalls} />}
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

/** A text field of what a Tool call was asked, or "". */
function field(call: AnswerToolCall, name: string): string {
  const value: unknown = call.input?.[name];
  return typeof value === "string" ? value : "";
}

/**
 * The Skills an Answer used, one card each: loaded by the model or chosen
 * for the Question, and the Skill's files it read.
 */
function SkillCalls({ calls }: { calls: AnswerToolCall[] }) {
  const t = useT();
  const skills = new Map<string, { use: AnswerToolCall | null; reads: AnswerToolCall[] }>();
  for (const call of calls) {
    if (call.source !== "skill") continue;
    const name = call.tool === "use_skill" ? field(call, "name") : field(call, "skill");
    const entry = skills.get(name) ?? { use: null, reads: [] };
    if (call.tool === "use_skill") entry.use ??= call;
    else entry.reads.push(call);
    skills.set(name, entry);
  }
  if (skills.size === 0) return null;
  return (
    <div contentEditable={false} className="answer-skills">
      {[...skills].map(([name, { use, reads }]) => {
        const status = use?.status ?? "done";
        const summary =
          status === "running"
            ? t("skills.answer.loading", { name })
            : status === "failed"
              ? t("skills.answer.failed", { name })
              : t("skills.answer.used", { name });
        return (
          <div
            key={name}
            data-testid="answer-skill"
            data-skill-name={name}
            data-status={status}
            data-forced={use?.forced === true}
            className={`answer-skill ${status === "failed" ? "answer-skill--failed" : ""}`}
          >
            <p className="flex items-center gap-1">
              <SkillIcon
                className={`size-3.5 shrink-0 ${status === "running" ? "animate-pulse" : ""}`}
              />
              <span className="truncate">{summary}</span>
              {use?.forced && (
                <span className="shrink-0 text-gray-400"> · {t("skills.answer.forced")}</span>
              )}
            </p>
            {reads.length > 0 && (
              <ul className="answer-skill-files">
                {reads.map((read) => (
                  <li key={read.id} data-testid="answer-skill-file" data-status={read.status}>
                    {read.status === "failed"
                      ? t("skills.answer.readFailed", { path: field(read, "path") })
                      : t("skills.answer.read", { path: field(read, "path") })}
                  </li>
                ))}
              </ul>
            )}
          </div>
        );
      })}
    </div>
  );
}

/**
 * The searches an Answer ran, small and out of the way: one line saying it
 * searched the User's Documents, which opens to show what it searched for.
 */
function ToolCalls({ calls }: { calls: AnswerToolCall[] }) {
  const t = useT();
  const [expanded, setExpanded] = useState(false);
  const searches = calls.filter((call) => call.tool === "search_documents");
  if (searches.length === 0) return null;
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
