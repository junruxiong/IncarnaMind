import { NodeViewContent, NodeViewWrapper, type ReactNodeViewProps } from "@tiptap/react";
import { useId, useMemo, useState } from "react";
import {
  type AnswerToolCall,
  type ApprovalRequest,
  BLOCK_ID_ATTRIBUTE,
  type ProviderErrorKind,
} from "../../../core/api";
import { useAnswers } from "../answers";
import { useApprovals, waitingFor } from "../approvals";
import { PlugIcon, RegenerateIcon, SearchIcon, SkillIcon, StopIcon } from "../components/icons";
import { useT } from "../i18n";
import { type SettingsPage, useAppStore } from "../store";
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
  const waiting = useApprovals((state) => state.waiting);
  const approvals = useMemo(() => waitingFor(waiting, answerId), [waiting, answerId]);

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
        {streaming && (
          <span className="answer-writing" data-testid="answer-writing">
            {approvals.length > 0 ? t("approvals.answer.waiting") : t("answer.status.streaming")}
          </span>
        )}
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
      {(toolCalls.length > 0 || approvals.length > 0) && (
        <ToolCalls calls={toolCalls} approvals={approvals} />
      )}

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
  onOpenSettings(page: SettingsPage): void;
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
          <button
            type="button"
            // A declined data flow is allowed again on the Privacy page.
            onClick={() => onOpenSettings(kind === "consent-declined" ? "privacy" : "general")}
            className="answer-notice-action"
          >
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

/**
 * The searches and Connector calls an Answer made: its searches as one line,
 * then a card for each Connector call. A call waiting for the User's approval
 * shows the approval card instead, in every window. Skills have their own
 * cards (`SkillCalls`).
 */
function ToolCalls({
  calls,
  approvals,
}: {
  calls: AnswerToolCall[];
  approvals: ApprovalRequest[];
}) {
  const searches = calls.filter((call) => call.tool === "search_documents");
  const connectorCalls = calls.filter((call) => call.source === "connector");
  const approvalFor = (call: AnswerToolCall) =>
    approvals.find((request) => request.toolCallId === call.id);
  // A request can arrive before its call is written into the Answer.
  const unmatched = approvals.filter((request) =>
    connectorCalls.every((call) => call.id !== request.toolCallId),
  );
  return (
    <>
      {searches.length > 0 && <Searches searches={searches} />}
      {connectorCalls.map((call) => {
        const request = approvalFor(call);
        return request ? (
          <ApprovalCard key={call.id} request={request} />
        ) : (
          <ConnectorCall key={call.id} call={call} />
        );
      })}
      {unmatched.map((request) => (
        <ApprovalCard key={request.requestId} request={request} />
      ))}
    </>
  );
}

/**
 * A Tool call waiting for the User: which Tool of which Connector, whether it
 * may change something or the User asked to approve it every time, the
 * arguments it would send, and three choices. The Answer waits meanwhile.
 */
function ApprovalCard({ request }: { request: ApprovalRequest }) {
  const t = useT();
  const respond = useApprovals((state) => state.respond);
  const titleId = useId();
  const params = { connector: request.connector.name, tool: request.tool };
  const decide = (decision: Parameters<typeof respond>[1]) => respond(request.requestId, decision);
  return (
    <div
      contentEditable={false}
      role="alertdialog"
      aria-labelledby={titleId}
      data-testid="approval-card"
      data-tool={request.tool}
      data-read-only={request.readOnly}
      className="approval-card"
    >
      <p id={titleId} className="flex items-center gap-1.5 font-medium">
        <PlugIcon className="size-3.5 shrink-0" />
        <span className="min-w-0">{t("approvals.card.title", params)}</span>
      </p>
      {request.title && request.title !== request.tool && (
        <p className="mt-0.5 text-amber-900/80">{request.title}</p>
      )}
      <p className="mt-0.5 text-amber-900/80">
        {request.readOnly
          ? t("approvals.card.readOnly", params)
          : t("approvals.card.changes", params)}
      </p>
      <ApprovalArguments input={request.input} />
      <div className="mt-2 flex flex-wrap gap-1.5">
        <button
          type="button"
          data-testid="approval-allow-once"
          onClick={() => decide("allow-once")}
          className="approval-button approval-button--primary"
        >
          {t("approvals.card.allowOnce")}
        </button>
        <button
          type="button"
          data-testid="approval-always-allow"
          title={t("approvals.card.alwaysAllowHint", params)}
          onClick={() => decide("always-allow")}
          className="approval-button"
        >
          {t("approvals.card.alwaysAllow")}
        </button>
        <button
          type="button"
          data-testid="approval-deny"
          onClick={() => decide("deny")}
          className="approval-button"
        >
          {t("approvals.card.deny")}
        </button>
      </div>
    </div>
  );
}

/**
 * The arguments a call would send, readably: each argument on its own row,
 * text as text (line breaks kept), anything else as indented JSON.
 */
function ApprovalArguments({ input }: { input: Record<string, unknown> }) {
  const t = useT();
  const entries = Object.entries(input);
  return (
    <div data-testid="approval-arguments" className="mt-1.5">
      <p className="text-custom-xs text-amber-900/70">{t("approvals.card.arguments")}</p>
      {entries.length === 0 ? (
        <p className="text-custom-xs text-amber-900/70">{t("approvals.card.noArguments")}</p>
      ) : (
        <dl className="approval-args">
          {entries.map(([name, value]) => (
            <div key={name} data-testid="approval-argument" className="contents">
              <dt>{name}</dt>
              <dd>
                {typeof value === "string" ? (
                  value
                ) : (
                  <pre className="whitespace-pre-wrap">{JSON.stringify(value, null, 2)}</pre>
                )}
              </dd>
            </div>
          ))}
        </dl>
      )}
    </div>
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
  if (call.signInRequired) return <SignInRequired connector={params.connector} />;
  const summary =
    call.approval === "denied"
      ? t("approvals.call.denied", params)
      : running && call.approval === "waiting"
        ? t("approvals.call.waiting", params)
        : running
          ? t("connectors.call.running", params)
          : call.status === "failed"
            ? t("connectors.call.failed", params)
            : call.approval === "allowed"
              ? t("approvals.call.allowed", params)
              : t("connectors.call.done", params);
  const compact = JSON.stringify(call.input);
  return (
    <div
      contentEditable={false}
      data-testid="answer-connector-call"
      data-status={call.status}
      data-approval={call.approval}
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
          <div className="mb-0.5 font-sans text-gray-400">
            {call.approval === "denied"
              ? t("approvals.call.notSent")
              : t("connectors.call.arguments")}
          </div>
          <pre data-testid="answer-connector-arguments">{JSON.stringify(call.input, null, 2)}</pre>
        </div>
      )}
    </div>
  );
}

/** A remote Connector the Answer couldn't use: it waits for the User to sign in again. */
function SignInRequired({ connector }: { connector: string }) {
  const t = useT();
  return (
    <div contentEditable={false} data-testid="answer-sign-in-required" className="answer-tools">
      <span className="answer-tools-summary max-w-full">
        <PlugIcon className="size-3.5 shrink-0" />
        <span>{t("remoteConnectors.answer.signInRequired", { connector })}</span>
      </span>
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
