import { NodeViewContent, NodeViewWrapper, type ReactNodeViewProps } from "@tiptap/react";
import { Fragment, type ReactNode, useId, useMemo, useState } from "react";
import {
  type AnswerToolCall,
  type ApprovalRequest,
  BLOCK_ID_ATTRIBUTE,
  type ProviderErrorKind,
  SKILL_SCRIPT_LIMITS,
  type SkillScriptApprovalRequest,
  type ToolApprovalRequest,
} from "../../../core/api";
import { AnswerPhaseText } from "../answerPhase";
import { useAnswers } from "../answers";
import { useApprovals, waitingFor } from "../approvals";
import { PlugIcon, ScriptIcon, SkillIcon, StopIcon } from "../components/icons";
import { useT } from "../i18n";
import { type SettingsPage, useAppStore } from "../store";
import { ChevronRightSmallIcon, RegenerateSmallIcon } from "./icons";
import { useMindId } from "./mindContext";

const text = (value: unknown) => (typeof value === "string" ? value : null);

const errorKinds: Record<ProviderErrorKind, { fixInSettings: boolean }> = {
  auth: { fixInSettings: true },
  model: { fixInSettings: true },
  "consent-declined": { fixInSettings: true },
  "rate-limit": { fixInSettings: false },
  network: { fixInSettings: false },
  provider: { fixInSettings: false },
  // A local model's context window.
  "too-long": { fixInSettings: false },
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

const RUN_SCRIPT = "run_skill_script";
const SEARCH = "search_documents";

/** Items of the meta line, with a dot between each two. */
function Dotted({ items }: { items: ReactNode[] }) {
  return items.map((item, index) => (
    // biome-ignore lint/suspicious/noArrayIndexKey: the items are fixed slots of the meta line.
    <Fragment key={index}>
      {index > 0 && <span aria-hidden="true">·</span>}
      {item}
    </Fragment>
  ));
}

/**
 * An Answer: no box, its text at the Mind's text edge with the "Answer" label
 * in the left margin. Its meta line says which model wrote it, whether it is
 * still being written (a pulsing dot, and a stop button) or was stopped, what
 * it searched (it opens to the list), and offers to write it again. Under it,
 * ruled rows for the Skills, Connector calls and Skill scripts it used; when
 * it failed, what went wrong and how to fix it; then its text, ordinary
 * editable Blocks; and the approval cards of calls waiting for the User.
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
  const searches = toolCalls.filter((call) => call.tool === SEARCH);
  const [searchesOpen, setSearchesOpen] = useState(false);
  // Written again, it starts with its searches closed.
  if (searchesOpen && searches.length === 0) setSearchesOpen(false);
  const waiting = useApprovals((state) => state.waiting);
  const approvals = useMemo(() => waitingFor(waiting, answerId), [waiting, answerId]);

  const writeAgain = (discardEdits = false) => {
    if (answerId && questionId) void regenerate(mindId, answerId, questionId, discardEdits);
  };

  const meta: ReactNode[] = [];
  if (modelId) {
    meta.push(
      <span data-testid="answer-model" className="answer-model">
        {modelId}
      </span>,
    );
  }
  if (streaming) {
    const paused = approvals.length > 0;
    meta.push(
      <span
        data-testid="answer-writing"
        className={`answer-writing ${paused ? "answer-writing--waiting" : ""}`}
      >
        {paused ? t("approvals.answer.waiting") : <AnswerPhaseText answerId={answerId} />}
      </span>,
    );
  } else if (status === "stopped") {
    meta.push(<span data-testid="answer-stopped">{t("answer.status.stopped")}</span>);
  }
  if (searches.length > 0) {
    meta.push(
      <SearchesSummary
        searches={searches}
        open={searchesOpen}
        onToggle={() => setSearchesOpen((open) => !open)}
      />,
    );
  }

  return (
    <NodeViewWrapper
      className="answer-block"
      data-testid="answer"
      data-status={streaming ? "streaming" : status}
      data-answer-id={answerId ?? undefined}
    >
      <div contentEditable={false} className="answer-head">
        <div className="answer-meta">
          <span className="answer-label">{t("answer.label")}</span>
          <Dotted items={meta} />
          {streaming && answerId && (
            <button
              type="button"
              data-testid="answer-stop"
              onClick={() => stop(mindId, answerId)}
              className="answer-action"
            >
              <StopIcon className="size-3" />
              {t("answer.stop")}
            </button>
          )}
          {!streaming && questionId && (
            <button
              type="button"
              data-testid="answer-regenerate"
              onClick={() => writeAgain()}
              className="answer-action"
            >
              <RegenerateSmallIcon className="size-3" />
              {t("answer.regenerate")}
            </button>
          )}
        </div>
        {searchesOpen && <SearchList searches={searches} />}
      </div>

      <CallRows calls={toolCalls} approvals={approvals} />

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
          <span className="answer-notice-text">{t("answer.edited.body")}</span>
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

      {approvals.length > 0 && (
        <div contentEditable={false} className="answer-approvals">
          {/* Keyed by request, so a card keeps its state (the "Always run" warning) as its call comes in. */}
          {approvals.map((request) => (
            <ApprovalCard key={request.requestId} request={request} />
          ))}
        </div>
      )}
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
      <p className="answer-error-text">{t(`answer.error.${kind}`)}</p>
      <div className="answer-error-actions">
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
          <details className="answer-error-details">
            <summary>{t("answer.error.details")}</summary>
            <p>{details}</p>
          </details>
        )}
      </div>
    </div>
  );
}

/**
 * The Skills, Connector calls and Skill script runs an Answer made, as ruled
 * rows, Skills first. A call waiting for the User's approval has no row: its
 * approval card shows instead, after the Answer's text.
 */
function CallRows({ calls, approvals }: { calls: AnswerToolCall[]; approvals: ApprovalRequest[] }) {
  const waitingIds = new Set(approvals.map((request) => request.toolCallId));
  const rows = calls
    .filter(
      (call) =>
        !waitingIds.has(call.id) &&
        (call.source === "connector" || (call.source === "skill" && call.tool === RUN_SCRIPT)),
    )
    .map((call) =>
      call.source === "connector" ? (
        <ConnectorCall key={call.id} call={call} />
      ) : (
        <ScriptCall key={call.id} call={call} />
      ),
    );
  const skills = skillRows(calls);
  if (skills.length === 0 && rows.length === 0) return null;
  return (
    <div contentEditable={false} className="answer-calls">
      {skills.map(({ name, use, reads }) => (
        <SkillRow key={name} name={name} use={use} reads={reads} />
      ))}
      {rows}
    </div>
  );
}

/** The approval card for a request: a Connector's Tool, or a Skill script. */
function ApprovalCard({ request }: { request: ApprovalRequest }) {
  return request.subject.kind === "skill-script" ? (
    <ScriptApprovalCard request={request as SkillScriptApprovalRequest} />
  ) : (
    <ToolApprovalCard request={request as ToolApprovalRequest} />
  );
}

/** Stands for the name in a title, which is then drawn in mono (see `TitleWithCode`). */
const NAME_SLOT = "\u0000";

/** A title with a Tool or script name in it, the name in mono. */
function TitleWithCode({ title, code }: { title: string; code: string }) {
  const [before, after = ""] = title.split(NAME_SLOT);
  return (
    <span className="approval-title-text">
      {before}
      <code>{code}</code>
      {after}
    </span>
  );
}

/**
 * A Tool call waiting for the User: a ruled block in three sections. Which
 * Tool of which Connector, and whether it may change something or the User
 * asked to approve it every time; the arguments it would send; and three
 * choices. The Answer waits meanwhile.
 */
function ToolApprovalCard({ request }: { request: ToolApprovalRequest }) {
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
      <div className="approval-head">
        <p id={titleId} className="approval-title">
          <span aria-hidden="true" className="approval-dot" />
          <TitleWithCode
            title={t("approvals.card.title", { ...params, tool: NAME_SLOT })}
            code={request.tool}
          />
        </p>
        {request.title && request.title !== request.tool && (
          <p className="approval-explanation">{request.title}</p>
        )}
        <p className="approval-explanation">
          {request.readOnly
            ? t("approvals.card.readOnly", params)
            : t("approvals.card.changes", params)}
        </p>
      </div>
      <ApprovalArguments input={request.input} />
      <div className="approval-actions">
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
          className="approval-button approval-button--ghost"
        >
          {t("approvals.card.deny")}
        </button>
      </div>
    </div>
  );
}

/**
 * A Skill script waiting for the User, in the same three sections: which
 * script of which Skill, and that scripts run on this computer with no
 * sandbox; the Skill, script and arguments it would run with; and three
 * choices. "Always run" first shows a warning in the last section, which must
 * be confirmed. The Answer waits meanwhile.
 */
function ScriptApprovalCard({ request }: { request: SkillScriptApprovalRequest }) {
  const t = useT();
  const respond = useApprovals((state) => state.respond);
  const [warning, setWarning] = useState(false);
  const titleId = useId();
  const warningId = useId();
  const params = { skill: request.skill.name, script: request.script };
  return (
    <div
      contentEditable={false}
      role="alertdialog"
      aria-labelledby={titleId}
      data-testid="approval-card"
      data-tool={request.tool}
      data-kind="skill-script"
      className="approval-card"
    >
      <div className="approval-head">
        <p id={titleId} className="approval-title">
          <span aria-hidden="true" className="approval-dot" />
          <TitleWithCode
            title={t("scripts.card.title", { ...params, script: NAME_SLOT })}
            code={request.script}
          />
        </p>
        <p data-testid="approval-no-sandbox" className="approval-explanation">
          {t("scripts.card.noSandbox")}
        </p>
      </div>
      <div className="approval-details">
        <p className="approval-label">{t("scripts.card.wouldRun")}</p>
        <dl data-testid="approval-script" className="approval-args">
          <dt>{t("scripts.card.skill")}</dt>
          <dd data-testid="approval-script-skill">{request.skill.name}</dd>
          <dt>{t("scripts.card.script")}</dt>
          <dd data-testid="approval-script-path" className="font-mono">
            {request.script}
          </dd>
          <dt>{t("scripts.card.args")}</dt>
          <dd>
            {request.args.length === 0 ? (
              <span className="approval-empty">{t("scripts.card.noArgs")}</span>
            ) : (
              <ol className="list-none">
                {request.args.map((arg, index) => (
                  // biome-ignore lint/suspicious/noArrayIndexKey: arguments are positional and may repeat.
                  <li key={index} data-testid="approval-argument" className="font-mono">
                    {arg}
                  </li>
                ))}
              </ol>
            )}
          </dd>
        </dl>
      </div>
      {warning ? (
        <div
          role="alertdialog"
          aria-labelledby={warningId}
          data-testid="always-run-warning"
          className="approval-actions approval-warning"
        >
          <p id={warningId} className="approval-warning-title">
            {t("scripts.alwaysRun.title", params)}
          </p>
          <p className="approval-warning-body">{t("scripts.alwaysRun.body", params)}</p>
          <div className="approval-buttons">
            <button
              type="button"
              data-testid="always-run-confirm"
              onClick={() => respond(request.requestId, "always-allow", { riskAccepted: true })}
              className="approval-button approval-button--danger"
            >
              {t("scripts.alwaysRun.confirm", params)}
            </button>
            <button
              type="button"
              data-testid="always-run-cancel"
              onClick={() => setWarning(false)}
              className="approval-button approval-button--ghost"
            >
              {t("scripts.alwaysRun.cancel")}
            </button>
          </div>
        </div>
      ) : (
        <div className="approval-actions">
          <button
            type="button"
            data-testid="approval-allow-once"
            onClick={() => respond(request.requestId, "allow-once")}
            className="approval-button approval-button--primary"
          >
            {t("scripts.card.allowOnce")}
          </button>
          <button
            type="button"
            data-testid="approval-always-run"
            title={t("scripts.card.alwaysRunHint", params)}
            onClick={() => setWarning(true)}
            className="approval-button"
          >
            {t("scripts.card.alwaysRun")}
          </button>
          <button
            type="button"
            data-testid="approval-deny"
            onClick={() => respond(request.requestId, "deny")}
            className="approval-button approval-button--ghost"
          >
            {t("scripts.card.deny")}
          </button>
        </div>
      )}
    </div>
  );
}

/**
 * The arguments a call would send, readably: a key/value grid with the names
 * in mono, text as text (line breaks kept), anything else as indented JSON.
 */
function ApprovalArguments({ input }: { input: Record<string, unknown> }) {
  const t = useT();
  const entries = Object.entries(input);
  return (
    <div data-testid="approval-arguments" className="approval-details">
      <p className="approval-label">{t("approvals.card.arguments")}</p>
      {entries.length === 0 ? (
        <p className="approval-empty">{t("approvals.card.noArguments")}</p>
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
 * A run of a Skill script: which script of which Skill, how it went, on one
 * row that opens to show its arguments and what it wrote (what the model
 * was given), or why it couldn't run.
 */
function ScriptCall({ call }: { call: AnswerToolCall }) {
  const t = useT();
  const [expanded, setExpanded] = useState(false);
  const params = { skill: field(call, "skill"), script: field(call, "script") };
  const args = Array.isArray(call.input.args) ? call.input.args.map(String) : [];
  const run = call.script;
  const running = call.status === "running";
  const summary =
    call.approval === "denied"
      ? t("scripts.call.denied", params)
      : running && call.approval === "waiting"
        ? t("scripts.call.waiting", params)
        : running
          ? t("scripts.call.running", params)
          : run?.error
            ? t("scripts.call.cantRun", params)
            : run?.timedOut
              ? t("scripts.call.timedOut", params)
              : run && run.exitCode !== null
                ? t("scripts.call.exited", { ...params, code: run.exitCode })
                : t("scripts.call.stopped", params);
  const limit = SKILL_SCRIPT_LIMITS.maxOutputBytes;
  return (
    <div
      contentEditable={false}
      data-testid="answer-script-call"
      data-status={call.status}
      data-approval={call.approval}
      data-exit-code={run?.exitCode ?? undefined}
      className="answer-call"
    >
      <button
        type="button"
        aria-expanded={expanded}
        aria-label={`${summary}. ${t("scripts.call.details")}`}
        onClick={() => setExpanded((open) => !open)}
        className="answer-call-summary"
      >
        <ScriptIcon className={`answer-call-icon ${running ? "animate-pulse" : ""}`} />
        <span className="answer-call-text">{summary}</span>
      </button>
      {expanded && (
        <div className="answer-call-details" data-testid="answer-script-details">
          <div className="answer-call-detail-label">{t("scripts.call.args")}</div>
          <pre data-testid="answer-script-args">
            {args.length > 0 ? args.join(" ") : t("scripts.card.noArgs")}
          </pre>
          {run?.error && (
            <p data-testid="answer-script-error" className="answer-call-error">
              {run.error}
            </p>
          )}
          {run && !run.error && (
            <>
              <div className="answer-call-detail-label">{t("scripts.call.stdout")}</div>
              <pre data-testid="answer-script-stdout">{run.stdout || "—"}</pre>
              {run.stdoutTruncated && (
                <p className="answer-call-detail-label">
                  {t("scripts.call.stdoutTruncated", { bytes: limit })}
                </p>
              )}
              <div className="answer-call-detail-label">{t("scripts.call.stderr")}</div>
              <pre data-testid="answer-script-stderr">{run.stderr || "—"}</pre>
              {run.stderrTruncated && (
                <p className="answer-call-detail-label">
                  {t("scripts.call.stderrTruncated", { bytes: limit })}
                </p>
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
}

/**
 * A call an Answer made through a Connector: which Connector and Tool, and
 * the arguments it sent, on one row that opens to show them in full. What
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
      className="answer-call"
    >
      <button
        type="button"
        aria-expanded={expanded}
        aria-label={`${summary}. ${t("connectors.call.details")}`}
        onClick={() => setExpanded((open) => !open)}
        className="answer-call-summary"
      >
        <PlugIcon className={`answer-call-icon ${running ? "animate-pulse" : ""}`} />
        <span className="answer-call-text shrink-0">{summary}</span>
        {!expanded && compact !== "{}" && <code className="answer-call-args">{compact}</code>}
      </button>
      {expanded && (
        <div className="answer-call-details">
          <div className="answer-call-detail-label">
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
    <div contentEditable={false} data-testid="answer-sign-in-required" className="answer-call">
      <span className="answer-call-summary answer-call-summary--static">
        <PlugIcon className="answer-call-icon" />
        <span className="answer-call-text">
          {t("remoteConnectors.answer.signInRequired", { connector })}
        </span>
      </span>
    </div>
  );
}

/** A text field of what a Tool call was asked, or "". */
function field(call: AnswerToolCall, name: string): string {
  const value: unknown = call.input?.[name];
  return typeof value === "string" ? value : "";
}

interface SkillUse {
  name: string;
  /** How it was loaded: by the model, or chosen for the Question. */
  use: AnswerToolCall | null;
  /** The Skill's files it read. */
  reads: AnswerToolCall[];
}

/** The Skills an Answer used, in the order it first used them. */
function skillRows(calls: AnswerToolCall[]): SkillUse[] {
  const skills = new Map<string, SkillUse>();
  for (const call of calls) {
    // Script runs have rows of their own (`ScriptCall`).
    if (call.source !== "skill" || call.tool === RUN_SCRIPT) continue;
    const name = call.tool === "use_skill" ? field(call, "name") : field(call, "skill");
    const entry = skills.get(name) ?? { name, use: null, reads: [] };
    if (call.tool === "use_skill") entry.use ??= call;
    else entry.reads.push(call);
    skills.set(name, entry);
  }
  return [...skills.values()];
}

/** A Skill an Answer used, loaded by the model or chosen for the Question, with the files it read. */
function SkillRow({ name, use, reads }: SkillUse) {
  const t = useT();
  const status = use?.status ?? "done";
  const summary =
    status === "running"
      ? t("skills.answer.loading", { name })
      : status === "failed"
        ? t("skills.answer.failed", { name })
        : t("skills.answer.used", { name });
  return (
    <div
      data-testid="answer-skill"
      data-skill-name={name}
      data-status={status}
      data-forced={use?.forced === true}
      className={`answer-call ${status === "failed" ? "answer-call--failed" : ""}`}
    >
      <p className="answer-call-summary answer-call-summary--static">
        <SkillIcon className={`answer-call-icon ${status === "running" ? "animate-pulse" : ""}`} />
        <span className="answer-call-text">{summary}</span>
        {use?.forced && <span className="answer-call-note"> · {t("skills.answer.forced")}</span>}
      </p>
      {reads.length > 0 && (
        <ul className="answer-call-files">
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
}

/** The searches an Answer ran, summed up on its meta line; it opens the list. */
function SearchesSummary({
  searches,
  open,
  onToggle,
}: {
  searches: AnswerToolCall[];
  open: boolean;
  onToggle(): void;
}) {
  const t = useT();
  const running = searches.some((call) => call.status === "running");
  const failed = searches.every((call) => call.status === "failed");
  const summary = running ? t("search.running") : failed ? t("search.failed") : t("search.done");
  return (
    <span data-testid="answer-tools" className="answer-searches">
      <button
        type="button"
        data-testid="answer-tools-toggle"
        aria-expanded={open}
        aria-label={`${summary}. ${t("search.details")}`}
        onClick={onToggle}
        className={`answer-meta-button ${running ? "animate-pulse" : ""}`}
      >
        <span>{summary}</span>
        {searches.length > 1 && <span>· {t("search.count", { count: searches.length })}</span>}
        <ChevronRightSmallIcon className="answer-meta-chevron" />
      </button>
    </span>
  );
}

/** What each search was for, and what it found. */
function SearchList({ searches }: { searches: AnswerToolCall[] }) {
  const t = useT();
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
    <ul className="answer-searches-list">
      {searches.map((call) => {
        const query = typeof call.input.query === "string" ? call.input.query : "";
        const found = results(call);
        return (
          <li key={call.id} data-testid="answer-tool-call" data-status={call.status}>
            {t("search.query", { query })}
            {found && <span className="answer-searches-found"> · {found}</span>}
          </li>
        );
      })}
    </ul>
  );
}
