import { type ReactNode, useEffect, useLayoutEffect, useRef, useState } from "react";
import type {
  AnswerToolCall,
  AskedFrom,
  ConversationAnswer,
  ConversationQuestion,
} from "../../../../core/api";
import { citationLocation, formatLocation } from "../../../../shared/locations";
import {
  type FoldedLine,
  foldedLine,
  type LiveStep,
  liveStep,
  liveStepKey,
  type StepRow,
  type StepSource,
  stepRows,
} from "../../answerSteps";
import { useAnswers } from "../../answers";
import { stepsScope } from "../../assistantContext";
import { NO_SCOPE } from "../../composer";
import { CallRow, SkillRow } from "../../editor/AnswerView";
import { CheckMarkIcon, RegenerateSmallIcon } from "../../editor/icons";
import { useT } from "../../i18n";
import { useMeetings } from "../../meetings";
import { useModelNames } from "../../modelNames";
import type { Translate } from "../../objects";
import { openCited, quoteSaysSo } from "../../openCited";
import { scopeChips } from "../../scope";
import { useAppStore } from "../../store";
import { SearchIcon } from "../icons";
import {
  CalculatorLineIcon,
  ChevronRightLineIcon,
  CloseLineIcon,
  DocumentLineIcon,
  MindLineIcon,
  SparkLineIcon,
} from "../lineIcons";
import { ProblemLine } from "../ProblemLine";

/** The least time a live step stays up while another replaces it, so the line never flickers. */
const MIN_STEP_MS = 1000;
/** Sources shown under a search before "+N more". */
const SHOWN_SOURCES = 3;

/**
 * An Answer's steps in the assistant (#391), at the top of the Answer: while
 * it is written, the step it is on; once written, one folded line, as
 * Claude's and ChatGPT's "Thought for 5s ›", which opens, at once, to a
 * timeline of what it did, with Regenerate at its end.
 */
export function AnswerSteps({
  answer,
  question,
  streaming,
  waitingForApproval,
  onRegenerate,
}: {
  answer: ConversationAnswer;
  question: ConversationQuestion | undefined;
  streaming: boolean;
  waitingForApproval: boolean;
  onRegenerate(): void;
}) {
  const t = useT();
  const phase = useAnswers((state) => state.phases[answer.id]);
  const askedFrom = question?.askedFrom ?? null;
  const meeting = useAskedInMeeting(question);
  const scopeOf = useStepsScopeOf(question);
  const [open, setOpen] = useState(false);
  // Written again, it starts folded.
  if (streaming && open) setOpen(false);
  if (streaming) {
    return <LiveLine step={liveStep({ answer, phase, waitingForApproval })} scope={scopeOf(0)} />;
  }
  const line = foldedLine({ answer, askedFrom, meeting });
  if (!line) return null;
  return (
    <div className="answer-steps">
      <FoldedToggle
        line={line}
        scopeOf={scopeOf}
        open={open}
        onToggle={() => setOpen((was) => !was)}
      />
      {open && (
        <>
          <Timeline
            rows={stepRows({ answer, askedFrom, meeting })}
            scope={scopeOf(0)}
            onRegenerate={onRegenerate}
          />
          <button
            type="button"
            data-testid="answer-regenerate"
            onClick={onRegenerate}
            className="answer-steps-regenerate"
          >
            <RegenerateSmallIcon className="size-3.5" />
            {t("answer.regenerate")}
          </button>
        </>
      )}
    </div>
  );
}

/**
 * The line of an Answer on its way or being written: its step, with the
 * pulsing accent dot, or what it waits for, with the attention dot. `scope`
 * is what its Question searches (`stepsScope`).
 */
export function LiveLine({ step, scope }: { step: LiveStep; scope: string }) {
  const t = useT();
  const shown = useHeldStep(step);
  const label = liveText(t, shown, scope);
  const waiting = shown.kind === "waiting" || shown.kind === "consent";
  return (
    <p
      role="status"
      data-testid="answer-steps-live"
      data-step={shown.kind}
      className="answer-steps-line answer-steps-live"
    >
      <span
        aria-hidden="true"
        className={`answer-steps-dot ${waiting ? "answer-steps-dot--waiting" : ""}`}
      />
      <span data-testid="answer-writing" title={label} className="answer-steps-text">
        {label}
      </span>
    </p>
  );
}

/**
 * The step to show: a new one shows once the one before has been up a
 * second, then the latest; one that waits for the User shows at once.
 */
function useHeldStep(step: LiveStep): LiveStep {
  const [shown, setShown] = useState(step);
  const since = useRef(Date.now());
  const latest = useRef(step);
  latest.current = step;
  const key = liveStepKey(step);
  const shownKey = liveStepKey(shown);
  useEffect(() => {
    if (key === shownKey) return;
    const show = () => {
      since.current = Date.now();
      setShown(latest.current);
    };
    const waits = latest.current.kind === "waiting" || latest.current.kind === "consent";
    const left = MIN_STEP_MS - (Date.now() - since.current);
    if (waits || left <= 0) {
      show();
      return;
    }
    const timer = setTimeout(show, left);
    return () => clearTimeout(timer);
  }, [key, shownKey]);
  // The same step with a new count (Citations) shows the new one.
  return key === shownKey ? step : shown;
}

function liveText(t: Translate, step: LiveStep, scope: string): string {
  switch (step.kind) {
    case "waiting":
      return t("approvals.answer.waiting");
    case "consent":
      return t("answer.phase.waiting-for-consent");
    case "loading":
      return t("answer.phase.loading");
    case "reading":
      return t("steps.live.reading", { mind: mindLabel(t, step.mind) });
    case "searching":
      return t("steps.live.searching", { scope });
    case "call":
      return runningText(t, step.call);
    case "writing":
      return t("answer.phase.writing");
    case "checking":
      return step.count === 1
        ? t("steps.live.checking.one")
        : t("steps.live.checking", { count: step.count });
    case "checking-figures":
      return t("answer.phase.checking-figures");
  }
}

/** A text field of what a Tool call was asked, or "". */
const field = (call: AnswerToolCall, name: string): string => {
  const value: unknown = call.input?.[name];
  return typeof value === "string" ? value : "";
};

/** What a call that runs is doing, in the words its row uses. */
function runningText(t: Translate, call: AnswerToolCall): string {
  if (call.source === "connector") {
    return t("connectors.call.running", { connector: call.connector?.name ?? "", tool: call.tool });
  }
  if (call.source === "sources") return t("sources.call.running", { query: field(call, "query") });
  if (call.source === "web") {
    return call.tool === "search_web"
      ? t("web.call.searching", { query: field(call, "query") })
      : t("web.call.reading", { address: field(call, "url") });
  }
  if (call.tool === "run_skill_script") {
    return t("scripts.call.running", {
      skill: field(call, "skill"),
      script: field(call, "script"),
    });
  }
  const skill = call.tool === "use_skill" ? field(call, "name") : field(call, "skill");
  return t("skills.answer.loading", { name: skill });
}

const mindLabel = (t: Translate, mind: AskedFrom) => mind.name || t("mind.untitled");

/** A step about the Mind: the one read, or the one asked from and not read. */
type MindStep = Extract<FoldedLine, { kind: "read" | "asked" }>;

/**
 * "Read “{Mind}”", "Read part of “{Mind}”" when it was cut to fit, or "Asked from
 * {Mind}"; with `name` where the Mind's name goes.
 */
function mindStepText(t: Translate, step: MindStep, name = mindLabel(t, step.mind)): string {
  if (step.kind === "asked") {
    return step.mind.deleted ? t("steps.asked.deleted") : t("steps.asked", { mind: name });
  }
  if (step.mind.deleted) return t("steps.read.deleted");
  return t(step.mind.partial ? "steps.read.part" : "steps.read", { mind: name });
}

/** A share read as a whole percent, at least 1 and at most 99: "about 40%" is part, never none or all. */
const percentOf = (share: number) => Math.min(99, Math.max(1, Math.round(share * 100)));

function sourcesText(t: Translate, count: number): string {
  if (count === 0) return t("steps.sources.none");
  return count === 1 ? t("steps.sources.one") : t("steps.sources", { count });
}

function foldedText(
  t: Translate,
  line: FoldedLine,
  scope: (extra: number) => string,
  modelOf: (modelId: string) => string,
): string {
  switch (line.kind) {
    case "searched":
      return t("steps.searched", {
        scope: scope(line.extraScopes),
        sources: sourcesText(t, line.sources),
      });
    case "searched-web":
      return t("steps.searchedWeb", { sources: sourcesText(t, line.sources) });
    case "listened":
      return t("steps.listened", { sources: sourcesText(t, line.sources) });
    case "read":
    case "asked":
      return mindStepText(t, line);
    case "answered":
      return t("steps.answered", { model: modelOf(line.model) });
  }
}

/** Stands for the Mind's name in a line, to split the line around it. */
const NAME = "\u0001";

/**
 * The folded line: a click, Enter or Space opens and folds it. Its tooltip is
 * the whole line. When it names the Mind read (or asked from), its timeline
 * leaves that step out, and the name goes to the Mind, as the step's row does.
 */
function FoldedToggle({
  line,
  scopeOf,
  open,
  onToggle,
}: {
  line: FoldedLine;
  scopeOf(extra: number): string;
  open: boolean;
  onToggle(): void;
}) {
  const t = useT();
  const nameOf = useModelNames();
  const label = foldedText(t, line, scopeOf, (id) => nameOf(id).name);
  const step = line.kind === "read" || line.kind === "asked" ? line : null;
  const mind = step?.mind;
  const Icon = mind ? MindLineIcon : line.kind === "answered" ? SparkLineIcon : SearchIcon;
  const parts = step && !step.mind.deleted ? mindStepText(t, step, NAME).split(NAME) : [];
  if (mind && parts.length === 2) {
    const [before, after] = parts;
    return (
      <div
        data-testid="answer-steps-line"
        title={label}
        className="answer-steps-line answer-steps-split"
      >
        <button
          type="button"
          data-testid="answer-steps"
          data-kind={line.kind}
          aria-expanded={open}
          aria-label={label}
          onClick={onToggle}
          className="answer-steps-toggle answer-steps-part"
        >
          <Icon className="answer-steps-icon" />
          {before}
        </button>
        <button
          type="button"
          data-testid="answer-steps-mind"
          onClick={() => useAppStore.getState().openMind(mind.mindId)}
          className="answer-steps-mind answer-step-link"
        >
          {mindLabel(t, mind)}
        </button>
        <button
          type="button"
          tabIndex={-1}
          aria-hidden="true"
          aria-expanded={open}
          onClick={onToggle}
          className={`answer-steps-toggle answer-steps-part ${after ? "" : "answer-steps-part--end"}`}
        >
          {after}
          <ChevronRightLineIcon className="answer-steps-chevron" />
        </button>
      </div>
    );
  }
  return (
    <button
      type="button"
      data-testid="answer-steps"
      data-kind={line.kind}
      aria-expanded={open}
      title={line.kind === "answered" ? `${label} · ${line.model}` : label}
      onClick={onToggle}
      className="answer-steps-line answer-steps-toggle"
    >
      <Icon className="answer-steps-icon" />
      <span className="answer-steps-text">{label}</span>
      <ChevronRightLineIcon aria-hidden="true" className="answer-steps-chevron" />
    </button>
  );
}

/** The unfolded steps, one row each, their icons joined by a rule. */
function Timeline({
  rows,
  scope,
  onRegenerate,
}: {
  rows: StepRow[];
  scope: string;
  onRegenerate(): void;
}) {
  const t = useT();
  return (
    <ol aria-label={t("steps.label")} data-testid="answer-timeline" className="answer-timeline">
      {rows.map((row, index) => (
        <StepItem
          // biome-ignore lint/suspicious/noArrayIndexKey: the rows are in a fixed order for one Answer.
          key={index}
          row={row}
          scope={scope}
          onRegenerate={onRegenerate}
        />
      ))}
    </ol>
  );
}

function StepItem({
  row,
  scope,
  onRegenerate,
}: {
  row: StepRow;
  scope: string;
  onRegenerate(): void;
}) {
  const t = useT();
  const shown = useModelNames()(
    row.kind === "reasoning" || row.kind === "answered" ? row.model : "",
  );
  switch (row.kind) {
    case "reasoning":
      return (
        <Step kind={row.kind} icon={<SparkLineIcon />}>
          <p className="answer-step-title" title={shown.id}>
            {t("steps.thought", { model: shown.name, seconds: row.seconds })}
          </p>
          <ReasoningSummary summary={row.summary} />
        </Step>
      );
    case "answered":
      return (
        <Step kind={row.kind} icon={<SparkLineIcon />}>
          <p className="answer-step-title" title={shown.id}>
            {row.seconds === null
              ? t("steps.answered", { model: shown.name })
              : t("steps.answered.in", { model: shown.name, seconds: row.seconds })}
          </p>
        </Step>
      );
    case "read":
    case "asked": {
      const label = mindStepText(t, row);
      return (
        <Step kind={row.kind} icon={<MindLineIcon />}>
          {row.mind.deleted ? (
            <p className="answer-step-title">{label}</p>
          ) : (
            <button
              type="button"
              data-testid={`answer-step-${row.kind}`}
              onClick={() => useAppStore.getState().openMind(row.mind.mindId)}
              className="answer-step-title answer-step-link"
            >
              {label}
            </button>
          )}
          {row.kind === "read" && row.share !== null && (
            <p data-testid="answer-step-read-share" className="answer-step-detail">
              {t("steps.read.share", { percent: percentOf(row.share) })}
            </p>
          )}
        </Step>
      );
    }
    case "search":
      return <SearchStep row={row} scope={scope} />;
    case "failed-search":
      return (
        <Step kind={row.kind} icon={<CloseLineIcon />}>
          <FailedSearch call={row.call} scope={scope} onRegenerate={onRegenerate} />
        </Step>
      );
    case "call":
      return (
        <li
          data-testid="answer-step"
          data-step={row.kind}
          className="answer-step answer-step--call"
        >
          <CallRow call={row.call} />
        </li>
      );
    case "skill":
      return (
        <li
          data-testid="answer-step"
          data-step={row.kind}
          className="answer-step answer-step--call"
        >
          <SkillRow name={row.name} use={row.use} reads={row.reads} />
        </li>
      );
    case "calculated":
      return (
        <Step kind={row.kind} icon={<CalculatorLineIcon />}>
          <p data-testid="answer-figures" className="answer-step-title">
            {row.count === 1
              ? t("answer.figures.calculated.one")
              : t("answer.figures.calculated", { count: row.count })}
          </p>
        </Step>
      );
    case "check":
      return <CheckStep count={row.count} notFound={row.notFound} />;
  }
}

function Step({
  kind,
  icon,
  children,
}: {
  kind: StepRow["kind"];
  icon: ReactNode;
  children: ReactNode;
}) {
  return (
    <li data-testid="answer-step" data-step={kind} className="answer-step">
      <span aria-hidden="true" className="answer-step-icon">
        {icon}
      </span>
      <div className="answer-step-body">{children}</div>
    </li>
  );
}

/** The model's reasoning summary, cut at three lines; Show more only when it is cut. */
function ReasoningSummary({ summary }: { summary: string }) {
  const t = useT();
  const text = useRef<HTMLParagraphElement>(null);
  const [whole, setWhole] = useState(false);
  const [cut, setCut] = useState(false);
  useLayoutEffect(() => {
    const element = text.current;
    if (!element || whole) return;
    const measure = () => setCut(element.scrollHeight > element.clientHeight + 1);
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(element);
    return () => observer.disconnect();
  }, [whole]);
  return (
    <>
      <p
        ref={text}
        data-testid="answer-reasoning"
        data-whole={whole || undefined}
        className="answer-step-detail answer-step-reasoning"
      >
        {plainSummary(summary)}
      </p>
      {(cut || whole) && (
        <button
          type="button"
          data-testid="answer-reasoning-more"
          aria-expanded={whole}
          onClick={() => setWhole((was) => !was)}
          className="answer-step-more"
        >
          {t(whole ? "steps.showLess" : "steps.showMore")}
        </button>
      )}
    </>
  );
}

/** A summary's Markdown emphasis left out: providers set its headings in bold. */
const plainSummary = (summary: string) => summary.replace(/\*\*(.+?)\*\*/g, "$1");

/** A search, or several in a row: what was searched for, what was found and used, and its sources. */
function SearchStep({ row, scope }: { row: Extract<StepRow, { kind: "search" }>; scope: string }) {
  const t = useT();
  const [all, setAll] = useState(false);
  const queries = row.calls.map((call) => field(call, "query"));
  const shown = all ? row.sources : row.sources.slice(0, SHOWN_SOURCES);
  const more = row.sources.length - shown.length;
  return (
    <Step kind={row.kind} icon={<SearchIcon />}>
      <p className="answer-step-title">
        {queries.length > 1
          ? t("steps.search.times", { scope, count: queries.length })
          : t("steps.search", { scope, query: queries[0] ?? "" })}
      </p>
      {queries.length > 1 && (
        <ul className="answer-step-queries">
          {queries.map((query, index) => (
            // biome-ignore lint/suspicious/noArrayIndexKey: the same query may be searched twice.
            <li key={index} className="answer-step-detail">
              {t("search.query", { query })}
            </li>
          ))}
        </ul>
      )}
      <p data-testid="answer-step-found" className="answer-step-detail">
        {row.found === 1
          ? t("steps.search.found.one", { used: row.used })
          : t("steps.search.found", { found: row.found, used: row.used })}
      </p>
      {row.sources.length > 0 && (
        <ul className="answer-step-sources">
          {shown.map((source) => (
            <li key={`${source.marker}`}>
              <SourceRow source={source} />
            </li>
          ))}
          {more > 0 && (
            <li>
              <button
                type="button"
                data-testid="answer-step-sources-more"
                onClick={() => setAll(true)}
                className="answer-step-source answer-step-detail"
              >
                {t("steps.sources.more", { count: more })}
              </button>
            </li>
          )}
        </ul>
      )}
    </Step>
  );
}

/** A source a search led to: its Document and Location, which open beside the Mind as its Citation does. */
function SourceRow({ source }: { source: StepSource }) {
  const t = useT();
  const kind = useAppStore(
    (state) => state.documents.find((document) => document.id === source.documentId)?.kind,
  );
  const location = citationLocation(source);
  const where = location ? formatLocation(location, t) : null;
  return (
    <button
      type="button"
      data-testid="answer-step-source"
      data-document-id={source.documentId}
      title={
        where
          ? t("citation.where", { document: source.documentName, location: where })
          : source.documentName
      }
      onClick={() =>
        openCited(source, {
          check: source.check,
          withQuote: quoteSaysSo(source.check),
          label: where,
        })
      }
      className="answer-step-source"
    >
      <DocumentLineIcon kind={kind ?? "text"} className="answer-step-source-icon" />
      <span className="answer-step-source-name">{source.documentName}</span>
      {where && <span className="answer-step-detail answer-step-source-where">{where}</span>}
    </button>
  );
}

/** A search that failed: one line in plain words, and the one way on. */
function FailedSearch({
  call,
  scope,
  onRegenerate,
}: {
  call: AnswerToolCall;
  scope: string;
  onRegenerate(): void;
}) {
  const t = useT();
  const refused = call.error?.kind === "auth";
  return (
    <ProblemLine
      testId="answer-step-failed"
      title={call.error?.message}
      action={
        refused
          ? {
              label: t("answer.error.checkKey"),
              onClick: () => useAppStore.getState().openSettings("search"),
            }
          : { label: t("answer.error.retry"), onClick: onRegenerate }
      }
      className="answer-step-problem"
    >
      {t(refused ? "steps.failed.search.auth" : "steps.failed.search", { scope })}
    </ProblemLine>
  );
}

/** The Citations checked against their sources: all found, or how many weren't, with the margin's mark. */
function CheckStep({ count, notFound }: { count: number; notFound: number }) {
  const t = useT();
  const allFound = notFound === 0;
  const result = allFound
    ? t(count === 1 ? "steps.check.found" : "steps.check.allFound")
    : t("steps.check.notFound", { count: notFound });
  return (
    <Step
      kind="check"
      icon={
        <CheckMarkIcon
          check={allFound ? "found" : "not-found"}
          data-check={allFound ? "found" : "not-found"}
          className="answer-step-check"
        />
      }
    >
      <p className="answer-step-title">
        {t(count === 1 ? "steps.check.one" : "steps.check", { count })}
        <span aria-hidden="true"> · </span>
        {result}
      </p>
    </Step>
  );
}

/** Whether a Question was asked from a Mind while its meeting was recording. */
function useAskedInMeeting(question: ConversationQuestion | undefined): boolean {
  const meetings = useMeetings((state) => state.meetings);
  const from = question?.askedFrom;
  if (!question || !from) return false;
  const at = Date.parse(question.createdAt);
  return meetings.some(
    (meeting) =>
      meeting.mindId === from.mindId &&
      meeting.startedAt !== null &&
      Date.parse(meeting.startedAt) <= at &&
      (meeting.stoppedAt === null || at <= Date.parse(meeting.stoppedAt)),
  );
}

/** What a Question searched, with `extra` scopes besides (see `stepsScope`). */
function useStepsScopeOf(question: ConversationQuestion | undefined): (extra: number) => string {
  const t = useT();
  const folders = useAppStore((state) => state.folders);
  const groups = useAppStore((state) => state.library?.groups);
  const deletedGroups = useAppStore((state) => state.library?.deletedGroups);
  const tags = useAppStore((state) => state.tags);
  const documents = useAppStore((state) => state.documents);
  const chips = scopeChips(
    { folders, groups, deletedGroups, tags, documents },
    question?.scope ?? NO_SCOPE,
  );
  return (extra) => stepsScope(t, chips, extra);
}
