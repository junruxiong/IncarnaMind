import { isMacOS } from "@tiptap/core";
import {
  type KeyboardEvent,
  type ReactNode,
  useEffect,
  useId,
  useLayoutEffect,
  useRef,
  useState,
} from "react";
import { useShallow } from "zustand/react/shallow";
import type { ApprovalRequest, Task, TaskEffect } from "../../../core/api";
import { useApprovals } from "../approvals";
import { keysLabel } from "../commands/keys";
import { formatMoney } from "../cost";
import { ApprovalCard } from "../editor/AnswerView";
import { useLanguage, useT } from "../i18n";
import { useAppStore } from "../store";
import { needsYou } from "../taskGroups";
import {
  answerTaskAsk,
  dismissTask,
  loadTasks,
  pauseTask,
  resumeTask,
  stopTask,
  trayTasks,
  unfoldTask,
  useTasks,
} from "../tasks";
import { stepLine, taskName } from "../taskText";
import {
  CheckLineIcon,
  ChevronDownLineIcon,
  CloseLineIcon,
  PauseLineIcon,
  ResumeLineIcon,
  StopLineIcon,
} from "./lineIcons";

type Translate = ReturnType<typeof useT>;

/** The share of the assistant's column the tray may take (DESIGN.md, Task tray). */
const TRAY_SHARE = 0.4;
/** The least the tray takes: one row; with an ask, its row and the ask's buttons. */
const MIN_TRAY = 64;
const MIN_TRAY_WITH_ASK = 220;
/** What the conversation keeps above the tray and the composer, at the least. */
const MIN_THREAD = 40;

const ALLOW_KEYS = "CmdOrCtrl+Enter";

const isCommand = (event: { metaKey: boolean; ctrlKey: boolean }) =>
  isMacOS() ? event.metaKey && !event.ctrlKey : event.ctrlKey && !event.metaKey;

/** The minutes since a Task started, rounded, at least 1, as of the minute shown. */
function useMinutes(since: string): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const timer = setInterval(() => setNow(Date.now()), 30_000);
    return () => clearInterval(timer);
  }, []);
  return Math.max(1, Math.round((now - new Date(since).getTime()) / 60_000));
}

/** A Task's row in the tray: its dot, name, step and time; then Pause or Resume, Stop and ⌄. */
function TaskRow({ task }: { task: Task }) {
  const t = useT();
  const language = useLanguage();
  const minutes = useMinutes(task.startedAt);
  const unfolded = useTasks((state) => state.unfolded.has(task.id));
  const shown = useTasks((state) => state.shown === task.id);
  const openSettings = useAppStore((state) => state.openSettings);
  const waiting = needsYou(task);
  const step = stepLine(task, t, language);
  const detailsId = useId();
  const running = task.status === "running" && !task.ask;
  return (
    <div
      data-testid="task-row"
      data-task={task.id}
      data-status={task.status}
      data-needs-you={waiting || undefined}
      data-shown={shown || undefined}
      className="task-row"
    >
      <div className="task-line">
        <span
          aria-hidden="true"
          className={
            waiting
              ? "task-dot task-dot--attention"
              : running
                ? "task-dot"
                : "task-dot task-dot--idle"
          }
        />
        <span className="task-name">{taskName(task, t)}</span>
        <span data-testid="task-step" className="task-step" title={step}>
          {step}
        </span>
        <span className="task-time">{t("task.minutes", { count: minutes })}</span>
      </div>
      <div className="task-line task-controls">
        {task.status === "paused" && task.pausedReason === "cost-cap" && (
          <button
            type="button"
            data-testid="task-change-cap"
            className="task-link"
            onClick={() => openSettings("general")}
          >
            {t("answer.error.changeCap")}
          </button>
        )}
        <span className="flex-1" />
        {running && (
          <TrayButton
            testId="task-pause"
            label={t("task.pause")}
            onClick={() => pauseTask(task.id)}
          >
            <PauseLineIcon className="size-3.5" />
          </TrayButton>
        )}
        {task.status === "paused" && !task.ask && (
          <TrayButton
            testId="task-resume"
            label={t("task.resume")}
            onClick={() => resumeTask(task.id)}
          >
            <ResumeLineIcon className="size-3.5" />
          </TrayButton>
        )}
        <TrayButton testId="task-stop" label={t("task.stop")} onClick={() => stopTask(task.id)}>
          <StopLineIcon className="size-3.5" />
        </TrayButton>
        <TrayButton
          testId="task-unfold"
          label={t(unfolded ? "task.fold" : "task.unfold")}
          expanded={unfolded}
          controls={detailsId}
          onClick={() => unfoldTask(task.id, !unfolded)}
        >
          <ChevronDownLineIcon className={unfolded ? "size-3.5 rotate-180" : "size-3.5"} />
        </TrayButton>
      </div>
      {unfolded && (
        <p id={detailsId} data-testid="task-details" className="task-details">
          {details(task, t, language).join(" · ")}
        </p>
      )}
    </div>
  );
}

/** What ⌄ shows: how far it is, when it started, and what it has cost. */
function details(task: Task, t: Translate, language: ReturnType<typeof useLanguage>): string[] {
  const parts: string[] = [];
  if (task.kind === "comparison" && task.progress) {
    parts.push(t("task.details.cells", { done: task.progress.done, total: task.progress.total }));
  }
  const time = new Intl.DateTimeFormat(language, { hour: "numeric", minute: "2-digit" }).format(
    new Date(task.startedAt),
  );
  parts.push(t("task.details.started", { time }));
  if (task.cost) parts.push(t("task.details.cost", { cost: formatMoney(task.cost, language) }));
  return parts;
}

/** A finished Task's row: done, or why it couldn't finish, until dismissed. */
function DoneRow({ task }: { task: Task }) {
  const t = useT();
  const total = task.progress?.total ?? 0;
  const text =
    task.status === "done"
      ? task.kind === "comparison"
        ? t(total === 1 ? "task.comparison.done.one" : "task.comparison.done.other", {
            count: total,
          })
        : `${taskName(task, t)} · ${t("task.status.stopped")}`
      : `${taskName(task, t)} · ${t("task.status.failed")}`;
  return (
    <div data-testid="task-row" data-task={task.id} data-status={task.status} className="task-row">
      <div className="task-line">
        {task.status === "done" ? (
          <CheckLineIcon aria-hidden="true" className="task-done-icon" />
        ) : (
          <span aria-hidden="true" className="task-dot task-dot--idle" />
        )}
        <span data-testid="task-step" className="task-step task-step--done" title={text}>
          {text}
        </span>
        <TrayButton
          testId="task-dismiss"
          label={t("task.dismiss")}
          onClick={() => dismissTask(task.id)}
        >
          <CloseLineIcon className="size-3.5" />
        </TrayButton>
      </div>
    </div>
  );
}

function TrayButton({
  testId,
  label,
  expanded,
  controls,
  onClick,
  children,
}: {
  testId: string;
  label: string;
  expanded?: boolean;
  controls?: string;
  onClick(): void;
  children: ReactNode;
}) {
  return (
    <button
      type="button"
      data-testid={testId}
      aria-label={label}
      title={label}
      aria-expanded={expanded}
      aria-controls={expanded ? controls : undefined}
      onClick={onClick}
      className="task-button"
    >
      {children}
    </button>
  );
}

/** What an Effect was, in the ask: the file's name, or the Connector and its Tool. */
function effectText(effect: TaskEffect, t: Translate): string {
  if (effect.kind === "file-write") {
    const name = effect.path.split(/[\\/]/).pop() ?? effect.path;
    return t("task.unknown.file", { name });
  }
  return t("task.unknown.connector", { connector: effect.connectorName, tool: effect.tool });
}

/** IncarnaMind can't tell whether an Effect happened: do it again, or go on without it. */
function UnknownOutcomeCard({ task, effect }: { task: Task; effect: TaskEffect }) {
  const t = useT();
  const titleId = useId();
  const mac = isMacOS();
  return (
    <div
      role="alertdialog"
      aria-labelledby={titleId}
      data-testid="task-unknown-card"
      className="approval-card"
    >
      <div className="approval-head">
        <p id={titleId} className="approval-title">
          <span aria-hidden="true" className="approval-dot" />
          <span className="approval-title-text">{t("task.unknown.title")}</span>
        </p>
        <p className="approval-explanation">{effectText(effect, t)}</p>
      </div>
      <div className="approval-actions task-ask-actions">
        <button
          type="button"
          data-testid="task-redo"
          data-allow
          onClick={() => answerTaskAsk(task.id, "redo")}
          className="approval-button approval-button--primary"
        >
          {t("task.unknown.redo")}
          <span className="approval-keys">{keysLabel(ALLOW_KEYS, mac)}</span>
        </button>
        <button
          type="button"
          data-testid="task-skip"
          data-deny
          onClick={() => answerTaskAsk(task.id, "skip")}
          className="approval-button"
        >
          {t("task.unknown.skip")}
        </button>
      </div>
    </div>
  );
}

/** The one ask waiting, nearest the keyboard: an approval or an Effect it can't check. */
function TaskAskView({ task }: { task: Task }) {
  const t = useT();
  const request = useApprovals((state) =>
    Object.values(state.waiting).find((each: ApprovalRequest) => each.taskId === task.id),
  );
  const ask = task.ask;
  if (!ask) return null;
  if (ask.kind === "approval" && !request) return null;
  // ⌘↵ takes the first choice, ⌘⌫ the last.
  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (!isCommand(event) || event.altKey || event.shiftKey) return;
    const card = event.currentTarget;
    const pick =
      event.key === "Enter"
        ? card.querySelector<HTMLButtonElement>("[data-allow], [data-testid='approval-allow-once']")
        : event.key === "Backspace"
          ? card.querySelector<HTMLButtonElement>("[data-deny], [data-testid='approval-deny']")
          : null;
    if (!pick) return;
    event.preventDefault();
    pick.click();
  };
  return (
    // biome-ignore lint/a11y/noStaticElementInteractions: the ask's keys, heard while the focus is on its buttons.
    <div data-testid="task-ask" data-task={task.id} className="task-ask" onKeyDown={onKeyDown}>
      {ask.number > 1 && (
        <p data-testid="task-ask-count" className="task-ask-count">
          {t("task.ask.count", { number: ask.number, total: ask.number })}
        </p>
      )}
      {ask.kind === "approval" && request ? (
        <ApprovalCard request={request} keys={keysLabel(ALLOW_KEYS, isMacOS())} />
      ) : (
        ask.kind === "unknown-outcome" && <UnknownOutcomeCard task={task} effect={ask.effect} />
      )}
    </div>
  );
}

/**
 * The Task tray (#126; DESIGN.md, Task tray; the boards Tasks-Compare and
 * Tasks-Paused-Needs-You): joined to the top of the assistant's composer,
 * the live state of the Tasks in the Mind it reads, one row each, then the
 * one waiting ask, its buttons pinned at the bottom while the rest of it
 * scrolls. It takes at most 40% of the column; with an ask, at least room
 * for its buttons. Nothing in it ever goes into the Mind.
 */
export function TaskTray({ mindId }: { mindId: string }) {
  const t = useT();
  useEffect(loadTasks, []);
  const tasks = useTasks(useShallow((state) => trayTasks(state, mindId)));
  const tray = useRef<HTMLElement>(null);
  const [maxHeight, setMaxHeight] = useState<number | undefined>(undefined);
  const asking = tasks.find((task) => task.ask !== null && task.finishedAt === null);
  const present = tasks.length > 0;
  const withAsk = asking !== undefined;

  useLayoutEffect(() => {
    const element = tray.current;
    const column = element?.closest<HTMLElement>(".assistant-main");
    if (!present || !element || !column) return;
    const fitTray = () => {
      const composer = element.nextElementSibling?.getBoundingClientRect().height ?? 0;
      const room = Math.max(MIN_TRAY, column.clientHeight - composer - MIN_THREAD);
      const share = Math.floor(column.clientHeight * TRAY_SHARE);
      setMaxHeight(Math.min(room, Math.max(share, withAsk ? MIN_TRAY_WITH_ASK : MIN_TRAY)));
    };
    fitTray();
    const observer = new ResizeObserver(fitTray);
    observer.observe(column);
    return () => observer.disconnect();
  }, [present, withAsk]);

  if (!present) return null;
  return (
    <section
      ref={tray}
      data-testid="task-tray"
      aria-label={t("task.tray.label")}
      className="task-tray"
      style={{ maxHeight }}
    >
      <div className="task-rows">
        {tasks.map((task) =>
          task.finishedAt === null ? (
            <TaskRow key={task.id} task={task} />
          ) : (
            <DoneRow key={task.id} task={task} />
          ),
        )}
      </div>
      {asking && <TaskAskView task={asking} />}
    </section>
  );
}
