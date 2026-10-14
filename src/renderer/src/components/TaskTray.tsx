import { isMacOS } from "@tiptap/core";
import { type KeyboardEvent, useEffect, useId, useLayoutEffect, useRef, useState } from "react";
import type { ApprovalRequest, Task, TaskEffect } from "../../../core/api";
import { useApprovals } from "../approvals";
import { keysLabel } from "../commands/keys";
import { ApprovalCard, withShortcut } from "../editor/AnswerView";
import { useT } from "../i18n";
import type { Translate } from "../objects";
import { answerTaskAsk, loadTasks, useTasks } from "../tasks";

/** The share of the assistant's column the tray may take (DESIGN.md, Task tray). */
const TRAY_SHARE = 0.4;
/** The least the tray takes with an ask, room allowing: its count, its title and its buttons. */
const MIN_TRAY = 220;
/** The least it takes in a short column: its buttons stay in reach, the ask scrolling above them. */
const MIN_ROOM = 56;
/** What the conversation keeps above the tray and the composer, at the least: a few lines (#678). */
const MIN_THREAD = 120;

const ALLOW_KEYS = "CmdOrCtrl+Enter";

const isCommand = (event: { metaKey: boolean; ctrlKey: boolean }) =>
  isMacOS() ? event.metaKey && !event.ctrlKey : event.ctrlKey && !event.metaKey;

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
          {...withShortcut(t("task.unknown.redo"), keysLabel(ALLOW_KEYS, mac))}
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
 * The Task tray (#126, narrowed by D189; DESIGN.md, Task tray): joined to the
 * top of the assistant's composer, only what waits for the User, nearest the
 * keyboard. Top to bottom: the plan card before a Task starts, the one
 * waiting ask, then queued messages, which wrap. Progress is in the thread
 * (`TaskProgress`); the plan card and the queue come with the first kind that
 * has them (#130), in those places around the ask. It takes at most 40% of
 * the column, and at least room for the ask's buttons; the ask scrolls inside
 * with its buttons pinned. Nothing in it goes into the Mind.
 */
export function TaskTray({ mindId }: { mindId: string }) {
  const t = useT();
  useEffect(loadTasks, []);
  const asking = useTasks((state) =>
    Object.values(state.tasks).find(
      (task) => task.mindId === mindId && task.finishedAt === null && task.ask !== null,
    ),
  );
  const tray = useRef<HTMLElement>(null);
  const [maxHeight, setMaxHeight] = useState<number | undefined>(undefined);
  const present = asking !== undefined;

  useLayoutEffect(() => {
    const element = tray.current;
    const column = element?.closest<HTMLElement>(".assistant-main");
    if (!present || !element || !column) return;
    const fitTray = () => {
      // The composer and the room under it, which the slot's padding keeps.
      const slot = element.parentElement?.getBoundingClientRect().height ?? 0;
      const composer = slot - element.getBoundingClientRect().height;
      const room = Math.max(MIN_ROOM, column.clientHeight - composer - MIN_THREAD);
      const share = Math.floor(column.clientHeight * TRAY_SHARE);
      setMaxHeight(Math.min(room, Math.max(share, MIN_TRAY)));
    };
    fitTray();
    const observer = new ResizeObserver(fitTray);
    observer.observe(column);
    return () => observer.disconnect();
  }, [present]);

  if (!asking) return null;
  return (
    <section
      ref={tray}
      data-testid="task-tray"
      aria-label={t("task.tray.label")}
      className="task-tray"
      style={{ maxHeight }}
    >
      <TaskAskView task={asking} />
    </section>
  );
}
