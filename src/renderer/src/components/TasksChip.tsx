import { useEffect, useLayoutEffect } from "react";
import { useShallow } from "zustand/react/shallow";
import type { Task } from "../../../core/api";
import { dayOf } from "../dates";
import { useLanguage, useT } from "../i18n";
import type { Translate } from "../objects";
import { useAppStore } from "../store";
import { chipOf, needsYou, taskGroups } from "../taskGroups";
import { loadTasks, showTask, useTasks } from "../tasks";
import { resultLine, stepLine, taskName } from "../taskText";
import { CheckLineIcon } from "./lineIcons";
import { usePopoverMenu } from "./usePopoverMenu";

/** Room kept between the popover and the assistant below it. */
const ASSISTANT_GAP_PX = 8;

const allTasks = (state: { tasks: Readonly<Record<string, Task>> }) => Object.values(state.tasks);

/**
 * Ends the popover above the assistant when the assistant sits below it (docked at the
 * bottom, or floating), so it never covers the composer or the Question being asked.
 */
function keepAboveAssistant(menu: HTMLElement | null): void {
  const assistant = document.querySelector('[data-testid="assistant"]');
  if (!menu || !assistant) return;
  menu.style.removeProperty("max-height");
  const own = menu.getBoundingClientRect();
  const other = assistant.getBoundingClientRect();
  const across = own.left < other.right && other.left < own.right;
  if (across && other.top > own.top && own.bottom > other.top - ASSISTANT_GAP_PX) {
    menu.style.maxHeight = `${Math.max(0, other.top - own.top - ASSISTANT_GAP_PX)}px`;
  }
}

/**
 * The Tasks chip (#126; DESIGN.md, Finding Tasks; the board Tasks-Chip-Popover):
 * in the band, only while a Task runs or needs the User. Its popover lists
 * Needs you · Running · Recent (7 days); a row opens its Mind at the Task's
 * place, with its row in the tray. Esc or a click outside closes it.
 */
export function TasksChip() {
  const t = useT();
  const tasks = useTasks(useShallow(allTasks));
  const chip = chipOf(tasks);
  // Under the band, its right edge by the chip's: over the card, never the assistant beside it.
  const menu = usePopoverMenu({
    focus: "first",
    around: (button) => button.closest<HTMLElement>('[data-testid="tab-strip"]'),
  });
  const { open } = menu;
  const menuRef = menu.menuProps.ref;

  useEffect(loadTasks, []);
  useLayoutEffect(() => {
    if (open) keepAboveAssistant(menuRef.current);
  }, [open, menuRef]);

  if (!chip) return null;
  const summary =
    chip.running > 0
      ? t("tasks.chip.running", { count: chip.running })
      : t("tasks.chip.needsYou", { count: chip.needsYou });
  return (
    <>
      <button
        type="button"
        {...menu.buttonProps}
        data-testid="tasks-chip"
        data-needs-you={chip.needsYou > 0 || undefined}
        aria-label={t("tasks.chip.label", { summary })}
        className="tasks-chip"
      >
        <span
          aria-hidden="true"
          className={chip.needsYou > 0 ? "task-dot task-dot--attention" : "task-dot"}
        />
        {summary}
      </button>
      <div
        {...menu.menuProps}
        role="menu"
        aria-label={t("tasks.popover.label")}
        data-testid="tasks-popover"
        className="model-menu tasks-popover"
      >
        {open && <TaskLists tasks={tasks} close={menu.close} />}
      </div>
    </>
  );
}

function TaskLists({ tasks, close }: { tasks: readonly Task[]; close(): void }) {
  const t = useT();
  const groups = taskGroups(tasks);
  const open = (task: Task) => {
    close();
    showTask(task);
  };
  const section = (
    heading: "tasks.group.needsYou" | "tasks.group.running" | "tasks.group.recent",
    list: readonly Task[],
  ) =>
    list.length > 0 && (
      <section className="model-menu-list" aria-label={t(heading)}>
        <p className="model-menu-heading">{t(heading)}</p>
        {list.map((task) => (
          <TaskItem key={task.id} task={task} onOpen={() => open(task)} />
        ))}
      </section>
    );
  return (
    <>
      {section("tasks.group.needsYou", groups.needsYou)}
      {section("tasks.group.running", groups.running)}
      {section("tasks.group.recent", groups.recent)}
    </>
  );
}

function TaskItem({ task, onOpen }: { task: Task; onOpen(): void }) {
  const t = useT();
  const language = useLanguage();
  const mind = useAppStore(
    (state) => state.minds.find((each) => each.id === task.mindId)?.title ?? "",
  );
  const finished = task.finishedAt !== null;
  const { title, line } = finished
    ? recentText(task, mind, t, language)
    : { title: taskName(task, t), line: [mind, stepLine(task, t, language)] };
  return (
    <button
      type="button"
      role="menuitem"
      data-testid="tasks-popover-row"
      data-task={task.id}
      onClick={onOpen}
      className="model-menu-item tasks-popover-row"
    >
      {finished ? (
        task.status === "done" ? (
          <CheckLineIcon aria-hidden="true" className="tasks-popover-done" />
        ) : (
          <span aria-hidden="true" className="task-dot task-dot--idle" />
        )
      ) : (
        <span
          aria-hidden="true"
          className={
            needsYou(task)
              ? "task-dot task-dot--attention"
              : task.status === "running"
                ? "task-dot"
                : "task-dot task-dot--idle"
          }
        />
      )}
      <span className="model-menu-name">
        <span className="tasks-popover-title">{title}</span>
        <span className="model-menu-hint tasks-popover-line">
          {line.filter(Boolean).join(" · ")}
        </span>
      </span>
    </button>
  );
}

function recentText(
  task: Task,
  mind: string,
  t: Translate,
  language: ReturnType<typeof useLanguage>,
): { title: string; line: string[] } {
  const name = taskName(task, t);
  return {
    title: mind ? t("tasks.row.recentTitle", { task: name, mind }) : name,
    line: [
      dayOf(new Date(task.finishedAt ?? task.updatedAt), new Date(), language, t),
      resultLine(task, t),
    ],
  };
}
