/**
 * Tasks (#126), the window's side: every Task the core lists, as each
 * changes ("task.changed"), the rows the User unfolded or dismissed, and the
 * Task a notification or a click asked to show.
 */
import { create } from "zustand";
import type { Task, TaskAskAnswer } from "../../core/api";
import { core } from "./core";
import { useAppStore } from "./store";

interface TasksState {
  /** By id. */
  tasks: Readonly<Record<string, Task>>;
  /** Tasks that finished while this window was open: their Done row stays until dismissed. */
  finishedHere: ReadonlySet<string>;
  /** Done rows the User dismissed. */
  dismissed: ReadonlySet<string>;
  /** Rows unfolded with ⌄. */
  unfolded: ReadonlySet<string>;
  /** The Task last asked to be shown (a notification's click): its row is marked. */
  shown: string | null;
}

export const useTasks = create<TasksState>()(() => ({
  tasks: {},
  finishedHere: new Set(),
  dismissed: new Set(),
  unfolded: new Set(),
  shown: null,
}));

const report = (error: unknown) => useAppStore.getState().reportError(error);

const toggle = (set: ReadonlySet<string>, id: string, on: boolean) => {
  const next = new Set(set);
  if (on) next.add(id);
  else next.delete(id);
  return next;
};

function put(task: Task, live: boolean): void {
  useTasks.setState((state) => {
    const before = state.tasks[task.id];
    const justFinished = live && task.finishedAt !== null && before?.finishedAt === null;
    return {
      tasks: { ...state.tasks, [task.id]: task },
      ...(justFinished && { finishedHere: toggle(state.finishedHere, task.id, true) }),
    };
  });
}

/** A Task's row shows in the tray: unfinished, or finished here (done or failed) and not dismissed. */
export function inTray(task: Task, state: Pick<TasksState, "finishedHere" | "dismissed">): boolean {
  if (task.finishedAt === null) return true;
  return (
    task.status !== "stopped" && state.finishedHere.has(task.id) && !state.dismissed.has(task.id)
  );
}

/** The Tasks whose rows the tray of a Mind shows, oldest first. */
export function trayTasks(state: TasksState, mindId: string): Task[] {
  return Object.values(state.tasks)
    .filter((task) => task.mindId === mindId && inTray(task, state))
    .sort((a, b) => a.startedAt.localeCompare(b.startedAt));
}

/** Whether a Task waits for the User: an ask, or a pause only they can lift. */
export const needsYou = (task: Task): boolean =>
  task.finishedAt === null &&
  (task.ask !== null ||
    (task.status === "paused" && task.pausedReason !== null && task.pausedReason !== "user"));

/** The unfinished Task filling a comparison table, if any. */
export function tableTask(state: TasksState, tableId: string): Task | null {
  return (
    Object.values(state.tasks).find(
      (task) => task.blockId === tableId && task.finishedAt === null,
    ) ?? null
  );
}

export const pauseTask = (taskId: string) => core.pauseTask(taskId).then(put0, report);
export const resumeTask = (taskId: string) => core.resumeTask(taskId).then(put0, report);
export const stopTask = (taskId: string) => core.stopTask(taskId).then(put0, report);
export const answerTaskAsk = (taskId: string, answer: TaskAskAnswer) =>
  core.answerTaskAsk(taskId, answer).then(put0, report);

function put0(task: Task): void {
  put(task, true);
}

export const dismissTask = (taskId: string) =>
  useTasks.setState((state) => ({ dismissed: toggle(state.dismissed, taskId, true) }));

export const unfoldTask = (taskId: string, unfolded: boolean) =>
  useTasks.setState((state) => ({ unfolded: toggle(state.unfolded, taskId, unfolded) }));

/** Opens a Task's Mind, with its row marked in the tray. */
export function showTask(task: Pick<Task, "id" | "mindId">): void {
  useTasks.setState({ shown: task.id });
  if (task.mindId) useAppStore.getState().openMind(task.mindId);
}

core.on("task.changed", (task) => put(task, true));
core.on("task.shown", (shown) => showTask({ id: shown.taskId, mindId: shown.mindId }));
core.listTasks().then((tasks) => {
  for (const task of tasks) put(task, false);
}, report);
