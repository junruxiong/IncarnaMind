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

/**
 * A Task's row shows in the tray: unfinished, or finished (done or failed)
 * here or opened from the Tasks chip, and not dismissed.
 */
export function inTray(
  task: Task,
  state: Pick<TasksState, "finishedHere" | "dismissed" | "shown">,
): boolean {
  if (task.finishedAt === null) return true;
  const seen = state.finishedHere.has(task.id) || state.shown === task.id;
  return task.status !== "stopped" && seen && !state.dismissed.has(task.id);
}

/** The Tasks whose rows the tray of a Mind shows, oldest first. */
export function trayTasks(state: TasksState, mindId: string): Task[] {
  return Object.values(state.tasks)
    .filter((task) => task.mindId === mindId && inTray(task, state))
    .sort((a, b) => a.startedAt.localeCompare(b.startedAt));
}

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

/** Frames to wait for a Mind just opened to draw the Block a Task works on. */
const BLOCK_FRAMES = 60;

/** Scrolls the Block a Task works on (a comparison table) into the middle, once its Mind draws it. */
function revealBlock(mindId: string, blockId: string, frames = BLOCK_FRAMES): void {
  const block = document.querySelector(
    `article[data-mind-id="${CSS.escape(mindId)}"] [data-comparison-table="${CSS.escape(blockId)}"]`,
  );
  if (block) block.scrollIntoView({ block: "center" });
  else if (frames > 0) requestAnimationFrame(() => revealBlock(mindId, blockId, frames - 1));
}

/** Opens a Task's Mind at its place (its Block, if any), with its row marked in the tray. */
export function showTask(task: Pick<Task, "id" | "mindId" | "blockId">): void {
  useTasks.setState({ shown: task.id });
  if (!task.mindId) return;
  useAppStore.getState().openMind(task.mindId);
  if (task.blockId) revealBlock(task.mindId, task.blockId);
}

core.on("task.changed", (task) => put(task, true));
core.on("task.shown", (shown) =>
  showTask({ id: shown.taskId, mindId: shown.mindId, blockId: shown.blockId }),
);
let listed = false;

/** Once, when the window first shows Tasks: those the core already has. */
export function loadTasks(): void {
  if (listed) return;
  listed = true;
  core.listTasks().then((tasks) => {
    for (const task of tasks) put(task, false);
  }, report);
}
