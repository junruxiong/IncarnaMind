/**
 * Tasks (#126), the window's side: every Task the core lists, as each
 * changes ("task.changed"), and the Task a notification or a click asked to
 * show.
 */
import { create } from "zustand";
import type { Task, TaskAskAnswer } from "../../core/api";
import { core } from "./core";
import { useAppStore } from "./store";

interface TasksState {
  /** By id. */
  tasks: Readonly<Record<string, Task>>;
  /** The Task last asked to be shown (a notification's click, the chip): its block is marked. */
  shown: string | null;
}

export const useTasks = create<TasksState>()(() => ({
  tasks: {},
  shown: null,
}));

const report = (error: unknown) => useAppStore.getState().reportError(error);

function put(task: Task): void {
  useTasks.setState((state) => ({ tasks: { ...state.tasks, [task.id]: task } }));
}

/** The unfinished Task filling a comparison table, if any. */
export function tableTask(state: TasksState, tableId: string): Task | null {
  return (
    Object.values(state.tasks).find(
      (task) => task.blockId === tableId && task.finishedAt === null,
    ) ?? null
  );
}

export const pauseTask = (taskId: string) => core.pauseTask(taskId).then(put, report);
export const resumeTask = (taskId: string) => core.resumeTask(taskId).then(put, report);
export const stopTask = (taskId: string) => core.stopTask(taskId).then(put, report);
export const answerTaskAsk = (taskId: string, answer: TaskAskAnswer) =>
  core.answerTaskAsk(taskId, answer).then(put, report);

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

core.on("task.changed", put);
core.on("task.shown", (shown) =>
  showTask({ id: shown.taskId, mindId: shown.mindId, blockId: shown.blockId }),
);
let listed = false;

/** Once, when the window first shows Tasks: those the core already has. */
export function loadTasks(): void {
  if (listed) return;
  listed = true;
  core.listTasks().then((tasks) => {
    for (const task of tasks) put(task);
  }, report);
}
