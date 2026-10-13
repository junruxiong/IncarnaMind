import type { Task } from "../../core/api";

/** Whether a Task waits for the User: an ask, or a pause only they can lift. */
export const needsYou = (task: Task): boolean =>
  task.finishedAt === null &&
  (task.ask !== null ||
    (task.status === "paused" && task.pausedReason !== null && task.pausedReason !== "user"));

/** Whether a Task is doing its steps: running, asking nothing. */
export const isRunning = (task: Task): boolean =>
  task.finishedAt === null && task.status === "running" && task.ask === null;

export interface TaskGroups {
  needsYou: Task[];
  /** Running, and those the User paused, most recent first. */
  running: Task[];
  /** Finished in the last 7 days (the core lists no older ones), most recent first. */
  recent: Task[];
}

const newestFirst = (a: Task, b: Task) =>
  (b.finishedAt ?? b.startedAt).localeCompare(a.finishedAt ?? a.startedAt);

/** The Tasks chip's popover (DESIGN.md, Finding Tasks): Needs you · Running · Recent. */
export function taskGroups(tasks: readonly Task[]): TaskGroups {
  const sorted = [...tasks].sort(newestFirst);
  return {
    needsYou: sorted.filter(needsYou),
    running: sorted.filter((task) => task.finishedAt === null && !needsYou(task)),
    recent: sorted.filter((task) => task.finishedAt !== null),
  };
}

/** What the chip says, or null while no Task runs or needs the User (it isn't shown then). */
export function chipOf(tasks: readonly Task[]): { running: number; needsYou: number } | null {
  const running = tasks.filter(isRunning).length;
  const waiting = tasks.filter(needsYou).length;
  return running + waiting > 0 ? { running, needsYou: waiting } : null;
}
