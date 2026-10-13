import type { Task } from "../../core/api";
import type { Language } from "../../core/language";
import { runStopText } from "./cost";
import type { Translate } from "./objects";

/** A Task's name, from its kind. */
export const taskName = (task: Task, t: Translate) =>
  t(task.kind === "comparison" ? "task.kind.comparison" : "task.kind.other");

/** What a Task is doing, in one line: its step, or what it waits for. */
export function stepLine(task: Task, t: Translate, language: Language): string {
  if (task.ask) {
    if (task.pausedReason === "closed") return t("task.status.pausedClosed");
    return task.ask.kind === "approval"
      ? t("task.status.waitingApproval")
      : t("task.status.needsYou");
  }
  if (task.status === "paused") {
    switch (task.pausedReason) {
      case "cost-cap":
        return runStopText({ reason: "cost-cap" }, t, language);
      case "plan-limit":
        return runStopText({ reason: "plan-limit", resetsAt: task.resetsAt }, t, language);
      case "not-ready":
        return t("task.status.notReady");
      default:
        return t("task.status.paused");
    }
  }
  if (task.kind === "comparison" && task.progress && task.progress.total > 0) {
    return t("task.comparison.filling", {
      current: Math.min(task.progress.done + 1, task.progress.total),
      total: task.progress.total,
    });
  }
  return t("task.comparison.starting");
}

/** How a finished Task ended, in a few words: "9 cells", "Stopped", "Couldn't finish". */
export function resultLine(task: Task, t: Translate): string {
  if (task.status === "failed") return t("task.status.failed");
  if (task.status === "stopped") return t("task.status.stopped");
  const total = task.progress?.total ?? 0;
  if (task.kind !== "comparison") return t("task.status.done");
  return t(total === 1 ? "task.comparison.cells.one" : "task.comparison.cells.other", {
    count: total,
  });
}
