import { describe, expect, test } from "vitest";
import type { Task } from "../../src/core/api";
import { chipOf, taskGroups } from "../../src/renderer/src/taskGroups";
import { resultLine, stepLine } from "../../src/renderer/src/taskText";
import { translate } from "../../src/shared/i18n";

const task = (id: string, patch: Partial<Task>): Task => ({
  id,
  kind: "comparison",
  mindId: "mind",
  blockId: "table",
  status: "running",
  pausedReason: null,
  resetsAt: null,
  ask: null,
  progress: { done: 2, total: 6 },
  problem: null,
  cost: null,
  startedAt: "2026-10-12T10:00:00.000Z",
  finishedAt: null,
  updatedAt: "2026-10-12T10:05:00.000Z",
  ...patch,
});

const running = task("running", {});
const asking = task("asking", {
  startedAt: "2026-10-12T11:00:00.000Z",
  ask: { kind: "approval", stepKey: "ask:a", number: 1 },
});
const capped = task("capped", { status: "paused", pausedReason: "cost-cap" });
const pausedByUser = task("paused", { status: "paused", pausedReason: "user" });
const done = task("done", {
  status: "done",
  progress: { done: 9, total: 9 },
  finishedAt: "2026-10-12T09:00:00.000Z",
});
const stopped = task("stopped", { status: "stopped", finishedAt: "2026-10-12T12:00:00.000Z" });

describe("the Tasks chip", () => {
  test("shows only while a Task runs or needs the User, counting each", () => {
    expect(chipOf([])).toBeNull();
    expect(chipOf([done, stopped, pausedByUser])).toBeNull();
    expect(chipOf([running])).toEqual({ running: 1, needsYou: 0 });
    expect(chipOf([running, asking, capped])).toEqual({ running: 1, needsYou: 2 });
  });

  test("its popover lists Needs you, then Running (those the User paused too), then Recent, newest first", () => {
    const groups = taskGroups([done, running, stopped, asking, capped, pausedByUser]);
    expect(groups.needsYou.map((each) => each.id)).toEqual(["asking", "capped"]);
    expect(groups.running.map((each) => each.id)).toEqual(["running", "paused"]);
    expect(groups.recent.map((each) => each.id)).toEqual(["stopped", "done"]);
  });

  test("says what each does and how each ended, in English and Chinese", () => {
    const en = (key: Parameters<typeof translate>[1], params?: Parameters<typeof translate>[2]) =>
      translate("en", key, params);
    const zh = (key: Parameters<typeof translate>[1], params?: Parameters<typeof translate>[2]) =>
      translate("zh-CN", key, params);
    expect(stepLine(running, en, "en")).toBe("Filling cell 3 of 6");
    expect(stepLine(asking, en, "en")).toBe("Waiting for your approval");
    expect(stepLine(pausedByUser, en, "en")).toBe("Paused");
    expect(resultLine(done, en)).toBe("9 cells");
    expect(resultLine(stopped, en)).toBe("Stopped");
    expect(stepLine(running, zh, "zh-CN")).toBe("正在填写第 3 个单元格，共 6 个");
    expect(resultLine(done, zh)).toBe("9 个单元格");
    expect(translate("zh-CN", "tasks.chip.running", { count: 1 })).toBe("1 个进行中");
  });
});
