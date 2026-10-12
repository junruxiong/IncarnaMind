import { afterEach, beforeAll, describe, expect, test, vi } from "vitest";
import {
  DEFAULT_ASSISTANT,
  type Meeting,
  type Settings,
  type WindowLayout,
} from "../../src/core/api";
import { keysLabel, keysOn } from "../../src/renderer/src/commands/keys";
import {
  type CommandContext,
  commands,
  isEnabled,
  labelOf,
  offered,
} from "../../src/renderer/src/commands/registry";
import type { CommandId } from "../../src/shared/commands";
import { type MessageKey, type MessageParams, translate } from "../../src/shared/i18n";

/*
 * Recording a Meeting from the command registry (#135, #222): its commands'
 * labels in both languages, ⌘⇧P, and when each can run.
 */

const IN_MIND: CommandContext = { selection: [], focus: null, surface: "mind" };
const en = (key: MessageKey, params?: MessageParams) => translate("en", key, params);
const zh = (key: MessageKey, params?: MessageParams) => translate("zh-CN", key, params);

type Stores = typeof import("../../src/renderer/src/store") &
  typeof import("../../src/renderer/src/meetings");
let stores: Stores;

beforeAll(async () => {
  // The commands reach the stores, which read the bridge the preload exposes.
  const bridge = new Proxy({}, { get: () => () => () => undefined });
  (globalThis as unknown as { window: unknown }).window = {
    incarnamind: bridge,
    incarnamindFiles: bridge,
  };
  await import("../../src/renderer/src/commands/app");
  stores = {
    ...(await import("../../src/renderer/src/store")),
    ...(await import("../../src/renderer/src/meetings")),
  };
});

afterEach(() => {
  stores.useMeetings.setState({ meetings: [] });
  stores.useAppStore.setState({ settings: null, layout: null });
});

const meetingsOn = (on = true) =>
  stores.useAppStore.setState({
    settings: {
      device: { meetingsEnabled: on },
      meetingAudio: "both",
    } as unknown as Settings,
  });

const meeting = (id: string, mindId: string, status: Meeting["status"]): Meeting =>
  ({ id, mindId, status }) as Meeting;

/** A window showing this Mind's tab. */
const showing = (mindId: string) =>
  stores.useAppStore.setState({
    layout: {
      groups: [
        {
          id: "group",
          tabs: [
            {
              id: "tab",
              item: { kind: "mind", mindId },
              preview: false,
              pinned: false,
              place: null,
              opened: 0,
            },
          ],
          activeTabId: "tab",
          width: 1,
        },
      ],
      focusedGroupId: "group",
      reopenable: 0,
      assistant: DEFAULT_ASSISTANT,
    } satisfies WindowLayout,
  });

const command = (id: CommandId) => {
  const found = commands.get(id);
  if (!found) throw new Error(`No command ${id}.`);
  return found;
};

describe("the meeting commands", () => {
  test("are named in English and Chinese, and Pause/Resume says which it will do", () => {
    meetingsOn();
    expect(labelOf(command("meeting.record"), IN_MIND, en)).toBe("Record a meeting");
    expect(labelOf(command("meeting.record"), IN_MIND, zh)).toBe("录制会议");
    expect(labelOf(command("meeting.stop"), IN_MIND, en)).toBe("Stop recording");
    expect(labelOf(command("meeting.stop"), IN_MIND, zh)).toBe("停止录制");

    stores.useMeetings.setState({ meetings: [meeting("m1", "mind-1", "recording")] });
    expect(labelOf(command("meeting.pauseResume"), IN_MIND, en)).toBe("Pause recording");
    expect(labelOf(command("meeting.pauseResume"), IN_MIND, zh)).toBe("暂停录制");
    stores.useMeetings.setState({ meetings: [meeting("m1", "mind-1", "paused")] });
    showing("mind-1");
    expect(labelOf(command("meeting.pauseResume"), IN_MIND, en)).toBe("Resume recording");
    expect(labelOf(command("meeting.pauseResume"), IN_MIND, zh)).toBe("继续录制");
  });

  test("Pause/Resume is ⌘⇧P on macOS and Ctrl+Shift+P elsewhere, a window-wide key", () => {
    const shortcut = command("meeting.pauseResume").shortcut;
    expect(shortcut?.list).toBeUndefined();
    expect(keysLabel(keysOn(shortcut?.keys ?? "", true), true)).toBe("⇧⌘P");
    expect(keysLabel(keysOn(shortcut?.keys ?? "", false), false)).toBe("Ctrl+Shift+P");
  });

  test("Stop can't be taken back, so it says so; the others can", () => {
    expect(command("meeting.record").reversible).toBe(true);
    expect(command("meeting.pauseResume").reversible).toBe(true);
    expect(command("meeting.stop").reversible).toEqual({
      confirm: "meetings.command.stopConfirm",
    });
  });

  test("are offered only while Meetings are on, and Pause/Resume and Stop only while a meeting is live", () => {
    meetingsOn(false);
    for (const id of ["meeting.record", "meeting.pauseResume", "meeting.stop"] as const) {
      expect(offered(command(id), IN_MIND), id).toBe(false);
    }

    meetingsOn();
    expect(isEnabled(command("meeting.record"), IN_MIND)).toBe(true);
    for (const id of ["meeting.pauseResume", "meeting.stop"] as const) {
      expect(offered(command(id), IN_MIND), id).toBe(true);
      expect(command(id).enabled?.(IN_MIND), id).toEqual({
        reason: "meetings.command.noneLive",
      });
    }
    // A meeting that hasn't started, or has stopped, isn't live.
    stores.useMeetings.setState({
      meetings: [meeting("m1", "mind-1", "ready"), meeting("m2", "mind-2", "stopped")],
    });
    showing("mind-1");
    expect(isEnabled(command("meeting.pauseResume"), IN_MIND)).toBe(false);

    stores.useMeetings.setState({ meetings: [meeting("m1", "mind-1", "recording")] });
    expect(isEnabled(command("meeting.pauseResume"), IN_MIND)).toBe(true);
    expect(isEnabled(command("meeting.stop"), IN_MIND)).toBe(true);
  });

  test("act on the meeting in view, else on the one recording, from any Mind", async () => {
    meetingsOn();
    const pause = vi.fn(async () => {});
    const resume = vi.fn(async () => {});
    const stop = vi.fn(async () => {});
    stores.useMeetings.setState({
      pause,
      resume,
      stop,
      meetings: [meeting("m1", "mind-1", "recording"), meeting("m2", "mind-2", "paused")],
    });

    // In another Mind, ⌘⇧P pauses the meeting that records.
    showing("mind-3");
    await commands.run("meeting.pauseResume", IN_MIND);
    expect(pause).toHaveBeenCalledWith("m1");

    // In a paused meeting's Mind, it resumes that one.
    showing("mind-2");
    await commands.run("meeting.pauseResume", IN_MIND);
    expect(resume).toHaveBeenCalledWith("m2");
    await commands.run("meeting.stop", IN_MIND);
    expect(stop).toHaveBeenCalledWith("m2");
  });
});

describe("Mark (#139)", () => {
  test("is ⌘D on macOS and Ctrl+D elsewhere, a window-wide key, named in both languages", () => {
    const mark = command("meeting.mark");
    expect(mark.shortcut?.list).toBeUndefined();
    expect(keysLabel(keysOn(mark.shortcut?.keys ?? "", true), true)).toBe("⌘D");
    expect(keysLabel(keysOn(mark.shortcut?.keys ?? "", false), false)).toBe("Ctrl+D");
    expect(labelOf(mark, IN_MIND, en)).toBe("Mark this moment");
    expect(labelOf(mark, IN_MIND, zh)).toBe("标记此刻");
    expect(mark.reversible).toBe(true);
  });

  test("is offered while Meetings are on, and works only while audio is recorded", () => {
    const mark = command("meeting.mark");
    meetingsOn(false);
    expect(offered(mark, IN_MIND)).toBe(false);

    meetingsOn();
    expect(offered(mark, IN_MIND)).toBe(true);
    expect(mark.enabled?.(IN_MIND)).toEqual({ reason: "meetings.command.noneLive" });

    stores.useMeetings.setState({ meetings: [meeting("m1", "mind-1", "paused")] });
    showing("mind-1");
    expect(mark.enabled?.(IN_MIND)).toEqual({ reason: "meetings.command.markPaused" });

    stores.useMeetings.setState({ meetings: [meeting("m1", "mind-1", "recording")] });
    expect(isEnabled(mark, IN_MIND)).toBe(true);
  });
});
