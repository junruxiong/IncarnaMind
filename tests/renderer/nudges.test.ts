import { describe, expect, test } from "vitest";
import { chooseNudge } from "../../src/renderer/src/nudges";

const progress = (patch: object = {}) => ({
  started: true,
  hidden: false,
  citationChecked: false,
  indexed: false,
  askedOwn: true,
  ...patch,
});

const state = (over: { progress?: object; usage?: object; notice?: string } = {}) =>
  ({
    settings: { device: { gettingStarted: progress(over.progress) } },
    privacy: {
      usageData: {
        available: true,
        enabled: false,
        testerBuild: false,
        asked: false,
        localMode: false,
        ...over.usage,
      },
    },
    ownAnswerArrived: true,
    usageNotice: over.notice ?? "auto",
  }) as unknown as Parameters<typeof chooseNudge>[0];

describe("the sidebar's one nudge (#123)", () => {
  test("Get started comes first, then the usage data question once it is out of the way", () => {
    expect(chooseNudge(state())).toBe("get-started");
    expect(chooseNudge(state({ progress: { hidden: true } }))).toBe("usage-data");
    expect(
      chooseNudge(state({ progress: { citationChecked: true, indexed: true, askedOwn: true } })),
    ).toBe("usage-data");
  });

  test("a release asks only after an own Answer, and only until answered", () => {
    const quiet = { hidden: true, askedOwn: false };
    expect(
      chooseNudge({ ...state({ progress: quiet }), ownAnswerArrived: false } as never),
    ).toBeNull();
    expect(chooseNudge(state({ progress: { hidden: true }, usage: { asked: true } }))).toBeNull();
    expect(chooseNudge(state({ progress: { hidden: true }, notice: "answered" }))).toBeNull();
  });

  test("a test build's notice shows from the first run and stays once its switch is used", () => {
    const tester = { testerBuild: true };
    const first = { ...state({ progress: { hidden: true, askedOwn: false }, usage: tester }) };
    expect(chooseNudge({ ...first, ownAnswerArrived: false } as never)).toBe("usage-data");
    const used = state({
      progress: { hidden: true },
      usage: { ...tester, asked: true },
      notice: "kept",
    });
    expect(chooseNudge(used)).toBe("usage-data");
  });

  test("nothing shows without a sender, or in local mode", () => {
    const hidden = { hidden: true };
    expect(chooseNudge(state({ progress: hidden, usage: { available: false } }))).toBeNull();
    expect(chooseNudge(state({ progress: hidden, usage: { localMode: true } }))).toBeNull();
  });
});
