import { describe, expect, test } from "vitest";
import { chooseNudge } from "../../src/renderer/src/nudges";

const state = (
  over: { usage?: object; notice?: string; ownAnswerArrived?: boolean; askedOwn?: boolean } = {},
) =>
  ({
    settings: { device: { gettingStarted: { askedOwn: over.askedOwn ?? false } } },
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
    ownAnswerArrived: over.ownAnswerArrived ?? false,
    usageNotice: over.notice ?? "auto",
  }) as unknown as Parameters<typeof chooseNudge>[0];

describe("the sidebar's one nudge (#123)", () => {
  test("a release asks only after an own Answer arrives, and only until answered", () => {
    expect(chooseNudge(state())).toBeNull();
    expect(chooseNudge(state({ ownAnswerArrived: true }))).toBe("usage-data");
    expect(chooseNudge(state({ askedOwn: true }))).toBe("usage-data");
    expect(chooseNudge(state({ ownAnswerArrived: true, usage: { asked: true } }))).toBeNull();
    expect(chooseNudge(state({ ownAnswerArrived: true, notice: "answered" }))).toBeNull();
  });

  test("a test build says so from the first run, and keeps it once the switch is used", () => {
    const tester = { testerBuild: true };
    expect(chooseNudge(state({ usage: tester }))).toBe("usage-data");
    expect(chooseNudge(state({ usage: { ...tester, asked: true }, notice: "kept" }))).toBe(
      "usage-data",
    );
    expect(
      chooseNudge(state({ usage: { ...tester, asked: true }, notice: "answered" })),
    ).toBeNull();
  });

  test("nothing shows without a sender, or in local mode", () => {
    expect(chooseNudge(state({ ownAnswerArrived: true, usage: { available: false } }))).toBeNull();
    expect(chooseNudge(state({ ownAnswerArrived: true, usage: { localMode: true } }))).toBeNull();
  });
});
