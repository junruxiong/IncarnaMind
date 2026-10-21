import { beforeEach, describe, expect, test, vi } from "vitest";
import type { CoreBridge, MindCheckPlan, RunEstimate } from "../../src/core/api";

const bridge = vi.hoisted(() => ({
  planMindCheck: vi.fn<CoreBridge["planMindCheck"]>(),
  estimateRun: vi.fn<CoreBridge["estimateRun"]>(),
  checkMind: vi.fn<CoreBridge["checkMind"]>(),
}));
vi.mock("../../src/renderer/src/core", () => ({ core: bridge }));
vi.mock("../../src/renderer/src/store", () => ({
  useAppStore: {
    getState: () => ({ openMind: async () => {}, settings: { language: "en" } }),
  },
}));
vi.mock("../../src/renderer/src/checks", () => ({
  useChecks: { getState: () => ({ running: {} }) },
  dismissCheckProblem: vi.fn(),
  setCheckProblem: vi.fn(),
  setCheckRunning: vi.fn(),
  setCheckSmallModel: vi.fn(),
}));

const plan: MindCheckPlan = {
  kind: "ready",
  sentences: 1,
  findsClaims: false,
  inFolder: true,
  model: null,
};
const estimate: RunEstimate = {
  billing: "per-token",
  calls: 2,
  tokens: 1000,
  cost: { amount: 0.001, currency: "USD" },
  cap: null,
  spentTowardCap: null,
  setAsideTowardCap: null,
};

beforeEach(async () => {
  vi.restoreAllMocks();
  vi.clearAllMocks();
  bridge.planMindCheck.mockResolvedValue(plan);
  bridge.estimateRun.mockResolvedValue(estimate);
  bridge.checkMind.mockResolvedValue({ kind: "cancelled" });
  const { useMindCheck } = await import("../../src/renderer/src/mindCheck");
  useMindCheck.setState({ asking: {} });
});

describe("Asking before checking against Documents added since", () => {
  test.each([true, false])("the ask uses the planned scope (inFolder: %s)", async (inFolder) => {
    bridge.planMindCheck.mockResolvedValue({ ...plan, inFolder });
    const { startMindCheck, useMindCheck, checkMind } = await import(
      "../../src/renderer/src/mindCheck"
    );
    await startMindCheck("mind", "added");
    expect(useMindCheck.getState().asking.mind).toMatchObject({ count: 1, inFolder });
    expect(bridge.checkMind).not.toHaveBeenCalled();
    await checkMind("mind", "added");
    expect(bridge.checkMind).toHaveBeenCalledOnce();
    expect(useMindCheck.getState().asking.mind).toBeUndefined();
  });

  test("a failed estimate still asks and sends no check until the User agrees", async () => {
    const error = new Error("Estimate unavailable");
    bridge.estimateRun.mockRejectedValue(error);
    const warning = vi.spyOn(console, "warn").mockImplementation(() => {});
    const { startMindCheck, useMindCheck, checkMind } = await import(
      "../../src/renderer/src/mindCheck"
    );
    await startMindCheck("mind", "added");
    expect(useMindCheck.getState().asking.mind).toMatchObject({
      count: 1,
      estimate: null,
      unestimated: true,
    });
    expect(warning).toHaveBeenCalledWith(expect.stringContaining("mind"), error);
    expect(bridge.checkMind).not.toHaveBeenCalled();
    await checkMind("mind", "added");
    expect(bridge.checkMind).toHaveBeenCalledOnce();
  });

  test.each(["free", "plan"] as const)("a known %s model checks at once", async (billing) => {
    bridge.estimateRun.mockResolvedValue({ billing });
    const { startMindCheck, useMindCheck } = await import("../../src/renderer/src/mindCheck");
    await startMindCheck("mind", "added");
    expect(bridge.checkMind).toHaveBeenCalledOnce();
    expect(useMindCheck.getState().asking.mind).toBeUndefined();
  });
});
