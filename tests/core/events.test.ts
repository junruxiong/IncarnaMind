import { describe, expect, test, vi } from "vitest";
import { createTempDataFolder, startCore } from "../helpers/core";

describe("Core events", () => {
  test("changing settings pushes the settings now in effect to listeners", async () => {
    const core = startCore(await createTempDataFolder());
    const listener = vi.fn();
    core.on("settings.changed", listener);

    const updated = await core.updateSettings({ user: { language: "zh-CN" } });

    expect(listener).toHaveBeenCalledExactlyOnceWith(updated);
    expect(updated.language).toBe("zh-CN");
  });

  test("a listener stops receiving events once unsubscribed", async () => {
    const core = startCore(await createTempDataFolder());
    const listener = vi.fn();
    const unsubscribe = core.on("settings.changed", listener);

    unsubscribe();
    unsubscribe();
    await core.updateSettings({ device: { sidebarWidth: 300 } });

    expect(listener).not.toHaveBeenCalled();
  });

  test("the host sees every event, and a failing listener doesn't stop the others", async () => {
    const core = startCore(await createTempDataFolder());
    const seen: string[] = [];
    const errors = vi.spyOn(console, "error").mockImplementation(() => undefined);
    core.on("settings.changed", () => {
      throw new Error("listener failed");
    });
    core.onAnyEvent((event) => seen.push(event));

    await core.updateSettings({ device: { viewerWidth: 500 } });

    expect(seen).toEqual(["settings.changed"]);
    expect(errors).toHaveBeenCalledOnce();
  });
});
