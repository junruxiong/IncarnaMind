import { join } from "node:path";
import { describe, expect, test } from "vitest";
import { InvalidInputError } from "../../src/core";
import { createTempDataFolder, startCore, tickingClock } from "../helpers/core";

const UUID_V4 = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/;

describe("Minds", () => {
  test("a new data folder has no Minds", async () => {
    const core = startCore(await createTempDataFolder());

    expect(await core.listMinds()).toEqual([]);
  });

  test("creating a Mind returns it with a random UUID and timestamps", async () => {
    const core = startCore(await createTempDataFolder(), {
      now: () => new Date("2026-10-06T12:00:00Z"),
    });

    const mind = await core.createMind({ title: "  Reading list  " });

    expect(mind).toEqual({
      id: expect.stringMatching(UUID_V4),
      title: "Reading list",
      createdAt: "2026-10-06T12:00:00.000Z",
      updatedAt: "2026-10-06T12:00:00.000Z",
    });
    expect(await core.listMinds()).toEqual([mind]);
  });

  test("a Mind created without a title has an empty title", async () => {
    const core = startCore(await createTempDataFolder());

    const mind = await core.createMind();

    expect(mind.title).toBe("");
    expect(await core.listMinds()).toEqual([mind]);
  });

  test("every Mind gets its own id", async () => {
    const core = startCore(await createTempDataFolder());

    const minds = await Promise.all([core.createMind(), core.createMind(), core.createMind()]);

    expect(new Set(minds.map((mind) => mind.id)).size).toBe(3);
  });

  test("Minds are listed most recently updated first", async () => {
    const core = startCore(await createTempDataFolder(), { now: tickingClock() });

    const first = await core.createMind({ title: "First" });
    const second = await core.createMind({ title: "Second" });
    const third = await core.createMind({ title: "Third" });

    expect(await core.listMinds()).toEqual([third, second, first]);
  });

  test("Minds created before a restart are still there after it", async () => {
    const dataDir = await createTempDataFolder();
    const before = startCore(dataDir, { now: tickingClock() });
    const papers = await before.createMind({ title: "Papers" });
    const contracts = await before.createMind({ title: "Contracts" });
    before.close();

    const after = startCore(dataDir);

    expect(await after.listMinds()).toEqual([contracts, papers]);
  });

  test("the data folder is created if it doesn't exist yet", async () => {
    const dataDir = join(await createTempDataFolder(), "nested", "IncarnaMind");
    const core = startCore(dataDir);

    const mind = await core.createMind({ title: "Fresh start" });

    expect(await core.listMinds()).toEqual([mind]);
  });

  test("a title that isn't text is rejected", async () => {
    const core = startCore(await createTempDataFolder());

    await expect(core.createMind({ title: 42 } as never)).rejects.toThrow(InvalidInputError);
    expect(await core.listMinds()).toEqual([]);
  });
});
