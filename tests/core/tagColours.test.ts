import { join } from "node:path";
import { describe, expect, test } from "vitest";
import { InvalidInputError } from "../../src/core";
import { migrate, migrations, openDatabase } from "../../src/core/storage";
import { nextTagColour, TAG_COLOURS } from "../../src/shared/tagColours";
import { createTempDataFolder, startCore } from "../helpers/core";
import { tagNamed } from "../helpers/tags";

describe("Tag colours", () => {
  test("the preset Tags come with their own colours, all different", async () => {
    const core = startCore(await createTempDataFolder());
    const colours = Object.fromEntries(
      (await core.listTags()).map((tag) => [tag.name, tag.colour]),
    );
    expect(colours).toEqual({
      Book: "brick",
      Contract: "indigo",
      Invoice: "rose",
      Notes: "taupe",
      Paper: "violet",
      Report: "petrol",
      Slides: "orchid",
    });
  });

  test("a new Tag takes the colour fewest Tags have, unless the User chooses one", async () => {
    const core = startCore(await createTempDataFolder());
    // Only stone is unused by the presets.
    expect((await core.createTag({ name: "Urgent" })).colour).toBe("stone");
    // Then every colour is on one Tag: the first in the palette again.
    expect((await core.createTag({ name: "Later" })).colour).toBe("stone");
    expect((await core.createTag({ name: "Travel", colour: "rose" })).colour).toBe("rose");
    expect((await core.createTag({ name: "Ideas" })).colour).toBe("taupe");
    await expect(core.createTag({ name: "Odd", colour: "green" as never })).rejects.toThrow(
      InvalidInputError,
    );
  });

  test("the User changes a Tag's colour; it stays through a merge and a restart", async () => {
    const dataDir = await createTempDataFolder();
    const core = startCore(dataDir);
    const report = await tagNamed(core, "Report");
    const notes = await tagNamed(core, "Notes");
    expect(await core.updateTag(report.id, { colour: "indigo" })).toMatchObject({
      colour: "indigo",
      name: "Report",
    });
    await expect(core.updateTag(report.id, { colour: "blue" as never })).rejects.toThrow(
      InvalidInputError,
    );
    // The Tag merged into keeps its own colour.
    expect((await core.mergeTags(notes.id, report.id)).colour).toBe("indigo");
    core.close();
    const reopened = startCore(dataDir);
    expect((await tagNamed(reopened, "Report")).colour).toBe("indigo");
  });

  test("Tags made before colours get them: presets their own, the User's in turn", async () => {
    const db = openDatabase(join(await createTempDataFolder(), "test.db"));
    migrate(
      db,
      migrations.filter((migration) => migration.version < 27),
    );
    const at = "2026-10-01T00:00:00.000Z";
    const insert = (id: string, name: string, preset: string | null, created: string) =>
      db.run(
        "INSERT INTO tags (id, name, description, preset, created_at, updated_at) VALUES (?, ?, '', ?, ?, ?)",
        [id, name, preset, created, at],
      );
    insert("a", "Paper", "paper", at);
    insert("b", "Second", null, "2026-10-03T00:00:00.000Z");
    insert("c", "First", null, "2026-10-02T00:00:00.000Z");
    insert("d", "Invoice", "invoice", at);

    migrate(db);

    expect(db.all("SELECT id, colour FROM tags ORDER BY id")).toEqual([
      { id: "a", colour: "violet" },
      { id: "b", colour: "taupe" },
      { id: "c", colour: "stone" },
      { id: "d", colour: "rose" },
    ]);
    db.close();
  });
});

describe("the next Tag colour", () => {
  test("is the least used, the earliest in the palette on a tie", () => {
    expect(nextTagColour([])).toBe(TAG_COLOURS[0]);
    expect(nextTagColour(["stone"])).toBe("taupe");
    expect(nextTagColour([...TAG_COLOURS])).toBe("stone");
    expect(nextTagColour([...TAG_COLOURS, "stone", "taupe"])).toBe("brick");
  });
});
