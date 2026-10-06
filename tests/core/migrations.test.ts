import { join } from "node:path";
import { describe, expect, test } from "vitest";
import { migrate, openDatabase } from "../../src/core/storage";
import { createTempDataFolder } from "../helpers/core";

const userVersion = (db: ReturnType<typeof openDatabase>) =>
  db.get<{ user_version: number }>("PRAGMA user_version")?.user_version;

describe("Migrations", () => {
  test("version numbers may skip numbers reserved by work in progress", async () => {
    const db = openDatabase(join(await createTempDataFolder(), "test.db"));

    migrate(db, [
      { version: 1, description: "one", sql: "CREATE TABLE one (id TEXT) STRICT;" },
      { version: 4, description: "four", sql: "CREATE TABLE four (id TEXT) STRICT;" },
    ]);

    expect(userVersion(db)).toBe(4);
    db.close();
  });

  test("versions out of order are refused before anything runs", async () => {
    const db = openDatabase(join(await createTempDataFolder(), "test.db"));

    expect(() =>
      migrate(db, [
        { version: 4, description: "four", sql: "CREATE TABLE four (id TEXT) STRICT;" },
        { version: 2, description: "two", sql: "CREATE TABLE two (id TEXT) STRICT;" },
      ]),
    ).toThrow(/must increase/);
    expect(userVersion(db)).toBe(0);
    db.close();
  });
});
