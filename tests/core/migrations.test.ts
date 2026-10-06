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

  test("a migration with a lower number that lands later still runs", async () => {
    const db = openDatabase(join(await createTempDataFolder(), "test.db"));
    const one = { version: 1, description: "one", sql: "CREATE TABLE one (id TEXT) STRICT;" };
    const six = { version: 6, description: "six", sql: "CREATE TABLE six (id TEXT) STRICT;" };
    const five = { version: 5, description: "five", sql: "CREATE TABLE five (id TEXT) STRICT;" };

    migrate(db, [one, six]);
    migrate(db, [one, five, six]);

    expect(db.get("SELECT name FROM sqlite_master WHERE name = 'five'")).toBeDefined();
    expect(userVersion(db)).toBe(6);
    db.close();
  });

  test("a database with a migration this version doesn't know is refused", async () => {
    const db = openDatabase(join(await createTempDataFolder(), "test.db"));
    const one = { version: 1, description: "one", sql: "CREATE TABLE one (id TEXT) STRICT;" };
    const two = { version: 2, description: "two", sql: "CREATE TABLE two (id TEXT) STRICT;" };

    migrate(db, [one, two]);

    expect(() => migrate(db, [one])).toThrow(/newer version/);
    db.close();
  });

  test("a database from before migrations were recorded keeps what it applied", async () => {
    const db = openDatabase(join(await createTempDataFolder(), "test.db"));
    db.exec("CREATE TABLE one (id TEXT) STRICT; PRAGMA user_version = 1;");

    migrate(db, [
      { version: 1, description: "one", sql: "CREATE TABLE one (id TEXT) STRICT;" },
      { version: 2, description: "two", sql: "CREATE TABLE two (id TEXT) STRICT;" },
    ]);

    expect(db.get("SELECT name FROM sqlite_master WHERE name = 'two'")).toBeDefined();
    expect(userVersion(db)).toBe(2);
    db.close();
  });
});
