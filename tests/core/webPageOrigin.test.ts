import { expect, test } from "vitest";
import { migrate, openDatabase } from "../../src/core/storage";
import { migrations } from "../../src/core/storage/migrations";

test("a page saved before its origin was kept reads as a web search's, until the User saves its address again", () => {
  const db = openDatabase(":memory:");
  const addsOrigin = migrations.find((each) => each.sql.includes("web_pages ADD COLUMN origin"));
  if (!addsOrigin) throw new Error("No migration adds web_pages.origin.");
  migrate(
    db,
    migrations.filter((each) => each.version < addsOrigin.version),
  );
  const at = "2026-10-01T09:00:00.000Z";
  db.run(
    `INSERT INTO documents (id, content_hash, name, kind, size, status, path, source_kind,
       relative_path, file_status, created_at, updated_at)
     VALUES ('d1', 'h1', 'Opinion', 'web', 10, 'ready', '/data/Web pages/d1.html', 'data-folder',
       'Web pages/d1.html', 'available', ?, ?)`,
    [at, at],
  );
  db.run(
    `INSERT INTO web_pages (id, document_id, content_hash, address, site_name, fetched_at,
       created_at, updated_at)
     VALUES ('w1', 'd1', 'h1', 'https://opinions.example/a', 'Opinions', ?, ?, ?)`,
    [at, at, at],
  );

  migrate(db);

  expect(db.get("SELECT origin FROM web_pages WHERE id = 'w1'")).toEqual({ origin: "search" });
  expect(() => db.run("UPDATE web_pages SET origin = 'someone else' WHERE id = 'w1'")).toThrow();
});
