import { join } from "node:path";
import { openDatabase } from "../../src/core/storage";

const FULL = "SELECT RAISE(ABORT, 'database or disk is full')";

/**
 * Makes a data folder's database refuse writes as a full disk does (#1119), through
 * triggers on synthetic data, until the returned function is called: of Mind content only,
 * or, `every`, of every table.
 */
export function refuseWrites(dataDir: string, { every = false }: { every?: boolean } = {}) {
  // `DATABASE_FILE` by its value: the core's index would bring its workers into a smoke test.
  const db = openDatabase(join(dataDir, "incarnamind.db"));
  const tables = every
    ? db
        .all<{ name: string }>(
          "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' AND sql NOT LIKE 'CREATE VIRTUAL%'",
        )
        .map(({ name }) => name)
    : ["mind_updates"];
  const triggers: string[] = [];
  for (const table of tables) {
    for (const action of every ? ["INSERT", "UPDATE", "DELETE"] : ["INSERT"]) {
      const name = `refuse_${action}_${table}`;
      try {
        db.exec(`CREATE TRIGGER "${name}" BEFORE ${action} ON "${table}" BEGIN ${FULL}; END`);
        triggers.push(name);
      } catch {
        // A full-text index's own tables take no triggers; what writes them writes a table above.
      }
    }
  }
  return () => {
    for (const name of triggers) db.exec(`DROP TRIGGER "${name}"`);
    db.close();
  };
}
