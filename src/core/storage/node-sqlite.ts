import { DatabaseSync, type StatementSync } from "node:sqlite";
import type { Database, SqlValue } from "./database";

/**
 * The `node:sqlite` adapter for the storage port. `node:sqlite` ships with
 * Electron's Node (24), so it needs no native module build.
 */
export function openNodeSqlite(file: string): Database {
  const db = new DatabaseSync(file);
  db.exec("PRAGMA journal_mode = WAL");
  db.exec("PRAGMA foreign_keys = ON");
  db.exec("PRAGMA busy_timeout = 5000");

  const statements = new Map<string, StatementSync>();
  const prepare = (sql: string): StatementSync => {
    let statement = statements.get(sql);
    if (!statement) {
      statement = db.prepare(sql);
      statements.set(sql, statement);
    }
    return statement;
  };
  let transactionDepth = 0;

  return {
    exec(sql) {
      db.exec(sql);
    },
    run(sql, params: readonly SqlValue[] = []) {
      prepare(sql).run(...params);
    },
    get<Row>(sql: string, params: readonly SqlValue[] = []) {
      return prepare(sql).get(...params) as Row | undefined;
    },
    all<Row>(sql: string, params: readonly SqlValue[] = []) {
      return prepare(sql).all(...params) as Row[];
    },
    transaction(fn) {
      if (transactionDepth > 0) return fn();
      db.exec("BEGIN IMMEDIATE");
      transactionDepth++;
      try {
        const result = fn();
        db.exec("COMMIT");
        return result;
      } catch (error) {
        db.exec("ROLLBACK");
        throw error;
      } finally {
        transactionDepth--;
      }
    },
    close() {
      if (!db.isOpen) return;
      statements.clear();
      db.close();
    },
  };
}
