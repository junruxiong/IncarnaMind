/**
 * The storage port: the only way the core talks to SQLite.
 *
 * Everything above this interface is plain SQL, so swapping the driver means
 * writing one new adapter (see `./index.ts`), not touching queries. ADR-0008
 * keeps the storage layer swappable.
 */

export type SqlValue = null | number | bigint | string | Uint8Array;

export interface Database {
  /** Runs one or more statements without parameters (migrations, pragmas). */
  exec(sql: string): void;
  run(sql: string, params?: readonly SqlValue[]): void;
  get<Row>(sql: string, params?: readonly SqlValue[]): Row | undefined;
  all<Row>(sql: string, params?: readonly SqlValue[]): Row[];
  /** Runs `fn` in a transaction: commits if it returns, rolls back if it throws. Nested calls join the outer one. */
  transaction<T>(fn: () => T): T;
  /** Closing an already closed database does nothing. */
  close(): void;
}
