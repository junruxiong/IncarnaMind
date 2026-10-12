import type { LogFields, Logger } from "../../src/core";

export interface LogEntry {
  level: "info" | "warn" | "error";
  event: string;
  fields: LogFields | undefined;
}

/** A log that keeps what it is told. */
export function recordingLog(): { log: Logger; entries: LogEntry[] } {
  const entries: LogEntry[] = [];
  const record = (level: LogEntry["level"]) => (event: string, fields?: LogFields) => {
    entries.push({ level, event, fields });
  };
  return { log: { info: record("info"), warn: record("warn"), error: record("error") }, entries };
}
