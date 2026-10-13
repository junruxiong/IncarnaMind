import { execFileSync } from "node:child_process";
import { userInfo } from "node:os";
import type { Logger } from "../core/adapters";

let known: string | null = null;

const messageOf = (error: unknown) => (error instanceof Error ? error.message : String(error));

/**
 * The User's full name as macOS keeps it for their account ("Priya Raman"),
 * else the account's name: what Word itself would put on their tracked
 * changes (#107). Read once, the first time it is asked for.
 */
export function systemUserName(log: Logger): string {
  if (known !== null) return known;
  let name = "";
  if (process.platform === "darwin") {
    try {
      name = execFileSync("/usr/bin/id", ["-F"], { encoding: "utf8", timeout: 2000 }).trim();
    } catch (error) {
      log.warn("userName.fullName", { message: messageOf(error) });
    }
  }
  if (!name) {
    try {
      name = userInfo().username;
    } catch (error) {
      log.warn("userName.account", { message: messageOf(error) });
    }
  }
  known = name;
  return name;
}
