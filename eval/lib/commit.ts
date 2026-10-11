import { execFileSync } from "node:child_process";

/** The short commit the working tree is on, marked when it has uncommitted changes. */
export function commitOf(root: string): string {
  const git = (...args: string[]) =>
    execFileSync("git", args, { cwd: root, encoding: "utf8" }).trim();
  try {
    const changed = git("status", "--porcelain") !== "";
    return `${git("rev-parse", "--short", "HEAD")}${changed ? " (with uncommitted changes)" : ""}`;
  } catch {
    return "unknown";
  }
}
