/**
 * cross-spawn ships no types. It takes the same arguments as Node's `spawn`
 * and returns a `ChildProcess`; on Windows it also runs `.cmd` shims such as
 * `npx.cmd` and reports a command that isn't found as an ENOENT error.
 */
declare module "cross-spawn" {
  import type { spawn } from "node:child_process";

  const crossSpawn: typeof spawn;
  export default crossSpawn;
}
