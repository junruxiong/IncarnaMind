// Prints the smoke specs shard `current` of `total` runs, balanced by how long each takes
// (e2e/durations.json, seconds per spec file, from CI's logs). Playwright's own --shard cuts the
// list into runs of neighbouring files with equal numbers of tests, which left one shard with
// the slowest files and the run waiting for it. A spec the table doesn't know yet counts
// as 5 seconds a test; refresh the table now and then from a green run's log.
import { readdirSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const e2e = join(dirname(fileURLToPath(import.meta.url)), "..", "e2e");
const [current, total] = process.argv.slice(2).map(Number);
if (!(current >= 1 && total >= 1 && current <= total)) {
  console.error("Usage: node scripts/smoke-shard.mjs <current> <total>");
  process.exit(1);
}
const known = JSON.parse(readFileSync(join(e2e, "durations.json"), "utf8"));
const seconds = (file) =>
  known[file] ?? (readFileSync(join(e2e, file), "utf8").match(/^\s*test\(/gm)?.length ?? 1) * 5;

const specs = readdirSync(e2e)
  .filter((file) => file.endsWith(".spec.ts"))
  .map((file) => ({ file, seconds: seconds(file) }))
  .sort((a, b) => b.seconds - a.seconds || a.file.localeCompare(b.file));
const shards = Array.from({ length: total }, () => ({ seconds: 0, files: [] }));
for (const spec of specs) {
  const lightest = shards.reduce((a, b) => (b.seconds < a.seconds ? b : a));
  lightest.seconds += spec.seconds;
  lightest.files.push(`e2e/${spec.file}`);
}
console.log((shards[current - 1]?.files ?? []).sort().join(" "));
