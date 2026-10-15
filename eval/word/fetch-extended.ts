/**
 * `npm run eval:word:fetch` (eval/word/README.md): puts the extended set in a cache folder,
 * `INCARNAMIND_WORD_CACHE` or `word-extended` in the evaluation's cache (~/.cache/incarnamind-eval).
 *
 * - Real files are downloaded from the commit-pinned URL in extended.json and checked against
 *   their SHA-256. One that doesn't match is not kept: the run fails and says which.
 * - Generated files are written by `generate-domains.py extended`, which needs Python 3.
 *
 * Files already in the cache with the right hash are not downloaded again. Everything fetched is
 * untrusted data: nothing here opens or runs it.
 */
import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

interface Entry {
  name: string;
  kind: "real" | "generated";
  url?: string;
  sha256?: string;
}

const here = dirname(fileURLToPath(import.meta.url));
// The folder eval/lib/cache.ts names: plain Node runs this script, and can't import a .ts file
// without the extension, which TypeScript doesn't allow here.
const cache =
  process.env.INCARNAMIND_WORD_CACHE ??
  join(
    process.env.INCARNAMIND_EVAL_CACHE?.trim() || join(homedir(), ".cache", "incarnamind-eval"),
    "word-extended",
  );
const entries = JSON.parse(readFileSync(join(here, "extended.json"), "utf8")) as Entry[];
const sha256 = (bytes: Uint8Array) => createHash("sha256").update(bytes).digest("hex");

const problems: string[] = [];
let downloaded = 0;
let kept = 0;
const real = entries.filter((entry) => entry.kind === "real");

async function fetchOne(entry: Entry): Promise<void> {
  const path = join(cache, entry.name);
  if (existsSync(path) && sha256(readFileSync(path)) === entry.sha256) {
    kept++;
    return;
  }
  let response: Response;
  try {
    response = await fetch(entry.url as string, { signal: AbortSignal.timeout(60_000) });
  } catch (error) {
    problems.push(`${entry.name}: ${error instanceof Error ? error.message : String(error)}`);
    return;
  }
  if (!response.ok) {
    problems.push(`${entry.name}: ${response.status} from ${entry.url}`);
    return;
  }
  const bytes = new Uint8Array(await response.arrayBuffer());
  if (sha256(bytes) !== entry.sha256) {
    problems.push(`${entry.name}: the download does not match its SHA-256, so it was not kept`);
    return;
  }
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, bytes);
  downloaded++;
}

const CONCURRENCY = 8;
for (let at = 0; at < real.length; at += CONCURRENCY) {
  await Promise.all(real.slice(at, at + CONCURRENCY).map(fetchOne));
}

let generated = 0;
if (entries.some((entry) => entry.kind === "generated")) {
  const result = spawnSync(
    "python3",
    [join(here, "generate-domains.py"), "extended", join(cache, "generated")],
    { stdio: ["ignore", "ignore", "inherit"] },
  );
  if (result.status === 0) generated = entries.filter((entry) => entry.kind === "generated").length;
  else problems.push("generate-domains.py failed (is Python 3 installed?)");
}

console.log(
  `${downloaded} downloaded, ${kept} already in ${cache}, ${generated} generated, ${problems.length} problems`,
);
for (const problem of problems) console.error(`  ${problem}`);
process.exit(problems.length > 0 ? 1 : 0);
