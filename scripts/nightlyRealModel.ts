/**
 * `npm run nightly:real-model` (docs/agents/testing.md): the checks that ask real models, which
 * .github/workflows/nightly-real-model.yml runs every night on main. It runs, one after the other,
 *
 * - the live tests (tests/live) on Claude Haiku 5.5,
 * - the everyday requests (eval/everyday) on Claude Haiku 5.5,
 * - the live smoke test of Kimi International, and
 * - the everyday requests in Chinese on Kimi K2.6,
 *
 * and writes `summary.md` (a result for every case, and the spend) with each step's reports into
 * NIGHTLY_OUT (eval/results/nightly-real-model by default). It exits 1 when a step fails.
 *
 * The keys come from the environment, never from the command line: NIGHTLY_ANTHROPIC_KEY_1 and _2
 * (the run moves to the second for good when the first is out of credit) and NIGHTLY_KIMI_KEY.
 * NIGHTLY_STEPS (live, everyday, kimi-live, kimi-everyday, separated by commas) runs only those.
 * Kimi's terms allow it to train on what it is sent, so it only ever sees the evaluation's own
 * public and made-up Documents.
 */
import { spawn } from "node:child_process";
import {
  appendFileSync,
  cpSync,
  existsSync,
  mkdirSync,
  readdirSync,
  readFileSync,
  statSync,
  writeFileSync,
} from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { costOf, inCurrency } from "../src/core/cost/price";
import type { Price } from "../src/core/providers/catalog/types";
import type { Usage } from "./usageTally";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");

export const SECRETS = {
  anthropic: ["NIGHTLY_ANTHROPIC_KEY_1", "NIGHTLY_ANTHROPIC_KEY_2"],
  kimi: ["NIGHTLY_KIMI_KEY"],
} as const;
type Provider = keyof typeof SECRETS;

/** Per million tokens, as the catalog has them (a test holds the two together). */
export const RATES: Readonly<Record<string, Price>> = {
  "claude-haiku-5-5": { currency: "USD", input: 0.1, output: 0.5, cacheRead: 0.01 },
  "claude-sonnet-5-5": { currency: "USD", input: 2, output: 10, cacheRead: 0.1 },
  "kimi-k2.6": { currency: "CNY", input: 6.5, output: 27, cacheRead: 1.1 },
  "kimi-k3": { currency: "CNY", input: 20, output: 100, cacheRead: 2 },
};

/** What Anthropic and Kimi say when a key's credit or spending limit is used up. */
export const OUT_OF_CREDIT =
  /credit balance is too low|insufficient[ _](?:credit|balance|funds|quota)|balance (?:is )?insufficient|exceeded_current_quota|specified (?:workspace )?API usage limits|payment required/i;

interface Case {
  name: string;
  passed: boolean;
  detail?: string;
}

export interface Step {
  id: string;
  name: string;
  provider: Provider;
  results: "vitest" | "everyday";
  command: (folder: string) => string[];
  env: (key: string) => Record<string, string>;
}

const HAIKU = "claude-haiku-5-5";

const vitestRun = (files: string, folder: string) => [
  "npx",
  "vitest",
  "run",
  files,
  "--maxWorkers=2",
  "--reporter=default",
  "--reporter=json",
  `--outputFile.json=${join(folder, "vitest.json")}`,
];

const requests = (fallback?: string): Record<string, string> => {
  const asked = process.env.INCARNAMIND_EVAL_EVERYDAY?.trim() || fallback;
  return asked ? { INCARNAMIND_EVAL_EVERYDAY: asked } : {};
};

export const STEPS: readonly Step[] = [
  {
    id: "live",
    name: "Live tests on Claude Haiku 5.5",
    provider: "anthropic",
    results: "vitest",
    command: (folder) => vitestRun("tests/live", folder),
    env: (key) => ({
      INCARNAMIND_LIVE_ANTHROPIC_KEY: key,
      INCARNAMIND_LIVE_ANTHROPIC_MODEL: HAIKU,
    }),
  },
  {
    id: "everyday",
    name: "Everyday requests on Claude Haiku 5.5",
    provider: "anthropic",
    results: "everyday",
    command: () => ["npm", "run", "eval:everyday"],
    env: (key) => ({
      ...requests(),
      INCARNAMIND_EVAL_CHAT_KIND: "anthropic",
      INCARNAMIND_EVAL_CHAT_MODEL: HAIKU,
      INCARNAMIND_EVAL_CHAT_KEY: key,
    }),
  },
  {
    id: "kimi-live",
    name: "Kimi International answers",
    provider: "kimi",
    results: "vitest",
    command: (folder) => vitestRun("tests/live/providers.live.test.ts", folder),
    env: (key) => ({ MOONSHOT_API_KEY: key, KIMI_REGION: "intl" }),
  },
  {
    id: "kimi-everyday",
    name: "Everyday requests in Chinese on Kimi K2.6",
    provider: "kimi",
    results: "everyday",
    command: () => ["npm", "run", "eval:everyday"],
    env: (key) => ({
      ...requests("chinese"),
      INCARNAMIND_EVAL_CHAT_KIND: "kimi",
      INCARNAMIND_EVAL_CHAT_MODEL: "kimi-k2.6",
      INCARNAMIND_EVAL_CHAT_ENDPOINT: "intl",
      INCARNAMIND_EVAL_CHAT_KEY: key,
      INCARNAMIND_EVAL_JUDGE_KIND: "kimi",
      INCARNAMIND_EVAL_JUDGE_MODEL: "kimi-k3",
      INCARNAMIND_EVAL_JUDGE_ENDPOINT: "intl",
      INCARNAMIND_EVAL_JUDGE_KEY: key,
    }),
  },
];

/** The secrets the steps need that the environment doesn't have, or has empty. */
export function missingSecrets(
  env: Readonly<Record<string, string | undefined>>,
  steps: readonly Step[],
): string[] {
  const needed = new Set(steps.flatMap((step) => SECRETS[step.provider]));
  return [...needed].filter((name) => !env[name]?.trim());
}

export function missingMessage(names: readonly string[]): string {
  return `Missing repository secret${names.length === 1 ? "" : "s"} ${names.join(", ")}: add ${
    names.length === 1 ? "it" : "them"
  } under Settings, Secrets and variables, Actions.`;
}

export interface Attempt {
  step: Step;
  /** The name of the secret the key came from, never the key. */
  secret: string;
  exitCode: number;
  log: string;
  seconds: number;
  outOfCredit: boolean;
  skipped?: true;
}

type Execute = (
  step: Step,
  secret: string,
) => Promise<Pick<Attempt, "exitCode" | "log" | "seconds">>;

/**
 * Runs each step with the first key of its provider that still has credit. A step that fails
 * saying a key is out of credit runs again with the next one, and every step after it starts there.
 */
export async function runSteps(steps: readonly Step[], execute: Execute): Promise<Attempt[]> {
  const attempts: Attempt[] = [];
  const spent = new Set<string>();
  for (const step of steps) {
    const names = SECRETS[step.provider];
    const candidates = names.filter((name) => !spent.has(name));
    if (candidates.length === 0) {
      const secret = names.at(-1) ?? "";
      attempts.push({
        step,
        secret,
        exitCode: 1,
        log: "",
        seconds: 0,
        outOfCredit: true,
        skipped: true,
      });
      continue;
    }
    for (const secret of candidates) {
      const run = await execute(step, secret);
      const outOfCredit = run.exitCode !== 0 && OUT_OF_CREDIT.test(run.log);
      attempts.push({ step, secret, outOfCredit, ...run });
      if (!outOfCredit) break;
      spent.add(secret);
    }
  }
  return attempts;
}

type Counted = Usage & { model?: string };

/** What the requests cost in dollars at the catalog's prices, and the tokens of models it has none for. */
export function spendOf(
  usage: readonly Counted[],
  provider: Provider,
): { dollars: number; unpricedTokens: number } {
  let dollars = 0;
  let unpricedTokens = 0;
  for (const each of usage) {
    const price = each.model === undefined ? undefined : RATES[each.model];
    if (!price) {
      unpricedTokens += each.inputTokens + each.outputTokens;
      continue;
    }
    const cost = costOf(each, price, new Date(), { cacheWrites: provider === "anthropic" });
    dollars += inCurrency(cost, price.currency, "USD");
  }
  return { dollars, unpricedTokens };
}

export interface Result extends Attempt {
  cases: Case[];
  usage: Counted[];
}

const cell = (text: string) => text.replace(/\|/g, "\\|");

/** The night's report: a line for each step, then a result for every case, and what it spent. */
export function summary(
  results: readonly Result[],
  date: string,
): { text: string; passed: boolean } {
  const steps = [...new Set(results.map((result) => result.step))];
  const rows: string[] = [];
  const sections: string[] = [];
  const notes = new Set<string>();
  let total = 0;
  let unpriced = 0;
  let passed = true;
  for (const step of steps) {
    const tries = results.filter((result) => result.step === step);
    const last = tries.at(-1);
    if (!last) continue;
    for (const attempt of tries) {
      if (attempt.outOfCredit) notes.add(`${attempt.secret} is out of credit.`);
    }
    const spend = tries.map((attempt) => spendOf(attempt.usage, step.provider));
    const dollars = spend.reduce((sum, each) => sum + each.dollars, 0);
    total += dollars;
    unpriced += spend.reduce((sum, each) => sum + each.unpricedTokens, 0);
    const ok = last.exitCode === 0;
    passed &&= ok;
    const failing = last.cases.filter((each) => !each.passed).length;
    const minutes = (tries.reduce((sum, attempt) => sum + attempt.seconds, 0) / 60).toFixed(1);
    rows.push(
      `| ${step.name} | ${last.secret} | ${ok ? "passed" : "**failed**"} | ${last.cases.length - failing} of ${last.cases.length} | $${dollars.toFixed(2)} | ${minutes} min |`,
    );
    sections.push(
      `## ${step.name}`,
      "",
      ...(last.cases.length === 0
        ? [
            last.skipped
              ? "Not run: every key is out of credit."
              : "No results: read the job's log.",
            "",
          ]
        : [
            "| Case | Result |",
            "|---|---|",
            ...last.cases.map(
              (each) =>
                `| ${cell(each.name)} | ${each.passed ? "passed" : `**failed**${each.detail ? `: ${cell(each.detail)}` : ""}`} |`,
            ),
            "",
          ]),
    );
  }
  const text = [
    `# Nightly real-model checks, ${date}`,
    "",
    `${passed ? "Every step passed." : "A step failed."} Spend: $${total.toFixed(2)}${unpriced > 0 ? ` and ${unpriced.toLocaleString("en")} tokens of models the catalog has no price for` : ""}, counted from every request's tokens at the catalog's prices (the provider's console has the exact bill).`,
    "",
    ...[...notes].map((note) => `- ${note}`),
    ...(notes.size > 0 ? [""] : []),
    "| Step | Key | Result | Cases | Spend | Time |",
    "|---|---|---|---|---|---|",
    ...rows,
    "",
    ...sections,
  ].join("\n");
  return { text, passed };
}

function readJson(path: string): unknown {
  return existsSync(path) ? JSON.parse(readFileSync(path, "utf8")) : null;
}

function readUsage(folder: string): Counted[] {
  const path = join(folder, "usage.jsonl");
  if (!existsSync(path)) return [];
  return readFileSync(path, "utf8")
    .split("\n")
    .filter(Boolean)
    .map((line) => JSON.parse(line) as Counted);
}

interface VitestReport {
  testResults?: {
    assertionResults?: { fullName: string; status: string; failureMessages?: string[] }[];
  }[];
}

interface EverydayReport {
  rows?: {
    id: string;
    language: string;
    passed: number;
    runs: number;
    floor: number;
    under: boolean;
  }[];
}

function casesOf(step: Step, folder: string): Case[] {
  if (step.results === "everyday") {
    const report = readJson(join(folder, "everyday", "report.json")) as EverydayReport | null;
    return (report?.rows ?? []).map((row) => ({
      name: `${row.id} (${row.language})`,
      passed: !row.under,
      detail: `${row.passed} of ${row.runs} runs passed, floor ${Math.round(row.floor * 100)}%`,
    }));
  }
  const report = readJson(join(folder, "vitest.json")) as VitestReport | null;
  return (report?.testResults ?? []).flatMap((file) =>
    (file.assertionResults ?? [])
      .filter((each) => each.status === "passed" || each.status === "failed")
      .map((each) => {
        const reason = each.failureMessages?.[0]?.split("\n")[0]?.slice(0, 200);
        return {
          name: each.fullName,
          passed: each.status === "passed",
          ...(reason && { detail: reason }),
        };
      }),
  );
}

const everydayReports = () => {
  const results = join(ROOT, "eval", "results");
  return new Set(
    existsSync(results) ? readdirSync(results).filter((n) => n.endsWith("-everyday")) : [],
  );
};

function executeIn(out: string, env: Readonly<Record<string, string | undefined>>): Execute {
  const preload = pathToFileURL(join(ROOT, "scripts", "usageTally.ts")).href;
  return (step, secret) => {
    const folder = join(out, step.id, secret);
    mkdirSync(folder, { recursive: true });
    const child = { ...env, ...step.env(env[secret] ?? "") };
    for (const name of Object.values(SECRETS).flat()) delete child[name];
    child.NIGHTLY_USAGE_FILE = join(folder, "usage.jsonl");
    child.NODE_OPTIONS = `${env.NODE_OPTIONS ?? ""} --import ${preload}`.trim();
    const before = everydayReports();
    const begun = Date.now();
    const [command = "", ...args] = step.command(folder);
    return new Promise((done) => {
      const run = spawn(command, args, {
        cwd: ROOT,
        env: child,
        stdio: ["ignore", "pipe", "pipe"],
      });
      let log = "";
      for (const [stream, to] of [
        [run.stdout, process.stdout],
        [run.stderr, process.stderr],
      ] as const) {
        stream.on("data", (chunk: Buffer) => {
          log += chunk.toString();
          to.write(chunk);
        });
      }
      const finish = (exitCode: number) => {
        for (const name of everydayReports()) {
          if (before.has(name)) continue;
          cpSync(join(ROOT, "eval", "results", name), join(folder, "everyday"), {
            recursive: true,
          });
        }
        done({ exitCode, log, seconds: (Date.now() - begun) / 1000 });
      };
      run.on("error", (error) => {
        log += error.message;
        finish(1);
      });
      run.on("close", (code) => finish(code ?? 1));
    });
  };
}

const withoutKeys = (text: string, keys: readonly string[]) =>
  keys.reduce((rest, key) => rest.split(key).join("[key]"), text);

/** Takes a key out of every file in a folder, should a report have caught one. */
function removeKeys(folder: string, keys: readonly string[]): void {
  for (const entry of readdirSync(folder)) {
    const path = join(folder, entry);
    if (statSync(path).isDirectory()) {
      removeKeys(path, keys);
      continue;
    }
    const text = readFileSync(path, "utf8");
    const clean = withoutKeys(text, keys);
    if (clean !== text) writeFileSync(path, clean);
  }
}

async function main(): Promise<void> {
  const env = process.env;
  const ids = (env.NIGHTLY_STEPS ?? "")
    .split(",")
    .map((id) => id.trim())
    .filter(Boolean);
  const unknown = ids.filter((id) => !STEPS.some((step) => step.id === id));
  if (unknown.length > 0) {
    console.error(
      `NIGHTLY_STEPS names no step ${unknown.join(", ")}: the steps are ${STEPS.map((step) => step.id).join(", ")}.`,
    );
    process.exit(1);
  }
  const steps = ids.length === 0 ? STEPS : STEPS.filter((step) => ids.includes(step.id));
  const missing = missingSecrets(env, steps);
  if (missing.length > 0) {
    console.error(`${env.GITHUB_ACTIONS ? "::error::" : ""}${missingMessage(missing)}`);
    process.exit(1);
  }
  const out = resolve(ROOT, env.NIGHTLY_OUT ?? join("eval", "results", "nightly-real-model"));
  mkdirSync(out, { recursive: true });
  const attempts = await runSteps(steps, executeIn(out, env));
  const results = attempts.map((attempt) => {
    const folder = join(out, attempt.step.id, attempt.secret);
    return { ...attempt, cases: casesOf(attempt.step, folder), usage: readUsage(folder) };
  });
  const keys = Object.values(SECRETS)
    .flat()
    .map((name) => env[name])
    .filter((key): key is string => Boolean(key));
  const night = summary(results, new Date().toISOString().slice(0, 10));
  const text = withoutKeys(night.text, keys);
  writeFileSync(join(out, "summary.md"), text);
  if (env.GITHUB_STEP_SUMMARY) appendFileSync(env.GITHUB_STEP_SUMMARY, `${text}\n`);
  removeKeys(out, keys);
  console.log(text);
  process.exit(night.passed ? 0 : 1);
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) await main();
