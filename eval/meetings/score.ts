/**
 * `npm run eval:meetings:score -- <results folder> [--collar <seconds>] [--until-minutes <n>]`
 * (eval/meetings/README.md): scores a transcriber's output on the meeting set. The folder holds
 * a `<case id>.json` for each case it did (the `Hypothesis` in ./types); a case with none is
 * left out. Needs the cases fetched (`npm run eval:meetings:fetch`) and no network.
 */
import { readFile, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { loadCase, readCases } from "./cache";
import { DEFAULT_COLLAR } from "./der";
import { type CaseScore, formatReport, scoreCase } from "./report";
import type { Hypothesis } from "./types";

const args = process.argv.slice(2);
const flag = (name: string) => {
  const at = args.indexOf(name);
  return at < 0 ? undefined : args[at + 1];
};
const folder = args.find((arg, at) => !arg.startsWith("--") && !args[at - 1]?.startsWith("--"));
if (!folder) {
  console.error(
    "Usage: npm run eval:meetings:score -- <results folder> [--collar s] [--until-minutes n]",
  );
  process.exit(2);
}
const collar = flag("--collar") ? Number(flag("--collar")) : DEFAULT_COLLAR;
const untilMinutes = flag("--until-minutes");
const until = untilMinutes ? Number(untilMinutes) * 60 : undefined;

const scores: CaseScore[] = [];
const missing: string[] = [];
for (const definition of await readCases()) {
  let hypothesis: Hypothesis;
  try {
    hypothesis = JSON.parse(
      await readFile(join(folder, `${definition.id}.json`), "utf8"),
    ) as Hypothesis;
  } catch {
    missing.push(definition.id);
    continue;
  }
  if (hypothesis.case !== definition.id) {
    throw new Error(`${definition.id}.json says it is for ${hypothesis.case}.`);
  }
  const { reference } = await loadCase(definition);
  scores.push(scoreCase(definition, reference, hypothesis, { collar, until }));
}
if (scores.length === 0) throw new Error(`No hypothesis for any case in ${folder}.`);

const report = formatReport(scores, collar);
console.log(report);
if (missing.length > 0) console.log(`\nNo hypothesis for: ${missing.join(", ")}`);
await writeFile(join(folder, "score.md"), `${report}\n`);
await writeFile(join(folder, "score.json"), `${JSON.stringify(scores, null, 2)}\n`);
