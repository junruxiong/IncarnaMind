import { readdir, readFile } from "node:fs/promises";
import { join } from "node:path";
import { type ElectronApplication, test } from "@playwright/test";
import type { AnalysisRun } from "../src/core/analysis";
import type { AnalysisJob } from "../src/shared/analysis";
import { createDataFolder, launchApp, removeDataFolder } from "./app";

type Analyses = { run(job: AnalysisJob, stopAfterMs?: number): Promise<AnalysisRun> };
const run = (app: ElectronApplication, job: AnalysisJob) =>
  app.evaluate(
    (_electron, asked) =>
      (globalThis as unknown as { incarnamindAnalysis: Analyses }).incarnamindAnalysis.run(asked),
    job,
  );
const python = (code: string): AnalysisJob => ({ language: "python", code, inputs: [] });

test("times", async () => {
  test.setTimeout(300_000);
  const dataDir = await createDataFolder();
  const { app } = await launchApp(dataDir);
  const memory = () =>
    app.evaluate(({ app: electron }) =>
      electron
        .getAppMetrics()
        .map((each) => `${each.type}:${Math.round(each.memory.workingSetSize / 1024)}MB`),
    );
  console.log("before", await memory());
  let t = performance.now();
  await run(app, python("print(1)"));
  console.log("cold job", Math.round(performance.now() - t), "ms");
  console.log("after", await memory());
  t = performance.now();
  await run(app, python("print(1)"));
  console.log("warm job", Math.round(performance.now() - t), "ms");
  await run(app, { ...python("while True: pass"), timeLimitMs: 2000 });
  t = performance.now();
  await run(app, python("print(1)"));
  console.log("job after a time-out (new window)", Math.round(performance.now() - t), "ms");
  for (const code of [
    'import urllib.request\nurllib.request.urlopen("http://127.0.0.1:9/urllib", timeout=5).read()',
    'import socket\nsocket.create_connection(("127.0.0.1", 9), 3)',
    'from pyodide.http import open_url\nopen_url("http://127.0.0.1:9/x").read()',
    'import js\nx = js.XMLHttpRequest.new()\nx.open("GET", "http://127.0.0.1:9/xhr", False)\nx.send()',
    'import js\njs.importScripts("http://127.0.0.1:9/script.js")',
    'import js\njs.eval("1 + 1")',
    'import js\njs.Function.new("return 1")()',
    'open("/etc/hosts").read()',
    'import os\nprint(os.listdir("/"))',
    "import micropip",
    'import subprocess\nsubprocess.run(["ls"])',
  ]) {
    const probe = await run(app, python(code));
    const end = probe.result.end;
    console.log(
      "PROBE",
      code.split("\n").at(-1),
      "=>",
      end.kind,
      end.kind === "error" ? end.message.split("\n").at(-1) : probe.result.output.printed,
    );
  }
  const logs = join(dataDir, "logs");
  for (const file of await readdir(logs)) {
    for (const line of (await readFile(join(logs, file), "utf8")).split("\n")) {
      if (line.includes("analysis.")) console.log(line.slice(0, 200));
    }
  }
  await app.close();
  await removeDataFolder(dataDir);
});
