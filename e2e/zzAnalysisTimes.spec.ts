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
  const { app } = await launchApp(dataDir, { sandboxed: true });
  const memory = () =>
    app.evaluate(({ app: electron, webContents }) => {
      const page = webContents
        .getAllWebContents()
        .find((each) => each.getURL().startsWith("incarnamind-analysis:"));
      const pid = page?.getOSProcessId();
      const tab = electron.getAppMetrics().find((each) => each.pid === pid);
      return tab ? Math.round(tab.memory.workingSetSize / 1024) : null;
    });
  const timed = async (label: string, code: string) => {
    const t = performance.now();
    const ran = await run(app, python(code));
    console.log(
      label,
      Math.round(performance.now() - t),
      "ms",
      ran.result.end.kind,
      "job",
      ran.result.ms,
      "ms",
    );
  };
  await timed("cold job", "print(1)");
  await new Promise((resolve) => setTimeout(resolve, 12_000));
  console.log("window memory with the spare ready", await memory(), "MB");
  await timed("job on the ready spare", "print(1)");
  await timed("job right after another", "print(1)");
  let peak = 0;
  const sampling = setInterval(async () => {
    peak = Math.max(peak, (await memory()) ?? 0);
  }, 200);
  await timed(
    "a 3 s job",
    "import time\nstart = time.perf_counter()\nwhile time.perf_counter() - start < 3: pass",
  );
  clearInterval(sampling);
  console.log("window memory during a job, at most", peak, "MB");
  await app.close();
  await removeDataFolder(dataDir);
});
