/**
 * The feel check (#1242, DESIGN.md › Feel): records in the window what a User feels on a flow,
 * step by step, with the browser's own measures (PerformanceObserver, the Event Timing API).
 * - Layout shifts no recent input caused, with the element that moved: any above
 *   `SHIFT_LIMIT` fails, since nothing should move that the User didn't move.
 * - Input to next paint for each step's input, and long tasks: reported for every run, failing
 *   only above ceilings no CI runner reaches by chance (`INPUT_CEILING_MS`, `LONG_TASK_CEILING_MS`).
 * - Where focus is after a close, a create or a delete: never on the page body.
 * Each flow's table goes to the job summary on CI, and a video of the flow is kept beside it.
 * What main does wrong today is listed in `KNOWN` with its issue, so the check is strict for
 * everything else; a listed problem that no longer happens fails too, so the list stays true.
 */
import { spawn } from "node:child_process";
import { appendFile, mkdir, rm, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { type ElectronApplication, expect, type Locator, type Page, test } from "@playwright/test";
import { type LaunchOptions, launchApp } from "./app";

/** A layout shift's score above which it fails: a shift this small is a pixel of rounding. */
export const SHIFT_LIMIT = 0.001;
/**
 * Input to next paint above which a step fails. The target is 100 ms on a laptop; CI's runners
 * are two to five times slower and share their cores, so the build fails only five times over.
 */
export const INPUT_CEILING_MS = 500;
/** A long task above which a step fails: ten times the 100 ms target, never reached by chance. */
export const LONG_TASK_CEILING_MS = 1000;

/** Where each flow's report and video go: uploaded by CI (`.github/workflows/ci.yml`). */
export const FEEL_FOLDER = join(__dirname, "..", "test-results", "feel");

/** A problem main has today, filed as its issue: reported, but it doesn't fail the build. */
interface Known {
  flow: string;
  step: string;
  kind: Problem["kind"];
  issue: number;
}

export const KNOWN: readonly Known[] = [];

interface Shift {
  value: number;
  /** What moved, by its nearest test id, else its tag and classes. */
  moved: string[];
}

interface StepRecord {
  name: string;
  shifts: Shift[];
  /** The longest input to next paint among the step's inputs, in ms; null: no input. */
  input: number | null;
  /** The step's long tasks, in ms. */
  longTasks: number[];
  /** Where focus went, when the step checks it. */
  focus: string | null;
}

type Problem =
  | { kind: "shift"; step: string; detail: string }
  | { kind: "focus"; step: string; detail: string }
  | { kind: "input"; step: string; detail: string }
  | { kind: "long-task"; step: string; detail: string };

interface FeelEntries {
  shifts: { at: number; value: number; moved: string[] }[];
  events: { at: number; duration: number }[];
  longTasks: { at: number; duration: number }[];
}

/** Starts the window's observers: buffered, so what happened before is there too. */
async function observe(window: Page): Promise<void> {
  await window.evaluate(() => {
    const holder = globalThis as unknown as { feel?: FeelEntries };
    if (holder.feel) return;
    const feel: FeelEntries = { shifts: [], events: [], longTasks: [] };
    holder.feel = feel;
    const describe = (node: Node | null): string => {
      const element = node instanceof Element ? node : (node?.parentElement ?? null);
      if (!element) return "(gone)";
      const named = element.closest("[data-testid]");
      if (named) return named.getAttribute("data-testid") ?? "";
      const classes = [...element.classList].slice(0, 2).join(".");
      return `${element.tagName.toLowerCase()}${classes ? `.${classes}` : ""}`;
    };
    // TypeScript's DOM types have no layout shift yet.
    const isShift = (
      entry: PerformanceEntry,
    ): entry is PerformanceEntry & {
      value: number;
      hadRecentInput: boolean;
      sources?: { node: Node | null }[];
    } => "value" in entry && "hadRecentInput" in entry;
    new PerformanceObserver((list) => {
      for (const shift of list.getEntries()) {
        if (!isShift(shift) || shift.hadRecentInput) continue;
        feel.shifts.push({
          at: shift.startTime,
          value: shift.value,
          moved: [...new Set((shift.sources ?? []).map((source) => describe(source.node)))],
        });
      }
    }).observe({ type: "layout-shift", buffered: true });
    new PerformanceObserver((list) => {
      for (const entry of list.getEntries()) {
        feel.longTasks.push({ at: entry.startTime, duration: entry.duration });
      }
    }).observe({ type: "longtask", buffered: true });
    // Every input slower than a frame, not only those over the default 104 ms.
    const events: PerformanceObserverInit & { durationThreshold: number } = {
      type: "event",
      buffered: true,
      durationThreshold: 16,
    };
    new PerformanceObserver((list) => {
      for (const event of list.getEntries()) {
        if (!(event instanceof PerformanceEventTiming) || event.interactionId === 0) continue;
        feel.events.push({ at: event.startTime, duration: event.duration });
      }
    }).observe(events);
  });
}

/** The page's clock now, after two frames: whatever the last step changed has been painted. */
const paintedAt = (window: Page) =>
  window.evaluate(
    () =>
      new Promise<number>((resolve) =>
        requestAnimationFrame(() => requestAnimationFrame(() => resolve(performance.now()))),
      ),
  );

const entriesBetween = (window: Page, from: number, to: number) =>
  window.evaluate(
    ({ from, to }) => {
      const { feel } = globalThis as unknown as { feel: FeelEntries };
      const inside = (entry: { at: number }) => entry.at >= from && entry.at < to;
      return {
        shifts: feel.shifts.filter(inside),
        events: feel.events.filter(inside),
        longTasks: feel.longTasks.filter(inside),
      };
    },
    { from, to },
  );

/** What has focus, as a step's check names it: its test id, else its tag; "body" for none. */
const focused = (window: Page) =>
  window.evaluate(() => {
    const element = document.activeElement;
    if (!element || element === document.body) return "body";
    const named = element.closest("[data-testid]");
    return named?.getAttribute("data-testid") ?? element.tagName.toLowerCase();
  });

export interface Feel {
  /** Runs one step of the flow, its action waiting for the state it leads to, and records it. */
  step(name: string, action: () => Promise<void>): Promise<void>;
  /**
   * After a close, a create or a delete: focus must be on `target` (where the opener, the new
   * name or the next item is), never on the page body.
   */
  expectFocus(step: string, target: Locator): Promise<void>;
  /** Records what is left and writes the report: what fails, and what `KNOWN` lists that didn't happen. */
  finish(): Promise<{ failing: string[]; stale: string[] }>;
}

/** Starts recording a flow in `window`, from now: what happened before its first step is its "start". */
export async function feelOf(window: Page, flow: string): Promise<Feel> {
  await observe(window);
  const steps: StepRecord[] = [];
  const problems: Problem[] = [];
  let from = 0;

  const record = async (name: string, focus: string | null = null) => {
    const to = await paintedAt(window);
    const entries = await entriesBetween(window, from, to);
    from = to;
    const shifts = entries.shifts
      .filter((shift) => shift.value > 0)
      .map(({ value, moved }) => ({ value, moved }));
    const input = entries.events.length
      ? Math.max(...entries.events.map((event) => event.duration))
      : null;
    const longTasks = entries.longTasks.map((task) => task.duration);
    steps.push({ name, shifts, input, longTasks, focus });
    for (const shift of shifts.filter((each) => each.value > SHIFT_LIMIT)) {
      problems.push({
        kind: "shift",
        step: name,
        detail: `${shift.value.toFixed(4)} moving ${shift.moved.join(", ") || "(unknown)"}`,
      });
    }
    if (input !== null && input > INPUT_CEILING_MS) {
      problems.push({ kind: "input", step: name, detail: `${Math.round(input)} ms to next paint` });
    }
    for (const task of longTasks.filter((each) => each > LONG_TASK_CEILING_MS)) {
      problems.push({ kind: "long-task", step: name, detail: `${Math.round(task)} ms` });
    }
  };

  return {
    async step(name, action) {
      if (steps.length === 0) await record("start");
      await test.step(`${flow}: ${name}`, action);
      await record(name);
    },

    async expectFocus(step, target) {
      const at = await focused(window);
      const onTarget = await target.evaluate((element) => {
        const active = document.activeElement;
        return active !== null && (element === active || element.contains(active));
      });
      const last = steps.findLast((each) => each.name === step);
      if (last) last.focus = at;
      if (!onTarget) {
        problems.push({
          kind: "focus",
          step,
          detail: `on ${at}, not the expected ${await target.evaluate((element) => element.closest("[data-testid]")?.getAttribute("data-testid") ?? element.tagName.toLowerCase())}`,
        });
      }
    },

    async finish() {
      await record("end");
      const known = (problem: Problem) =>
        KNOWN.find(
          (each) => each.flow === flow && each.step === problem.step && each.kind === problem.kind,
        );
      // A time depends on the runner, so a known slow step may not be slow this run: only a shift
      // or a focus that no longer happens is stale.
      const stale = KNOWN.filter(
        (each) =>
          each.flow === flow &&
          (each.kind === "shift" || each.kind === "focus") &&
          !problems.some((problem) => problem.step === each.step && problem.kind === each.kind),
      );
      const report = tableOf(flow, steps, problems, known);
      await mkdir(FEEL_FOLDER, { recursive: true });
      await writeFile(join(FEEL_FOLDER, `${flow}.md`), report);
      await writeFile(
        join(FEEL_FOLDER, `${flow}.json`),
        JSON.stringify({ steps, problems }, null, 2),
      );
      if (process.env.GITHUB_STEP_SUMMARY)
        await appendFile(process.env.GITHUB_STEP_SUMMARY, report);
      return {
        failing: problems
          .filter((problem) => !known(problem))
          .map((problem) => `${problem.step}: ${problem.kind} ${problem.detail}`),
        stale: stale.map((each) => `${each.step}: ${each.kind} (#${each.issue})`),
      };
    },
  };
}

function tableOf(
  flow: string,
  steps: readonly StepRecord[],
  problems: readonly Problem[],
  known: (problem: Problem) => Known | undefined,
): string {
  const cell = (step: StepRecord) => {
    const shift = step.shifts.reduce((sum, each) => sum + each.value, 0);
    const marks = problems
      .filter((problem) => problem.step === step.name)
      .map((problem) => {
        const listed = known(problem);
        return `${listed ? `known #${listed.issue}` : "**fails**"}: ${problem.kind} ${problem.detail}`;
      });
    return [
      step.name,
      shift === 0 ? "0" : shift.toFixed(4),
      step.input === null ? "–" : `${Math.round(step.input)} ms`,
      step.longTasks.length === 0
        ? "–"
        : `${step.longTasks.length}, longest ${Math.round(Math.max(...step.longTasks))} ms`,
      step.focus ?? "–",
      marks.join("; ") || "ok",
    ].join(" | ");
  };
  return [
    `### Feel: ${flow}`,
    "",
    "Step | Unexpected shift | Input to next paint | Long tasks | Focus | Result",
    "--- | --- | --- | --- | --- | ---",
    ...steps.map(cell),
    "",
    "",
  ].join("\n");
}

/** Films the window from now: each painted frame to disk, with when it was painted. */
async function film(window: Page, folder: string): Promise<() => Promise<void>> {
  await rm(folder, { recursive: true, force: true });
  await mkdir(folder, { recursive: true });
  const session = await window.context().newCDPSession(window);
  const frames: { file: string; at: number }[] = [];
  const writing: Promise<void>[] = [];
  session.on("Page.screencastFrame", ({ data, metadata, sessionId }) => {
    const file = `${String(frames.length).padStart(5, "0")}.jpg`;
    frames.push({ file, at: metadata.timestamp ?? 0 });
    writing.push(writeFile(join(folder, file), Buffer.from(data, "base64")));
    // A frame can arrive as the window closes, when there is nothing left to acknowledge.
    void session.send("Page.screencastFrameAck", { sessionId }).catch(() => undefined);
  });
  await session.send("Page.startScreencast", { format: "jpeg", quality: 70, everyNthFrame: 1 });
  return async () => {
    await session.send("Page.stopScreencast").catch(() => undefined);
    await session.detach().catch(() => undefined);
    await Promise.all(writing);
    // Each frame stays on screen until the next was painted: ffmpeg's concat list keeps that time.
    const list = frames.flatMap((frame, index) => {
      const next = frames[index + 1];
      return [
        `file '${frame.file}'`,
        `duration ${next ? Math.max(next.at - frame.at, 0.001) : 0.5}`,
      ];
    });
    const last = frames.at(-1);
    if (last) list.push(`file '${last.file}'`);
    await writeFile(join(folder, "frames.txt"), `${list.join("\n")}\n`);
  };
}

/** Makes the frames a video when ffmpeg is there (CI installs it), then drops them; else keeps them. */
async function encode(folder: string, video: string): Promise<void> {
  const made = await new Promise<boolean>((resolve) => {
    const ffmpeg = spawn(
      "ffmpeg",
      [
        ["-y", "-loglevel", "error"],
        ["-f", "concat", "-safe", "0", "-i", "frames.txt"],
        // H.264 needs even sides; the window's may be odd.
        ["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"],
        ["-fps_mode", "vfr", "-pix_fmt", "yuv420p", video],
      ].flat(),
      { cwd: folder, stdio: "ignore" },
    );
    let missing = false;
    ffmpeg.on("error", (error) => {
      missing = true;
      console.warn(`No video of ${video}, its frames are in ${folder}: ${error.message}`);
      resolve(false);
    });
    ffmpeg.on("close", (code) => {
      if (code !== 0 && !missing) console.warn(`ffmpeg stopped with ${code} making ${video}`);
      resolve(code === 0);
    });
  });
  if (made) await rm(folder, { recursive: true, force: true });
}

export interface FeelRun {
  app: ElectronApplication;
  window: Page;
  feel: Feel;
  flow: string;
  stopFilming(): Promise<void>;
}

/** Launches the app for a flow, filming its window. */
export async function startFeel(
  dataDir: string,
  flow: string,
  options: LaunchOptions = {},
): Promise<FeelRun> {
  await mkdir(FEEL_FOLDER, { recursive: true });
  const { app, window } = await launchApp(dataDir, options);
  const stopFilming = await film(window, join(FEEL_FOLDER, `${flow}-frames`));
  return { app, window, feel: await feelOf(window, flow), flow, stopFilming };
}

/** Ends a flow: its report and its video, then what fails. */
export async function endFeel({ app, feel, flow, stopFilming }: FeelRun): Promise<void> {
  const { failing, stale } = await feel.finish();
  await stopFilming();
  await app.close();
  await encode(join(FEEL_FOLDER, `${flow}-frames`), join(FEEL_FOLDER, `${flow}.mp4`));
  expect(
    failing,
    "Fix what the feel check found, or file it and list it in KNOWN (e2e/feel.ts)",
  ).toEqual([]);
  expect(stale, "These no longer happen: take them out of KNOWN (e2e/feel.ts)").toEqual([]);
}
