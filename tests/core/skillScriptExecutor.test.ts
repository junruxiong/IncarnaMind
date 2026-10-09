/**
 * Skill scripts go through the Executor (#61): the script runner chooses the
 * interpreter, checks the arguments and writes the result text, and asks the
 * Executor to run it with what it may touch. A fake Executor records each
 * request: the Skill's folder and the working folder to read, the working
 * folder to write, and no network.
 */
import { realpath } from "node:fs/promises";
import { join } from "node:path";
import { describe, expect, test, vi } from "vitest";
import {
  type ExecRequest,
  type ExecResult,
  type Executor,
  type SandboxLevel,
  SKILL_SCRIPT_LIMITS,
} from "../../src/core";
import { WORKING_FOLDER } from "../../src/core/execution";
import { createScriptRunner, type ScriptToRun } from "../../src/core/skills/scripts";
import { answerEnded, askNew } from "../helpers/citations";
import { createTempDataFolder, startCore } from "../helpers/core";
import { connectToMind } from "../helpers/mindClient";
import { answerIn } from "../helpers/minds";
import { scriptedModel, scriptedModels } from "../helpers/models";
import { importSkill, simpleSkillMd, writeFolder } from "../helpers/skills";

const RAN: ExecResult = {
  exitCode: 0,
  timedOut: false,
  stdout: "faked\n",
  stderr: "",
  stdoutTruncated: false,
  stderrTruncated: false,
};

/**
 * An Executor at `level` that runs nothing: it records each request and
 * resolves with what `respond` gives (`RAN` by default; a promise that may
 * wait for the request's signal).
 */
function fakeExecutor(
  level: SandboxLevel = "none",
  respond: (request: ExecRequest) => Promise<ExecResult> = async () => RAN,
) {
  const requests: ExecRequest[] = [];
  const executor: Executor = {
    level,
    run: async (request) => {
      requests.push(request);
      return respond(request);
    },
  };
  return { executor, requests };
}

const SKILL_DIR = "/data/skills/toolbox-id";

/** A run of the toolbox's echo script, with what the case changes. */
const toRun = (overrides: Partial<ScriptToRun>): ScriptToRun => ({
  skillDir: SKILL_DIR,
  script: "scripts/echo.js",
  args: [],
  timeoutMs: 5_000,
  signal: new AbortController().signal,
  ...overrides,
});

describe("The script runner asks the Executor", () => {
  test("for the interpreter, the script and its arguments, with SKILL_DIR, its timeout and output caps, and what it may touch", async () => {
    // An Executor may say more than a run's card keeps: the card stores only its own fields.
    const { executor, requests } = fakeExecutor("none", async () => ({ ...RAN, pid: 4242 }));
    const scripts = createScriptRunner({
      executor,
      runtimes: { node: { command: "/app/node", env: { ELECTRON_RUN_AS_NODE: "1" } } },
    });
    const signal = new AbortController().signal;

    const run = await scripts.run(
      toRun({ args: ["two words", "--flag"], timeoutMs: 7_000, signal }),
    );

    expect(run).toStrictEqual({ ...RAN, error: null });
    expect(requests).toEqual([
      {
        command: "/app/node",
        args: [join(SKILL_DIR, "scripts", "echo.js"), "two words", "--flag"],
        env: { ELECTRON_RUN_AS_NODE: "1", SKILL_DIR },
        allow: { read: [SKILL_DIR, WORKING_FOLDER], write: [WORKING_FOLDER], network: "none" },
        timeoutMs: 7_000,
        maxOutputBytes: SKILL_SCRIPT_LIMITS.maxOutputBytes,
        signal: expect.any(AbortSignal),
      },
    ]);
    // No working folder of its own: the Executor makes a new one.
    expect(requests[0]).not.toHaveProperty("cwd");
  });

  test("each kind of script with its interpreter, and the same allow sets", async () => {
    const { executor, requests } = fakeExecutor();
    const scripts = createScriptRunner({ executor, runtimes: { platform: "linux" } });

    await scripts.run(toRun({ script: "scripts/hello.py" }));
    await scripts.run(toRun({ script: "scripts/hello.sh" }));

    expect(requests.map(({ command, env }) => ({ command, env }))).toEqual([
      {
        command: "python3",
        env: {
          PYTHONDONTWRITEBYTECODE: "1",
          PYTHONIOENCODING: "utf-8",
          PYTHONUTF8: "1",
          SKILL_DIR,
        },
      },
      { command: "bash", env: { SKILL_DIR } },
    ]);
    for (const request of requests) {
      expect(request.allow).toEqual({
        read: [SKILL_DIR, WORKING_FOLDER],
        write: [WORKING_FOLDER],
        network: "none",
      });
    }
  });

  test("a script that can't run here isn't sent to the Executor", async () => {
    const { executor, requests } = fakeExecutor();
    const scripts = createScriptRunner({ executor, runtimes: { platform: "win32" } });

    await expect(scripts.run(toRun({ script: "scripts/hello.sh" }))).rejects.toThrow(
      "scripts/hello.sh is a shell script, and shell scripts can't run on Windows.",
    );
    await expect(scripts.run(toRun({ script: "scripts/convert.ts" }))).rejects.toThrow(
      /TypeScript/,
    );
    expect(requests).toEqual([]);
  });

  test("a command the Executor can't find becomes a plain error naming what to install", async () => {
    const { executor } = fakeExecutor("none", async () => {
      throw Object.assign(new Error("spawn python3 ENOENT"), { code: "ENOENT" });
    });
    const scripts = createScriptRunner({ executor, runtimes: { platform: "darwin" } });

    await expect(scripts.run(toRun({ script: "scripts/hello.py" }))).rejects.toThrow(
      "python3 not found: install Python 3 to run scripts/hello.py.",
    );
  });

  test("stopping every script, or closing, stops what the Executor runs; closed, nothing more is sent", async () => {
    const { executor, requests } = fakeExecutor(
      "none",
      (request) =>
        new Promise((resolve) =>
          request.signal.addEventListener("abort", () => resolve({ ...RAN, exitCode: null }), {
            once: true,
          }),
        ),
    );
    const scripts = createScriptRunner({ executor });

    const first = scripts.run(toRun({}));
    await vi.waitFor(() => expect(requests).toHaveLength(1));
    scripts.stopAll();
    expect(await first).toMatchObject({ exitCode: null });

    const second = scripts.run(toRun({}));
    await vi.waitFor(() => expect(requests).toHaveLength(2));
    scripts.close();
    expect(await second).toMatchObject({ exitCode: null });

    await expect(scripts.run(toRun({}))).rejects.toThrow(
      "IncarnaMind is closing, so the script didn't run.",
    );
    expect(requests).toHaveLength(2);
  });

  test("a Stop of the Answer reaches the Executor", async () => {
    const { executor, requests } = fakeExecutor(
      "none",
      (request) =>
        new Promise((resolve) =>
          request.signal.addEventListener("abort", () => resolve({ ...RAN, exitCode: null }), {
            once: true,
          }),
        ),
    );
    const scripts = createScriptRunner({ executor });
    const answer = new AbortController();

    const running = scripts.run(toRun({ signal: answer.signal }));
    await vi.waitFor(() => expect(requests).toHaveLength(1));
    expect(requests[0]?.signal.aborted).toBe(false);
    answer.abort();

    expect(await running).toMatchObject({ exitCode: null });
    expect(requests[0]?.signal.aborted).toBe(true);
  });

  test("run_skill_script's declared access comes from the Executor's level", () => {
    const unconfined = createScriptRunner({ executor: fakeExecutor("none").executor });
    expect(unconfined.access(SKILL_DIR)).toEqual({
      confined: false,
      read: "anywhere",
      write: "anywhere",
      network: "any",
    });

    const sandboxed = createScriptRunner({ executor: fakeExecutor("os").executor });
    expect(sandboxed.access(SKILL_DIR)).toEqual({
      confined: true,
      read: [SKILL_DIR, WORKING_FOLDER],
      write: [WORKING_FOLDER],
      network: "none",
    });
  });
});

describe("The core runs Skill scripts through its Executor", { timeout: 30_000 }, () => {
  test("an Executor in the core's adapters runs the scripts Answers run, and the model reads what it returns", async () => {
    const dataDir = await createTempDataFolder();
    const sources = await createTempDataFolder();
    const { executor, requests } = fakeExecutor();
    const results: string[] = [];
    const model = scriptedModel((call) => {
      const result = call.results.find((each) => each.tool === "run_skill_script");
      if (result) {
        results.push(result.text);
        return { text: "Done." };
      }
      return {
        calls: [
          {
            tool: "run_skill_script",
            input: { skill: "toolbox", script: "scripts/echo.js", args: ["one"] },
          },
        ],
      };
    });
    const core = startCore(dataDir, {
      createChatModel: scriptedModels(model).createChatModel,
      executor,
    });
    await core.saveChatProvider({ kind: "ollama", modelId: "local-model" });
    const skill = await importSkill(
      core,
      await writeFolder(sources, "toolbox", {
        "SKILL.md": simpleSkillMd("toolbox", "Small scripts for tests. Use for running scripts."),
        "scripts/echo.js": 'console.log("real");\n',
      }),
    );
    await core.setApprovalPolicy({
      subject: { kind: "skill-script", skillId: skill.id },
      policy: "always",
      riskAccepted: true,
    });
    const mind = await core.createMind({ title: "Scripts" });
    const client = await connectToMind(core, mind.id);

    const answerId = await askNew(core, client, mind.id, "Run the echo script.");
    expect((await answerEnded(core, answerId)).payload).toMatchObject({ status: "done" });

    const skillDir = await realpath(join(dataDir, "skills", skill.id));
    expect(requests).toHaveLength(1);
    expect(requests[0]).toMatchObject({
      args: [join(skillDir, "scripts", "echo.js"), "one"],
      env: { SKILL_DIR: skillDir },
      allow: { read: [skillDir, WORKING_FOLDER], write: [WORKING_FOLDER], network: "none" },
      timeoutMs: SKILL_SCRIPT_LIMITS.defaultTimeoutSeconds * 1000,
    });
    expect(results).toEqual([
      [
        "scripts/echo.js exited with code 0.",
        "<stdout>",
        "faked",
        "</stdout>",
        "<stderr>",
        "</stderr>",
      ].join("\n"),
    ]);
    const calls = JSON.parse(String(answerIn(client, answerId).attrs.toolCalls ?? "[]"));
    expect(calls).toMatchObject([{ status: "done", script: { ...RAN, error: null } }]);
  });
});
