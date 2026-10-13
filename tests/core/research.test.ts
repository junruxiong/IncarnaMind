/**
 * Research (#130): the composer's model plans a topic's steps, and the
 * Research Task does each as an Answer kept nowhere, searching the Library
 * within the composer's Search scope and the web, saving the pages it reads,
 * each step and each search and page read journalled. A local server stands
 * in for the web (../helpers/web).
 */
import { readFileSync } from "node:fs";
import type { MockLanguageModelV4 } from "ai/test";
import { describe, expect, test, vi } from "vitest";
import type { Core, ResearchPlan, StartResearchInput, Task } from "../../src/core";
import { shownPassages } from "../helpers/citations";
import { createTempDataFolder, queryDatabase, startCore } from "../helpers/core";
import { addAndProcess, createSourceFolder, writeSourceFile } from "../helpers/documents";
import { type ModelCall, type ScriptedReply, scriptedModel } from "../helpers/models";
import { startWeb, webSearchingModel } from "../helpers/web";

const fixture = (name: string) =>
  readFileSync(new URL(`../fixtures/web/${name}`, import.meta.url), "utf8");

const PORTUGUESE =
  "Tea was first introduced to Portuguese priests and merchants in China during the 16th century";
const MARGIN = "Harrow Foods' gross margin fell to 4.1% in the half year.";
const TEA = "http://encyclopedia.example/tea";
const MEMBERS = "http://news.example/members";
const FOUND = [
  { url: TEA, title: "Tea - The Open Encyclopedia" },
  { url: MEMBERS, title: "Tea's history, for members" },
];

const PLAN: ResearchPlan = {
  title: "How did tea reach Europe?",
  steps: ["Find who first brought tea to Europe", "Compare what the sources say"],
};

const waitFor = <T>(check: () => T | Promise<T>) => vi.waitFor(check, { timeout: 15_000 });

/** The step a request does: the last paragraph of its last user message. */
function stepOf(call: ModelCall): string {
  const last = call.options.prompt.filter((message) => message.role === "user").at(-1);
  const text =
    !last || typeof last.content === "string"
      ? ""
      : last.content.map((part) => (part.type === "text" ? part.text : "")).join("");
  return text.split("\n\n").at(-1) ?? "";
}

const requestText = (call: ModelCall) =>
  call.options.prompt
    .map((message) =>
      typeof message.content === "string"
        ? message.content
        : message.content.map((part) => ("text" in part ? part.text : "")).join(""),
    )
    .join("\n");

/**
 * A model that, for each step, searches the Library and the web as offered,
 * reads the pages found, cites the first Passage it was given, then writes
 * its notes. `hold` keeps a step's notes from finishing until it resolves.
 */
function researcher(seen: ModelCall[], hold?: { step: string; until: Promise<void> }) {
  return scriptedModel(
    (call): ScriptedReply => {
      seen.push(call);
      const step = stepOf(call);
      const done = (tool: string) => call.results.some((result) => result.tool === tool);
      if (call.tools.includes("search_documents") && !done("search_documents")) {
        return { calls: [{ tool: "search_documents", input: { query: "gross margin" } }] };
      }
      if (call.tools.includes("search_web") && !done("search_web")) {
        return { calls: [{ tool: "search_web", input: { query: "tea reached Europe" } }] };
      }
      if (call.tools.includes("read_web_page") && !done("read_web_page")) {
        return {
          calls: [TEA, MEMBERS].map((url) => ({
            tool: "read_web_page",
            input: { url, query: "Portuguese merchants" },
          })),
        };
      }
      if (!done("cite")) {
        const passage = call.results.flatMap((result) => shownPassages(result.text))[0];
        const quote = passage?.text.includes(PORTUGUESE) ? PORTUGUESE : MARGIN;
        return {
          calls: [
            { tool: "cite", input: { citations: [{ marker: 1, passage: passage?.id, quote }] } },
          ],
        };
      }
      const notes = `Notes on “${step}”: the sources say so [^1].`;
      return hold?.step === step
        ? { text: notes, pause: { after: 4, until: hold.until } }
        : { text: notes };
    },
    {
      generate: (call) =>
        /Change it as the User asks/.test(call.prompt)
          ? {
              text: "TITLE: How tea came to Europe\n1. Find the first traders\n2. **Check the dates**",
            }
          : call.prompt.includes("Topic:") && !call.prompt.includes("Plain words")
            ? {
                text: "TITLE: How did tea reach Europe?\n1. Find who first brought tea\n2) Compare the sources\n\nThat's the plan.",
              }
            : { text: "Sure, I can help." },
    },
  );
}

async function setUp(answering: MockLanguageModelV4) {
  const { model, searches } = webSearchingModel(answering, FOUND);
  const web = await startWeb({
    "/tea": () => ({ status: 200, body: fixture("article.html") }),
    "/members": () => ({ status: 200, body: fixture("login.html") }),
  });
  const dataDir = await createTempDataFolder();
  const core = startCore(dataDir, { createChatModel: () => model, ...web.adapters });
  const provider = await core.saveChatProvider({
    kind: "anthropic",
    apiKey: "sk-ant-test",
    modelId: "claude-haiku-5-5",
  });
  if (!provider.service) throw new Error("Anthropic has a service.");
  await core.allowDataFlow("chat", provider.service.id);
  core.on("consent.requested", (request) => {
    void core.respondToConsent(request.requestId, true);
  });
  await core.updateSettings({ user: { webSearch: true } });
  const folder = await core.createLibraryGroup({ name: "Grocers", description: "" });
  const sources = await createSourceFolder();
  const [report] = await addAndProcess(core, [
    await writeSourceFile(
      sources,
      "Harrow Foods.txt",
      `Harrow Foods interim results.\n${MARGIN}\n`,
    ),
  ]);
  if (!report) throw new Error("The report was added.");
  await core.moveToFolder({ documentIds: [report.id], folderId: folder.id });
  const mind = await core.createMind({ title: "Credit memo", folderId: folder.id });
  return { core, dataDir, web, searches, provider, folder, mind, report };
}

const startInput = (mindId: string, folderId: string, patch: Partial<StartResearchInput> = {}) => ({
  mindId,
  topic: "Research how tea reached Europe",
  plan: PLAN,
  sources: { web: true, library: true },
  writeInto: { mindId },
  scope: { folderIds: [folderId], tagIds: [], documentIds: [] },
  ...patch,
});

async function started(core: Core, input: StartResearchInput): Promise<Task> {
  const result = await core.startResearch(input);
  if (!result.started) throw new Error(`Research didn't start: ${result.reason}`);
  return result.task;
}

async function waitForTask(core: Core, taskId: string, done: (task: Task) => boolean) {
  let found: Task | undefined;
  await waitFor(async () => {
    found = (await core.listTasks()).find((task) => task.id === taskId);
    expect(found && done(found)).toBe(true);
  });
  return found as Task;
}

describe("Research", { timeout: 30_000 }, () => {
  test("plans a topic's steps with the composer's model, and redoes them as the User says", async () => {
    const { core, mind } = await setUp(researcher([]));

    const planned = await core.planResearch({
      mindId: mind.id,
      topic: "Research how tea reached Europe",
      sources: { web: true, library: true },
    });
    expect(planned).toEqual({
      planned: true,
      plan: {
        title: "How did tea reach Europe?",
        steps: ["Find who first brought tea", "Compare the sources"],
      },
    });

    const changed = await core.planResearch({
      mindId: mind.id,
      topic: "Research how tea reached Europe",
      sources: { web: true, library: true },
      change: { steps: ["Find who first brought tea"], text: "Check the dates too" },
    });
    expect(changed).toEqual({
      planned: true,
      plan: {
        title: "How tea came to Europe",
        steps: ["Find the first traders", "Check the dates"],
      },
    });
  });

  test("a reply with no steps is a plan that failed, and with no model nothing is sent", async () => {
    const answering = researcher([]);
    const { core, mind } = await setUp(answering);
    const failed = await core.planResearch({
      mindId: mind.id,
      topic: "Plain words",
      sources: { web: false, library: true },
    });
    expect(failed).toMatchObject({ planned: false, reason: "failed" });

    const bare = startCore(await createTempDataFolder());
    const own = await bare.createMind({ title: "Empty" });
    const notReady = await bare.planResearch({
      mindId: own.id,
      topic: "Anything",
      sources: { web: false, library: true },
    });
    expect(notReady).toMatchObject({ planned: false, reason: "not-ready" });
    expect(
      await bare.startResearch(
        startInput(own.id, "none", { scope: { folderIds: [], tagIds: [], documentIds: [] } }),
      ),
    ).toMatchObject({
      started: false,
      reason: "not-ready",
    });
    expect(await bare.listTasks()).toEqual([]);
  });

  test("each step searches the Library and the web as an Answer, saves the pages it reads, and is journalled with its searches and pages", async () => {
    const seen: ModelCall[] = [];
    const { core, dataDir, folder, mind, web } = await setUp(researcher(seen));

    const task = await started(core, startInput(mind.id, folder.id));
    expect(task).toMatchObject({
      kind: "research",
      mindId: mind.id,
      question: "Research how tea reached Europe",
      status: "running",
    });
    const done = await waitForTask(core, task.id, (each) => each.finishedAt !== null);

    expect(done).toMatchObject({ status: "done", progress: { done: 2, total: 2 } });
    // The two pages are read at once: whichever ends first comes first.
    const reads = expect.arrayContaining([
      ["Read “Tea - The Open Encyclopedia”", "done"],
      ["Read a page on news.example", "failed"],
    ]);
    const steps = (await core.listTaskSteps(task.id)).map((step) => [step.text, step.status]);
    expect(steps).toHaveLength(10);
    expect(steps.slice(0, 3)).toEqual([
      ["Find who first brought tea to Europe", "done"],
      ["Searched the Library for “gross margin”", "done"],
      ["Searched the web for “tea reached Europe”", "done"],
    ]);
    expect(steps.slice(3, 5)).toEqual(reads);
    expect(steps.slice(5, 8)).toEqual([
      ["Compare what the sources say", "done"],
      ["Searched the Library for “gross margin”", "done"],
      ["Searched the web for “tea reached Europe”", "done"],
    ]);
    expect(steps.slice(8)).toEqual(reads);
    // The page it read is a Document in the Mind's Folder; the one behind a sign-in isn't.
    const page = (await core.listDocuments()).find((each) => each.kind === "web");
    expect(page?.web?.address).toBe(TEA);
    const { assignments } = await core.getLibrary();
    expect(assignments.find((each) => each.documentId === page?.id)?.groupId).toBe(folder.id);
    expect(web.heard).toContain("encyclopedia.example/tea");
    // The second step reads what the first found, and the plan; nothing went into the Mind.
    const second = seen.find((call) => stepOf(call) === "Compare what the sources say");
    expect(second && requestText(second)).toContain("What the steps before found:");
    expect(second && requestText(second)).toContain(
      "Step 1: Notes on “Find who first brought tea to Europe”: the sources say so.",
    );
    const [row] = queryDatabase<{ result: string }>(
      dataDir,
      "SELECT result FROM run_steps WHERE run_id = ? AND key = 'step:0'",
      [task.id],
    );
    expect(JSON.parse(row?.result ?? "{}").documents).toEqual(expect.arrayContaining([page?.id]));
  });

  test("with Web off it is offered no web Tool, and with the Library off no Document of the Library", async () => {
    const seen: ModelCall[] = [];
    const { core, folder, mind, report } = await setUp(researcher(seen));

    const libraryOnly = await started(
      core,
      startInput(mind.id, folder.id, {
        sources: { web: false, library: true },
        plan: { ...PLAN, steps: ["Find margins"] },
      }),
    );
    await waitForTask(core, libraryOnly.id, (each) => each.finishedAt !== null);
    expect(seen.every((call) => !call.tools.includes("search_web"))).toBe(true);
    expect(seen.some((call) => call.results.some((result) => result.text.includes(MARGIN)))).toBe(
      true,
    );

    seen.length = 0;
    const webOnly = await started(
      core,
      startInput(mind.id, folder.id, {
        sources: { web: true, library: false },
        plan: { ...PLAN, steps: ["Find tea"] },
      }),
    );
    await waitForTask(core, webOnly.id, (each) => each.finishedAt !== null);
    expect(seen.some((call) => call.tools.includes("search_web"))).toBe(true);
    const searched = seen.flatMap((call) =>
      call.results.filter((result) => result.tool === "search_documents"),
    );
    expect(searched.every((result) => !result.text.includes(report.name))).toBe(true);
  });

  test("Stop ends it in the middle of a step; the step is left to do again and nothing more runs", async () => {
    const seen: ModelCall[] = [];
    let release = () => {};
    const until = new Promise<void>((resolve) => {
      release = resolve;
    });
    const { core, dataDir, folder, mind } = await setUp(
      researcher(seen, { step: "Find who first brought tea to Europe", until }),
    );
    const task = await started(core, startInput(mind.id, folder.id));
    await waitFor(() =>
      expect(seen.some((call) => call.results.some((result) => result.tool === "cite"))).toBe(true),
    );

    const stopped = await core.stopTask(task.id);
    release();
    expect(stopped.status).toBe("stopped");
    const rows = queryDatabase<{ key: string; status: string }>(
      dataDir,
      "SELECT key, status FROM run_steps WHERE run_id = ? AND key LIKE 'step:%' ORDER BY rowid",
      [task.id],
    );
    expect(rows).toEqual([{ key: "step:0", status: "started" }]);
    expect(seen.some((call) => stepOf(call) === "Compare what the sources say")).toBe(false);
  });
});
