/**
 * Taint, recorded only (#64; docs/designs/agent-extensibility.md §4.6): a Run
 * records when it has read untrusted content. Tools declare whether their
 * results are untrusted; the Answer's Tool-calling loop marks its Run tainted
 * after the first call of one that returned some, above the Run engine, and
 * tells the gate of each call after it; the gate hands it to approvals, and
 * the approval request carries it. What asks and what runs doesn't change
 * (see approvalDecision.test.ts).
 */
import { describe, expect, test } from "vitest";
import type { AnswerToolCall, ApprovalRequest, Core } from "../../src/core";
import {
  type AnswerTools,
  createAiSdkAnswerEngine,
  documentTools,
} from "../../src/core/answers/engine";
import { connectorToolEffects } from "../../src/core/connectors";
import { declaredAccess } from "../../src/core/execution";
import { skillTools } from "../../src/core/skills/tools";
import type { Tool } from "../../src/core/tools";
import { answerEnded, askNew, setUpWithDocuments } from "../helpers/citations";
import { logFileIn, testProcesses, tideServer, waitForState } from "../helpers/connectors";
import { createTempDataFolder, nextEvent, startCore } from "../helpers/core";
import { connectToMind, type MindClient } from "../helpers/mindClient";
import { answerIn } from "../helpers/minds";
import { type ModelCall, scriptedModel, scriptedModels } from "../helpers/models";

const QUESTION = "Book me a boat from Dover at high water.";
const BOOK = "tides__book_boat";
const LOOKUP = "tides__lookup_tide";

/** A Connector's Tool that may change something, as the Connectors hand it out. */
const book: Tool = {
  name: BOOK,
  description: 'From the User\'s Connector "Tides". Book a boat trip.',
  inputSchema: {
    type: "object",
    properties: { place: { type: "string", description: "Where from." } },
    required: ["place"],
  },
  provider: { kind: "connector", id: "connector-1", name: "Tides" },
  providerTool: "book_boat",
  effects: () => connectorToolEffects({ id: "connector:connector-1", name: "Tides" }, false),
  untrustedResult: true,
  call: async () => "Booked a boat trip from Dover.",
};

/** The Skills' Tools, for one Skill the User installed: "tide-tables", which has a script. */
const skills = (): Tool[] =>
  skillTools({
    loadable: true,
    skill: (name) =>
      name === "tide-tables" ? { skillId: "skill-1", skillDir: "/skills/1" } : null,
    useSkill: async (name) => `The Skill ${name}: read the tide tables.`,
    readSkillFile: async (skill, path) => `${skill}/${path}`,
    scripts: {
      access: () => declaredAccess("none", { read: [], write: [], network: "none" }),
      run: async () => "Exit code 0.",
    },
  });

/** Documents that hold `documentCount` Documents, whose every search finds `passageCount` Passages. */
const documentsFinding = (passageCount: number, documentCount = 1): AnswerTools => ({
  documentCount,
  searchDocuments: async () =>
    passageCount > 0
      ? {
          text: '<passage id="P1" document="Tides">\nHigh water at Dover is at six.\n</passage>',
          passageCount,
        }
      : { text: "No Passages in the User's Documents match this search.", passageCount: 0 },
  cite: () => "Recorded.",
});

/**
 * Has the Answer engine answer a model that makes `calls`, one per model
 * call, then writes. Returns what the gate was told of each call: its Tool,
 * and whether the Run had read untrusted content before it.
 */
async function gated(
  documents: AnswerTools,
  calls: { tool: string; input: Record<string, unknown> }[],
  tools: Tool[] = [book],
) {
  const model = scriptedModel((call: ModelCall) => {
    const next = calls[call.index];
    return next ? { calls: [next] } : { text: "Booked." };
  });
  const seen: { tool: string; tainted: boolean }[] = [];
  const events: string[] = [];
  for await (const event of createAiSdkAnswerEngine().generate({
    instructions: () => "Answer from the Documents.",
    messages: [{ role: "user", content: QUESTION }],
    question: QUESTION,
    model,
    documents,
    tools,
    gate: async (tool, { tainted }) => {
      seen.push({ tool: tool.name, tainted });
      return { run: true };
    },
    support: "tools",
    signal: new AbortController().signal,
  })) {
    events.push(event.type);
  }
  expect(events.at(-1)).toBe("finished");
  return seen;
}

const SEARCH = { tool: "search_documents", input: { query: "high water Dover" } };
const BOOKING = { tool: BOOK, input: { place: "Dover" } };

describe("Tools declare whether their results are untrusted", () => {
  test("the Documents' Passages and what a script printed are; cite's reply and the User's own Skills aren't", () => {
    const { search, cite } = documentTools(documentsFinding(1), {
      fit: (found) => found,
      searched: () => undefined,
      cited: () => undefined,
    });

    expect(
      Object.fromEntries(
        [search, cite, ...skills()].map((each) => [each.name, each.untrustedResult]),
      ),
    ).toEqual({
      search_documents: true,
      cite: false,
      use_skill: false,
      read_skill_file: false,
      run_skill_script: true,
    });
  });
});

describe("A Run is tainted once it has read untrusted content", () => {
  test("after Document search returns Passages, the next gated call sees taint; the search itself didn't", async () => {
    expect(await gated(documentsFinding(2), [SEARCH, BOOKING])).toEqual([
      { tool: "search_documents", tainted: false },
      { tool: BOOK, tainted: true },
    ]);
  });

  test("a search that found no Passages leaves it clean", async () => {
    expect(await gated(documentsFinding(0), [SEARCH, BOOKING])).toEqual([
      { tool: "search_documents", tainted: false },
      { tool: BOOK, tainted: false },
    ]);
  });

  test("with no Documents it stays clean: reading the User's own Skill doesn't taint it", async () => {
    const calls = [{ tool: "use_skill", input: { name: "tide-tables" } }, BOOKING];

    expect(await gated(documentsFinding(0, 0), calls, [...skills(), book])).toEqual([
      { tool: "use_skill", tainted: false },
      { tool: BOOK, tainted: false },
    ]);
  });
});

/** Accepts every consent request: these tests are about approvals, not consent. */
function acceptConsent(core: Core) {
  core.on("consent.requested", (request) => {
    void core.respondToConsent(request.requestId, true);
  });
}

const toolCallsOf = (client: MindClient, answerId: string): AnswerToolCall[] =>
  JSON.parse(String(answerIn(client, answerId).attrs.toolCalls ?? "[]")) as AnswerToolCall[];

/**
 * Asks the Question, waits for the call that asks first, denies it, and waits
 * for the Answer to finish. Returns the request and the Answer's id.
 */
async function denyTheRequest(core: Core, client: MindClient, mindId: string) {
  const requested = nextEvent(core, "approval.requested");
  const answerId = await askNew(core, client, mindId, QUESTION);
  const request: ApprovalRequest = await requested;
  const ended = answerEnded(core, answerId);
  await core.respondToApproval(request.requestId, "deny");
  expect((await ended).event).toBe("finished");
  return { request, answerId };
}

describe("Approval requests carry the taint", { timeout: 30_000 }, () => {
  test("a Connector's Tool called after a search of the User's Documents found Passages: its request says so, and it asks as before", async () => {
    const model = scriptedModel((call: ModelCall) =>
      call.index === 0
        ? { calls: [{ tool: "search_documents", input: { query: "high water" } }] }
        : call.index === 1
          ? { calls: [BOOKING] }
          : { text: "I couldn't book it." },
    );
    const { core, mind, client } = await setUpWithDocuments(
      model,
      [{ name: "Tides.txt", contents: "High water at Dover comes twice a day." }],
      { processes: testProcesses() },
    );
    const connector = await core.addConnector(tideServer(logFileIn(await createTempDataFolder())));
    await waitForState(core, connector.id, "ready");
    acceptConsent(core);

    const { request, answerId } = await denyTheRequest(core, client, mind.id);

    const [search] = toolCallsOf(client, answerId);
    expect(search?.tool).toBe("search_documents");
    expect(search?.resultCount).toBeGreaterThan(0);
    expect(request).toMatchObject({ tool: "book_boat", readOnly: false, tainted: true });
  });

  test("a Connector's reply taints it too: a call after it says so", async () => {
    const model = scriptedModel((call: ModelCall) =>
      call.index === 0
        ? { calls: [{ tool: LOOKUP, input: { place: "Dover" } }] }
        : call.index === 1
          ? { calls: [BOOKING] }
          : { text: "I couldn't book it." },
    );
    const dataDir = await createTempDataFolder();
    const core = startCore(dataDir, {
      createChatModel: scriptedModels(model).createChatModel,
      processes: testProcesses(),
    });
    await core.saveChatProvider({ kind: "ollama", modelId: "local-model" });
    const connector = await core.addConnector(tideServer(logFileIn(dataDir)));
    await waitForState(core, connector.id, "ready");
    acceptConsent(core);
    const mind = await core.createMind({ title: "Boats" });
    const client = await connectToMind(core, mind.id);

    const { request, answerId } = await denyTheRequest(core, client, mind.id);

    // The lookup only reads, so it ran without asking; its reply came from the Connector's service.
    expect(toolCallsOf(client, answerId)[0]).toMatchObject({ tool: "lookup_tide", status: "done" });
    expect(request).toMatchObject({ tool: "book_boat", tainted: true });
  });
});
