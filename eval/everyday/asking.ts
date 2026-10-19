/**
 * Asking an everyday request through the core, as the assistant does, and
 * holding its Answer to the guards in ./checks: what `npm run eval:everyday`
 * runs for each request, and what other evaluations that ask the assistant
 * the same way reuse. Also the Documents the requests are asked over.
 */
import { createHash } from "node:crypto";
import { existsSync } from "node:fs";
import { mkdir, readdir, readFile, rename, writeFile } from "node:fs/promises";
import { join, resolve } from "node:path";
import { vi } from "vitest";
import * as Y from "yjs";
import {
  type AnswerUsage,
  BLOCK_ID_ATTRIBUTE,
  type ChatModelChoice,
  CITATION_NODE,
  type ConversationAnswer,
  type Core,
  MIND_CONTENT_FIELD,
  type MindFocus,
  messageReference,
} from "../../src/core";
import { forPage } from "../../src/core/answers/pointersBack";
import { toolRecordLines } from "../../src/core/answers/toolRecord";
import { plainText } from "../../src/core/mindText";
import type { createAiSdkChatModel } from "../../src/core/providers/models";
import { pageBlocks } from "../../src/shared/answerBlocks";
import { namesOneOf, sourcesAsText } from "../../src/shared/namedSources";
import { resolveProposals } from "../../src/shared/proposals";
import { connectToMind } from "../../tests/helpers/mindClient";
import { heading, note, writeMind } from "../../tests/helpers/minds";
import { paragraph, wordFile } from "../../tests/helpers/word";
import { importWordBytes } from "../../tests/helpers/wordMinds";
import type { ChatSettings, readConfig } from "../lib/config";
import type { EvalDocument } from "../lib/evaluationSet";
import type { Library } from "../lib/library";
import type { Log } from "../lib/log";
import type { EverydayCase } from "./cases";
import {
  pointersIn,
  type RequestFailure,
  type RequestOutcome,
  requestFailures,
  rewrittenLater,
  takenOut,
} from "./checks";
import { DATED_TEXTS } from "./dated";
import { citedClaims } from "./dates";
import { judgeClaims } from "./judge";
import type { Asks, ReaskWatch } from "./reasks";
import { checkCitedSentences, type SentenceSupport } from "./support";

const ROOT = resolve(__dirname, "../..");

/** A request as `runCase` asks it: which Documents it is asked over is the caller's. */
export type AskedRequest = Omit<EverydayCase, "library">;

/** The two arXiv papers of the owner's own conversation, fetched once into the cache and checked. */
export const PAPERS = [
  {
    key: "graphst",
    url: "https://arxiv.org/pdf/2306.10683v1",
    file: "2306.10683v1.pdf",
    sha256: "295ffa4d0840a1956ea4094473978b01d4e2fb00299eb7fde9e45a300d4958db",
  },
  {
    key: "tutorial",
    url: "https://arxiv.org/pdf/1609.04846v1",
    file: "A Tutorial about Random Neural Networks in Supervised Learning.pdf",
    sha256: "41799311b54eb75bee7011f6d6210cd14a8e3517bf79686dd9acd40569598cfa",
  },
];

const sha256 = (bytes: Uint8Array) => createHash("sha256").update(bytes).digest("hex");

/** A request whose Answer takes longer is stopped and counted as failed. */
export const ANSWER_TIMEOUT_MS = 5 * 60_000;

export async function fetchPapers(cacheDir: string, log: Log): Promise<EvalDocument[]> {
  const folder = join(cacheDir, "everyday");
  await mkdir(folder, { recursive: true });
  const documents: EvalDocument[] = [];
  for (const paper of PAPERS) {
    const path = join(folder, paper.file);
    if (!existsSync(path) || sha256(await readFile(path)) !== paper.sha256) {
      log(`Fetching ${paper.url}`);
      // arXiv asks for one request every three seconds.
      await new Promise((done) => setTimeout(done, 3_000));
      const response = await fetch(paper.url, {
        headers: { "User-Agent": "IncarnaMind document evaluation" },
      });
      if (!response.ok) throw new Error(`${paper.url} answered HTTP ${response.status}.`);
      const bytes = new Uint8Array(await response.arrayBuffer());
      if (sha256(bytes) !== paper.sha256) throw new Error(`${paper.url} isn't the file expected.`);
      await writeFile(`${path}.partial`, bytes);
      await rename(`${path}.partial`, path);
    }
    documents.push({ key: paper.key, path });
  }
  return documents;
}

export async function reports(): Promise<EvalDocument[]> {
  const folder = join(ROOT, "data");
  return (await readdir(folder))
    .filter((name) => name.endsWith(".pdf"))
    .map((name) => ({ key: name, path: join(folder, name) }));
}

export interface CaseResult {
  id: string;
  language: "en" | "zh-CN";
  request: string;
  model: string;
  failures: RequestFailure[];
  reply: string;
  page: string;
  /** What the guard took out: sentences of the reply, by Add to note, and Blocks of what the calls gave to write into Minds. */
  removed: string[];
  /** Sentences of what the calls gave that the model's edit_mind or the app's own check rewrote after the guard, and what the Mind holds now. */
  rewritten: { given: string; now: string | null }[];
  pointersLeft: number;
  calls: string[];
  /** The Mind's Blocks that changed, as they read before and once its changes are accepted. */
  changes: { block: number; before: string; after: string }[];
  seconds: number;
  /** From the start to the last Answer's end, before it is checked. */
  answeredSeconds: number;
  error: string | null;
  /** The Answers' tokens, its earlier requests' included (not Edit with AI's or the judge's). */
  usage: AnswerUsage;
  /** The markers the app took out of its Answers because no Citation backed them. */
  droppedMarkers: number;
  /** How often the app asked the model inside this Answer: again, and once for Citations. */
  asks: Asks;
  /** A research request: each cited sentence of the reply, as the app's own check judged it; null for the others. */
  support: SentenceSupport[] | null;
  /** A summary: its sentences outside headings that make a claim, and how many carry a Citation; null for the others. */
  claims: { claims: number; cited: number } | null;
}

/** Asks in the assistant and resolves with the Answer as stored once it has ended. */
export async function ask(
  core: Core,
  input: {
    text: string;
    mindId: string | null;
    conversationId: string | null;
    focus: MindFocus | null;
  },
): Promise<ConversationAnswer> {
  const asked = await core.askInConversation(input);
  if (!asked.asked) throw new Error(`Not asked: ${JSON.stringify(asked)}`);
  const ended = await new Promise<boolean>((done) => {
    const timer = setTimeout(() => end(false), ANSWER_TIMEOUT_MS);
    const stops = [
      core.on("answer.finished", ({ answerId }) => answerId === asked.answerId && end(true)),
      core.on("answer.failed", ({ answerId }) => answerId === asked.answerId && end(true)),
    ];
    function end(inTime: boolean) {
      clearTimeout(timer);
      for (const stop of stops) stop();
      done(inTime);
    }
  });
  if (!ended) {
    await core.stopConversationAnswer({
      conversationId: asked.conversation.id,
      answerId: asked.answerId,
    });
    throw new Error(`No Answer within ${ANSWER_TIMEOUT_MS / 60_000} minutes.`);
  }
  const { messages } = await core.getConversation(asked.conversation.id);
  const answer = messages.find((message) => message.id === asked.answerId);
  if (answer?.kind !== "answer") throw new Error("No Answer was kept.");
  return answer;
}

async function askExample(core: Core, language: "en" | "zh-CN") {
  await core.updateSettings({ user: { language } });
  const { mindId } = await core.createExamples();
  const asked = await vi.waitFor(() => core.askReadyQuestion(), {
    timeout: 2 * 60_000,
    interval: 500,
  });
  return { conversationId: asked.conversationId, mindId };
}

async function importWord(library: Library, word: NonNullable<EverydayCase["word"]>) {
  const [title = "", ...body] = word.paragraphs;
  const folder = join(library.dataDir, "word");
  await mkdir(folder, { recursive: true });
  return importWordBytes(
    library.core,
    folder,
    wordFile({
      body: [paragraph(title, { style: "Title" }), ...body.map((each) => paragraph(each))].join(""),
    }),
    { name: word.name, origin: "own", addOriginal: true },
  );
}

/** The top-level Blocks, by id, as they read once these requests' changes are accepted. */
function acceptedTexts(doc: Y.Doc, requests: ReadonlySet<string>): Map<string, string> {
  const copy = new Y.Doc();
  Y.applyUpdate(copy, Y.encodeStateAsUpdate(doc));
  const blocks = copy.getXmlFragment(MIND_CONTENT_FIELD);
  copy.transact(() => resolveProposals(blocks, (ref) => requests.has(ref.request), "accept"));
  return new Map(
    blocks
      .toArray()
      .flatMap((block) =>
        block instanceof Y.XmlElement
          ? [[String(block.getAttribute(BLOCK_ID_ATTRIBUTE)), plainText(block)]]
          : [],
      ),
  );
}

function citedDocuments(doc: Y.Doc): string[] {
  const found: string[] = [];
  const visit = (element: Y.XmlElement | Y.XmlFragment) => {
    for (const child of element.toArray()) {
      if (!(child instanceof Y.XmlElement)) continue;
      const id = child.getAttribute("documentId");
      if (child.nodeName === CITATION_NODE && typeof id === "string") found.push(id);
      visit(child);
    }
  };
  visit(doc.getXmlFragment(MIND_CONTENT_FIELD));
  return found;
}

export async function runCase(
  library: Library,
  scoped: { id: string; name: string }[],
  judge: ReturnType<typeof createAiSdkChatModel>,
  model: string,
  each: AskedRequest,
  {
    watch,
    checkModel = null,
  }: {
    /** Watches the app's own asks inside the Answer; none watched without it. */
    watch?: ReaskWatch;
    /** The model the cited sentences are checked on; null: the default chat model. */
    checkModel?: ChatModelChoice | null;
  } = {},
): Promise<CaseResult> {
  const { core } = library;
  const started = Date.now();
  const usage: AnswerUsage = {
    inputTokens: 0,
    outputTokens: 0,
    cachedInputTokens: 0,
    cacheWriteTokens: 0,
  };
  let droppedMarkers = 0;
  const counting = core.on("answer.finished", (event) => {
    droppedMarkers += event.droppedMarkers;
    usage.inputTokens += event.usage.inputTokens;
    usage.outputTokens += event.usage.outputTokens;
    usage.cachedInputTokens += event.usage.cachedInputTokens;
    usage.cacheWriteTokens += event.usage.cacheWriteTokens;
  });
  const example = each.afterExample ? await askExample(core, each.language) : null;
  const imported = each.word ? await importWord(library, each.word) : null;
  const mindId =
    imported?.mindId ??
    (each.inExample ? example?.mindId : null) ??
    (each.mind ? (await core.createMind({ title: each.mind.title })).id : null);
  const mind = mindId ? { id: mindId } : null;
  const client = imported?.client ?? (mind ? await connectToMind(core, mind.id) : null);
  if (client && each.mind) {
    writeMind(
      client,
      each.mind.blocks.map((line) =>
        line.startsWith("## ") ? heading(2, line.slice(3)) : note(line),
      ),
    );
    await client.settled();
  }
  const idsNow = () =>
    client
      ? client.blocks
          .toArray()
          .map((block) =>
            block instanceof Y.XmlElement ? String(block.getAttribute(BLOCK_ID_ATTRIBUTE)) : "",
          )
      : [];
  let ids = idsNow();
  let before = client ? acceptedTexts(client.doc, new Set()) : new Map<string, string>();
  const focus: MindFocus | null =
    each.selectAll && client
      ? { selection: [...before.values()].join("\n\n"), blockIds: ids }
      : null;
  let conversationId: string | null = example?.conversationId ?? null;
  const earlierCalls: ConversationAnswer["toolCalls"] = [];
  let last: ConversationAnswer | null = null;
  for (const earlier of each.before ?? []) {
    const fromExample = each.beforeInExample === true;
    const answer = await ask(core, {
      text: earlier,
      mindId: (fromExample ? example?.mindId : mind?.id) ?? null,
      conversationId,
      focus: fromExample ? null : focus,
    });
    conversationId = answer.conversationId;
    earlierCalls.push(...answer.toolCalls);
    last = answer;
  }
  let addedBlocks: number[] | undefined;
  if (each.addToNote && last && mind && client) {
    const earlier = new Set(ids);
    await core.addToNote({
      mindId: mind.id,
      blocks: pageBlocks(last, each.before?.at(-1) ?? ""),
      origin: {
        from: messageReference(last.conversationId, last.id),
        question: each.before?.at(-1) ?? "",
      },
      model: last.model,
    });
    await client.settled();
    ids = idsNow();
    addedBlocks = ids.flatMap((id, block) => (earlier.has(id) ? [] : [block]));
    before = acceptedTexts(client.doc, new Set());
  }
  watch?.begin(each.firstReply ?? null);
  const answer = await ask(core, {
    text: each.request,
    mindId: mind?.id ?? null,
    conversationId,
    focus,
  });
  const answeredSeconds = (Date.now() - started) / 1000;
  const asks = watch ? { ...watch.end() } : { again: 0, citations: 0 };
  counting();
  const minds = await core.listMinds();
  const names = (id: string) => minds.find((each) => each.id === id)?.title ?? null;
  const record = toolRecordLines(answer.toolCalls, names);
  const support = each.research ? await checkCitedSentences(core, answer, checkModel) : null;
  const verdict = await judgeClaims(
    judge,
    answer.text,
    record,
    toolRecordLines(earlierCalls, names),
  );
  const changed = new Set(
    answer.toolCalls.flatMap((call) => (call.mindChange ? [call.mindChange.mindId] : [])),
  );
  const requests = new Set(
    answer.toolCalls.flatMap((call) => (call.mindChange ? [call.mindChange.request] : [])),
  );
  const citedInMinds: string[] = [];
  const sourcesAsTextIn: string[] = [];
  const writtenInMinds: string[] = [];
  const documentNames = scoped.map((document) => document.name);
  for (const mindId of changed) {
    const written = await connectToMind(core, mindId);
    await written.settled();
    citedInMinds.push(...citedDocuments(written.doc));
    const was = mindId === mind?.id ? before : new Map<string, string>();
    for (const [id, text] of acceptedTexts(written.doc, requests)) {
      sourcesAsTextIn.push(
        ...sourcesAsText(text, (source) => namesOneOf(source.name, documentNames)),
      );
      if (was.get(id) !== text) writtenInMinds.push(text);
    }
  }
  let changes: CaseResult["changes"] = [];
  if (client) {
    await client.settled();
    const after = acceptedTexts(client.doc, requests);
    changes = ids.flatMap((id, block) =>
      after.get(id) === before.get(id)
        ? []
        : [{ block, before: before.get(id) ?? "", after: after.get(id) ?? "(removed)" }],
    );
  }
  const changedBlocks = changes.map((change) => change.block);
  const page = forPage(answer.text, each.request);
  const givenToMinds = answer.toolCalls.flatMap((call) =>
    ["add_to_mind", "create_mind", "edit_mind"].includes(call.tool) &&
    typeof call.input.content === "string"
      ? [call.input.content]
      : [],
  );
  const outcome: RequestOutcome = {
    expected: each.expected,
    answer,
    verdict,
    citedInMinds,
    changedBlocks,
    addedBlocks,
    sourcesAsText: sourcesAsTextIn,
    writtenInMinds,
    addedToPage: page,
    givenToMinds,
    askedAgain: asks.again > 0,
    scoped,
    texts: each.expected.textDates ? DATED_TEXTS : undefined,
  };
  const failures = requestFailures(outcome);
  return {
    id: each.id,
    language: each.language,
    request: each.request,
    model,
    failures,
    reply: answer.text,
    page,
    removed: takenOut(outcome).removed,
    rewritten: rewrittenLater(outcome),
    pointersLeft: pointersIn([page, ...writtenInMinds].join("\n\n")).length,
    calls: record,
    changes,
    seconds: (Date.now() - started) / 1000,
    answeredSeconds,
    error: answer.status === "failed" ? `${answer.errorKind}: ${answer.errorMessage}` : null,
    usage,
    droppedMarkers,
    asks,
    support,
    claims: each.summary ? citedClaims(answer.text) : null,
  };
}

// Never the model under test, which would judge its own claims.
const JUDGE_MODEL = "claude-sonnet-5-5";
const JUDGE_OF_SONNET = "claude-opus-5-5";

/**
 * The model that judges claims: INCARNAMIND_EVAL_JUDGE_*, else, on an
 * Anthropic key, Claude Sonnet 5.5, or Opus 5.5 when Sonnet is under test.
 */
export function judgeOf(config: ReturnType<typeof readConfig>, chat: ChatSettings): ChatSettings {
  if (config.judge) return config.judge;
  if (chat.kind !== "anthropic") {
    throw new Error(
      "Set INCARNAMIND_EVAL_JUDGE_KIND, _MODEL and _KEY: the claims are judged by another model than the one asked.",
    );
  }
  return { ...chat, modelId: chat.modelId === JUDGE_MODEL ? JUDGE_OF_SONNET : JUDGE_MODEL };
}
