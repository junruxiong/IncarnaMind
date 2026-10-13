/**
 * Research (#130): a Task that plans, then searches the Library and the web
 * step by step, as Answers do, before it writes.
 *
 * - The plan is one request to the composer's model: the steps it proposes
 *   for the topic, which the User edits on the plan card before Start.
 * - Each step is a step of the Task's journal (`step:<n>`): an Answer kept
 *   nowhere (`answerResearch`), from the Library within the composer's
 *   Search scope and, with Web on, the web, whose pages are saved as
 *   Documents as they are read (#128). Its notes and the Documents it used
 *   are its result, read by the steps after it. Its searches and the pages it
 *   read are steps too (`call:…`), journalled as each ends, so its progress
 *   says what it did.
 * - A step comes below Answers as every Task's does (its turn), is tried
 *   again after a pause when its provider limits requests, and stops the Task
 *   at the monthly cap or a plan's limit (../runs/stop). Quit or paused, the
 *   Task goes on from the first step not done.
 */
import { generateText } from "ai";
import type { MessageKey, MessageParams } from "../../shared/i18n";
import {
  MAX_RESEARCH_STEP,
  MAX_RESEARCH_STEPS,
  MAX_RESEARCH_TOPIC,
  RESEARCH_TASK,
} from "../../shared/research";
import type { CellAnswer } from "../answers";
import type { QuestionContext } from "../answers/context";
import type {
  AnswerToolCall,
  ChatModelChoice,
  ChatReadiness,
  PlanResearchInput,
  ResearchPlan,
  ResearchPlanResult,
  ResearchSources,
  ResearchStartResult,
  ResearchTarget,
  SearchScope,
  StartResearchInput,
  Task,
} from "../api";
import { ChatNotReadyError, ConsentDeclinedError, InvalidInputError, isRecord } from "../errors";
import type { PreparedChatModel } from "../providers/chat";
import { classifyProviderError } from "../providers/providerErrors";
import type { StepRecord } from "../runs/journal";
import { RunItemError, RunItemGaveWay, runItems } from "../runs/stop";
import type { TaskEnd, TaskKind, TaskRun } from "../runs/tasks";
import { parseSearchScope } from "../scope";
import { notesOf, PLAN_INSTRUCTIONS, planPrompt, readPlan, stepContext } from "./prompt";

/** A plan that takes longer than this (a local model loading) fails. */
const PLAN_TIMEOUT_MS = 120_000;

/** What a Research Task does, as its journal keeps it. */
interface ResearchInput {
  topic: string;
  plan: ResearchPlan;
  sources: ResearchSources;
  writeInto: ResearchTarget;
  scope: SearchScope | null;
  providerId: string;
  modelId: string;
}

/** What a step found: its notes, and the Documents it cited or read. */
interface StepFound {
  notes: string;
  documents: string[];
}

/** A search or a page read, as a step of its own. */
type CallStep =
  | { kind: "web-search" | "library-search"; query: string; found: number | null }
  | { kind: "read"; name: string }
  | { kind: "unread"; address: string };

export interface ResearchOptions {
  /** Throws unless the Mind exists and this version may write to it; returns its id. */
  requireMind(mindId: unknown): string;
  chat: {
    providerExists(providerId: string): boolean;
    readiness(choice: ChatModelChoice | undefined): Promise<ChatReadiness>;
    /** The model, once the User has allowed its data flow (asking them if needed). */
    prepareModel(choice: ChatModelChoice): Promise<PreparedChatModel>;
    /** Whether the model runs on this computer, which serves one request at a time. */
    runsLocally(choice: ChatModelChoice): boolean;
  };
  answers: {
    answerResearch(input: {
      mindId: string;
      context: QuestionContext;
      documentIds: string[] | null;
      web: boolean;
      model: ChatModelChoice;
      signal: AbortSignal;
      onCall(call: AnswerToolCall): void;
    }): Promise<CellAnswer>;
  };
  /** The ids of the live Documents a Search scope covers; null with none (every Document). */
  resolveScope(scope: SearchScope): string[] | null;
  tasks: {
    start(kind: string, mindId: string, input: ResearchInput): Task;
    /** A Task's journal, in the order its steps first started. */
    steps(taskId: string): StepRecord[];
  };
  /** In the User's language. */
  say(key: MessageKey, params?: MessageParams): string;
}

function text(value: unknown, what: string, max: number): string {
  if (typeof value !== "string") throw new InvalidInputError(`${what} must be text.`);
  const line = value.replace(/\s+/g, " ").trim();
  if (line === "") throw new InvalidInputError(`${what} is empty.`);
  if (line.length > max) throw new InvalidInputError(`${what} is longer than ${max} characters.`);
  return line;
}

function parseModel(value: unknown): ChatModelChoice | null {
  if (value === undefined || value === null) return null;
  if (
    !isRecord(value) ||
    typeof value.providerId !== "string" ||
    typeof value.modelId !== "string" ||
    value.providerId === "" ||
    value.modelId === ""
  ) {
    throw new InvalidInputError("model must be a provider and a model.");
  }
  return { providerId: value.providerId, modelId: value.modelId };
}

function parseSources(value: unknown): ResearchSources {
  if (!isRecord(value) || typeof value.web !== "boolean" || typeof value.library !== "boolean") {
    throw new InvalidInputError("sources must say whether to search the web and the Library.");
  }
  if (!value.web && !value.library) {
    throw new InvalidInputError("Research needs the web, the Library or both.");
  }
  return { web: value.web, library: value.library };
}

function parseSteps(value: unknown): string[] {
  if (!Array.isArray(value)) throw new InvalidInputError("A plan's steps must be a list.");
  const steps = value
    .map((step) => (typeof step === "string" ? step.replace(/\s+/g, " ").trim() : step))
    .filter((step) => step !== "");
  if (steps.length === 0) throw new InvalidInputError("A plan needs a step.");
  if (steps.length > MAX_RESEARCH_STEPS) {
    throw new InvalidInputError(`A plan has at most ${MAX_RESEARCH_STEPS} steps.`);
  }
  return steps.map((step) => text(step, "A step", MAX_RESEARCH_STEP));
}

function parseTarget(value: unknown): ResearchTarget {
  if (isRecord(value) && value.newMind === true) return { newMind: true };
  if (isRecord(value) && typeof value.mindId === "string" && value.mindId !== "") {
    return { mindId: value.mindId };
  }
  throw new InvalidInputError("writeInto must be a Mind or a new Mind.");
}

function inputOf(input: unknown): ResearchInput | null {
  if (!isRecord(input) || !isRecord(input.plan) || !isRecord(input.sources)) return null;
  const { topic, plan, sources, writeInto, scope, providerId, modelId } = input;
  if (
    typeof topic !== "string" ||
    typeof plan.title !== "string" ||
    !Array.isArray(plan.steps) ||
    !plan.steps.every((step) => typeof step === "string") ||
    typeof sources.web !== "boolean" ||
    typeof sources.library !== "boolean" ||
    typeof providerId !== "string" ||
    typeof modelId !== "string"
  ) {
    return null;
  }
  let target: ResearchTarget;
  let within: SearchScope | null;
  try {
    target = parseTarget(writeInto);
    within = parseSearchScope(scope);
  } catch {
    return null;
  }
  return {
    topic,
    plan: { title: plan.title, steps: plan.steps as string[] },
    sources: { web: sources.web, library: sources.library },
    writeInto: target,
    scope: within,
    providerId,
    modelId,
  };
}

const stepKey = (index: number) => `step:${index}`;

function foundOf(step: StepRecord | null): StepFound | null {
  if (step?.status !== "completed" || !isRecord(step.result)) return null;
  const { notes, documents } = step.result;
  return {
    notes: typeof notes === "string" ? notes : "",
    documents: Array.isArray(documents)
      ? documents.filter((id): id is string => typeof id === "string")
      : [],
  };
}

/** A search or a page read worth a step of its own; null for the rest (`cite`, a Skill). */
function callStepOf(call: AnswerToolCall): CallStep | null {
  const query = typeof call.input.query === "string" ? call.input.query : "";
  if (call.tool === "search_web") return { kind: "web-search", query, found: call.resultCount };
  if (call.tool === "search_documents") {
    return { kind: "library-search", query, found: call.resultCount };
  }
  if (call.tool !== "read_web_page" || !call.webPage) return null;
  return "documentId" in call.webPage
    ? { kind: "read", name: call.webPage.name }
    : { kind: "unread", address: call.webPage.address };
}

function callStepFrom(value: unknown): CallStep | null {
  if (!isRecord(value)) return null;
  if (
    (value.kind === "web-search" || value.kind === "library-search") &&
    typeof value.query === "string"
  ) {
    return {
      kind: value.kind,
      query: value.query,
      found: typeof value.found === "number" ? value.found : null,
    };
  }
  if (value.kind === "read" && typeof value.name === "string")
    return { kind: "read", name: value.name };
  if (value.kind === "unread" && typeof value.address === "string") {
    return { kind: "unread", address: value.address };
  }
  return null;
}

const hostOf = (address: string) => {
  try {
    return new URL(address).host || address;
  } catch {
    return address;
  }
};

export type Research = ReturnType<typeof createResearch>;

export function createResearch(options: ResearchOptions) {
  type Readied =
    | { model: ChatModelChoice; prepared: PreparedChatModel }
    | Exclude<ResearchStartResult, { started: true }>;

  /** The model, ready and allowed; or why it can't be. */
  async function readyModel(picked: ChatModelChoice | null): Promise<Readied> {
    const choice = picked && options.chat.providerExists(picked.providerId) ? picked : undefined;
    const readiness = await options.chat.readiness(choice);
    if (!readiness.ready) return { started: false, reason: "not-ready", readiness };
    const model = { providerId: readiness.provider.id, modelId: readiness.modelId };
    try {
      return { model, prepared: await options.chat.prepareModel(model) };
    } catch (error) {
      if (error instanceof ChatNotReadyError) {
        return { started: false, reason: "not-ready", readiness: error.readiness };
      }
      if (error instanceof ConsentDeclinedError)
        return { started: false, reason: "consent-declined" };
      throw error;
    }
  }

  function describe(
    step: StepRecord | undefined,
    key: string,
    input: ResearchInput | null,
  ): string {
    if (key.startsWith("step:")) {
      return input?.plan.steps[Number(key.slice("step:".length))] ?? key;
    }
    const call = callStepFrom(step?.result);
    if (!call) return key;
    switch (call.kind) {
      case "web-search":
        return options.say("research.step.searchedWeb", { query: call.query });
      case "library-search":
        return options.say("research.step.searchedLibrary", { query: call.query });
      case "read":
        return options.say("research.step.read", { name: call.name });
      case "unread":
        return options.say("research.step.unread", { host: hostOf(call.address) });
    }
  }

  /** Does one step, from where the journal is: its Answer, and a step for each search and page read. */
  async function doStep(
    task: TaskRun,
    input: ResearchInput,
    index: number,
    model: ChatModelChoice,
    local: boolean,
  ): Promise<{ found: StepFound } | { failed: string } | { end: TaskEnd }> {
    const key = stepKey(index);
    const notes = Array.from(
      { length: index },
      (_, at) => foundOf(task.step(stepKey(at)))?.notes ?? "",
    );
    const context = stepContext({
      topic: input.topic,
      plan: input.plan,
      index,
      sources: input.sources,
      notes,
    });
    const documentIds = input.sources.library
      ? input.scope
        ? options.resolveScope(input.scope)
        : null
      : [];
    task.started(key);
    const attempt = task.step(key)?.attempts ?? 1;
    const read = new Set<string>();
    let calls = 0;
    const onCall = (call: AnswerToolCall) => {
      if (call.webPage && "documentId" in call.webPage) read.add(call.webPage.documentId);
      const shown = callStepOf(call);
      if (!shown) return;
      calls += 1;
      const callKey = `call:${index}:${attempt}:${calls}`;
      task.started(callKey);
      task.settled(
        callKey,
        call.status === "done" && shown.kind !== "unread" ? "completed" : "failed",
        shown,
      );
    };
    const { outcomes, stop } = await runItems(
      [index],
      async () => {
        const turn = await task.turn(local);
        try {
          const answer = await options.answers.answerResearch({
            mindId: task.mindId ?? "",
            context,
            documentIds,
            web: input.sources.web,
            model,
            signal: AbortSignal.any([task.signal, turn.signal]),
            onCall,
          });
          if (turn.signal.aborted && !task.signal.aborted) throw new RunItemGaveWay();
          if (answer.status === "stopped") throw new DOMException("Stopped.", "AbortError");
          if (answer.status === "failed" && answer.error) throw new RunItemError(answer.error);
          return answer;
        } catch (error) {
          if (turn.signal.aborted && !task.signal.aborted) throw new RunItemGaveWay();
          throw error;
        } finally {
          turn.end();
        }
      },
      { signal: task.signal },
    );
    if (stop) {
      return {
        end: {
          status: "paused",
          reason: stop.reason,
          resetsAt: stop.reason === "plan-limit" ? stop.resetsAt : null,
        },
      };
    }
    const outcome = outcomes[0];
    if (!outcome || outcome.status === "stopped" || outcome.status === "not-checked") {
      return { end: { status: "stopped" } };
    }
    if (outcome.status === "failed") {
      task.settled(key, "failed", { problem: outcome.error.kind });
      return { failed: outcome.error.kind };
    }
    const answer = outcome.value;
    const found: StepFound = {
      notes: notesOf(answer.markdown),
      documents: [...new Set([...answer.citations.map((each) => each.documentId), ...read])],
    };
    task.settled(key, "completed", found);
    return { found };
  }

  async function runTask(task: TaskRun): Promise<TaskEnd> {
    const input = inputOf(task.input);
    if (!input || !task.mindId) return { status: "failed", problem: "unknown" };
    const ready = await readyModel({ providerId: input.providerId, modelId: input.modelId });
    if (!("model" in ready)) return { status: "paused", reason: "not-ready" };
    if (task.signal.aborted) return { status: "stopped" };
    const local = options.chat.runsLocally(ready.model);
    const total = input.plan.steps.length;
    const doneSteps = () =>
      input.plan.steps.filter((_, index) => task.step(stepKey(index))?.status === "completed")
        .length;
    task.progress({ done: doneSteps(), total });
    let problem: string | null = null;
    for (const [index] of input.plan.steps.entries()) {
      if (task.signal.aborted) return { status: "stopped" };
      if (task.step(stepKey(index))?.status === "completed") continue;
      const result = await doStep(task, input, index, ready.model, local);
      if ("end" in result) return result.end;
      if ("failed" in result) problem = result.failed;
      task.progress({ done: doneSteps(), total });
    }
    if (doneSteps() === 0) return { status: "failed", problem: problem ?? "unknown" };
    return { status: "done" };
  }

  const taskKind: TaskKind = {
    kind: RESEARCH_TASK,
    run: runTask,
    questionOf: (input) => inputOf(input)?.topic ?? null,
    describeSteps(task, keys) {
      const input = inputOf(task.input);
      const steps = new Map(options.tasks.steps(task.id).map((step) => [step.key, step]));
      return keys.map((key) => describe(steps.get(key), key, input));
    },
  };

  return {
    taskKind,

    async plan(given: PlanResearchInput): Promise<ResearchPlanResult> {
      if (!isRecord(given)) throw new InvalidInputError("Planning needs a topic.");
      options.requireMind(given.mindId);
      const topic = text(given.topic, "The topic", MAX_RESEARCH_TOPIC);
      const sources = parseSources(given.sources);
      const model = parseModel(given.model);
      const change = isRecord(given.change)
        ? {
            steps: parseSteps(given.change.steps),
            text: text(given.change.text, "The change", MAX_RESEARCH_TOPIC),
          }
        : null;
      const ready = await readyModel(model);
      if (!("model" in ready)) {
        return ready.reason === "not-ready"
          ? { planned: false, reason: "not-ready", readiness: ready.readiness }
          : { planned: false, reason: "consent-declined" };
      }
      let reply: string;
      try {
        const result = await generateText({
          model: ready.prepared.model,
          instructions: PLAN_INSTRUCTIONS,
          prompt: planPrompt(topic, sources, change),
          maxOutputTokens: 1_024,
          maxRetries: 1,
          abortSignal: AbortSignal.timeout(PLAN_TIMEOUT_MS),
        });
        reply = result.text;
      } catch (error) {
        return { planned: false, reason: "failed", error: classifyProviderError(error) };
      }
      const plan = readPlan(reply, topic);
      if (!plan) {
        return {
          planned: false,
          reason: "failed",
          error: { kind: "provider", message: "The model gave no steps." },
        };
      }
      return { planned: true, plan };
    },

    async start(given: StartResearchInput): Promise<ResearchStartResult> {
      if (!isRecord(given) || !isRecord(given.plan)) {
        throw new InvalidInputError("Research needs a topic and a plan.");
      }
      const mindId = options.requireMind(given.mindId);
      const topic = text(given.topic, "The topic", MAX_RESEARCH_TOPIC);
      const plan: ResearchPlan = {
        title: text(given.plan.title, "The plan's title", MAX_RESEARCH_STEP),
        steps: parseSteps(given.plan.steps),
      };
      const sources = parseSources(given.sources);
      const writeInto = parseTarget(given.writeInto);
      if ("mindId" in writeInto) options.requireMind(writeInto.mindId);
      const scope = parseSearchScope(given.scope);
      const ready = await readyModel(parseModel(given.model));
      if (!("model" in ready)) return ready;
      const task = options.tasks.start(RESEARCH_TASK, mindId, {
        topic,
        plan,
        sources,
        writeInto,
        scope,
        providerId: ready.model.providerId,
        modelId: ready.model.modelId,
      });
      return { started: true, task };
    },
  };
}
