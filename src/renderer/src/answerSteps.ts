import type {
  AnswerPhase,
  AnswerToolCall,
  AskedFrom,
  ConversationAnswer,
  MarkedCitation,
  MindRead,
} from "../../core/api";

/**
 * An Answer's steps in the assistant (#391), from what its row keeps: its
 * phases, Tool calls, reasoning and Citations. The folded line sums them up,
 * the timeline lists them, and while the Answer is written the line says
 * which step it is on. Only data: the component words them.
 */

/** The Tool an Answer searches the Documents with. */
export const SEARCH = "search_documents";
export const RUN_SCRIPT = "run_skill_script";
/** The Documents' Tool whose figures an Answer shows: one call sums up all of them (ADR-0021). */
export const CALCULATE = "calculate";

/** What the steps are read from. */
export interface StepsInput {
  answer: Pick<ConversationAnswer, "text" | "citations" | "toolCalls" | "model" | "steps" | "read">;
  /**
   * The Mind in focus when the Question was asked; null when none was. Its
   * Answer read it (`answer.read`), unless it couldn't: say, an Answer written
   * before #544.
   */
  askedFrom: AskedFrom | null;
  /** It was asked while its Mind's meeting was recording: the assistant listened to the live transcript. */
  meeting: boolean;
}

/** A source a search led to: the first of its Citations at one place in one Document. */
export type StepSource = MarkedCitation;

export type StepRow =
  /** The model's reasoning summary, and how long it thought. */
  | { kind: "reasoning"; model: string; summary: string; seconds: number }
  /** A model that gave no reasoning summary, and how long the Answer took when that is known. */
  | { kind: "answered"; model: string; seconds: number | null }
  /** The Mind the Answer read, all of it or part of it. */
  | { kind: "read"; mind: MindRead }
  /** The Mind it was asked from, which it didn't read. */
  | { kind: "asked"; mind: AskedFrom }
  /** One search, or several in a row, with the sources their Citations point to. */
  | {
      kind: "search";
      calls: AnswerToolCall[];
      found: number;
      used: number;
      sources: StepSource[];
    }
  /** A search that failed, where it happened. */
  | { kind: "failed-search"; call: AnswerToolCall }
  /** A Connector's Tool, Find sources or a Skill script. */
  | { kind: "call"; call: AnswerToolCall }
  /** A Skill, loaded by the model or chosen for the Question, with the files it read. */
  | { kind: "skill"; name: string; use: AnswerToolCall | null; reads: AnswerToolCall[] }
  /** The figures the Answer shows, each calculated from its cited inputs. */
  | { kind: "calculated"; count: number }
  /** The Citations checked against their sources, and how many weren't found. */
  | { kind: "check"; count: number; notFound: number };

export type FoldedLine =
  | { kind: "searched"; sources: number; extraScopes: number }
  | { kind: "searched-web"; sources: number }
  | { kind: "listened"; sources: number }
  | { kind: "read"; mind: MindRead }
  | { kind: "asked"; mind: AskedFrom }
  | { kind: "answered"; model: string };

export type LiveStep =
  | { kind: "waiting" }
  | { kind: "consent" }
  | { kind: "loading" }
  | { kind: "reading"; mind: AskedFrom }
  | { kind: "searching" }
  | { kind: "call"; call: AnswerToolCall }
  | { kind: "writing" }
  | { kind: "checking"; count: number }
  | { kind: "checking-figures" };

const isSearch = (call: AnswerToolCall) => call.source === "documents" && call.tool === SEARCH;
export const isCalculated = (call: AnswerToolCall) =>
  call.source === "documents" && call.tool === CALCULATE;

/** Whole seconds, at least 1. */
const seconds = (ms: number) => Math.max(1, Math.round(ms / 1000));

/** A text field of what a Tool call was asked, or "". */
const field = (call: AnswerToolCall, name: string): string => {
  const value: unknown = call.input?.[name];
  return typeof value === "string" ? value : "";
};

/** The Citation markers in an Answer's text, each once, in order. */
export function markersIn(text: string): number[] {
  const markers: number[] = [];
  for (const match of text.matchAll(/\[\^(\d{1,4})\]/g)) {
    const marker = Number(match[1]);
    if (!markers.includes(marker)) markers.push(marker);
  }
  return markers;
}

/** The first row: the model's reasoning, or the model that answered and how long it took. */
function modelRow(answer: StepsInput["answer"]): StepRow | null {
  const model = answer.model?.modelId;
  if (!model) return null;
  const reasoning = answer.steps?.reasoning;
  if (reasoning) {
    return { kind: "reasoning", model, summary: reasoning.summary, seconds: seconds(reasoning.ms) };
  }
  const sentAt = Date.parse(answer.steps?.sentAt ?? "");
  const finishedAt = Date.parse(answer.steps?.finishedAt ?? "");
  const took = finishedAt - sentAt;
  return { kind: "answered", model, seconds: Number.isNaN(took) ? null : seconds(took) };
}

/** Where a Citation points, to tell two sources apart. */
const placeOf = (citation: MarkedCitation) =>
  `${citation.documentId}\n${citation.pageFrom}-${citation.pageTo}\n${JSON.stringify(citation.location)}`;

/** The Mind the Answer read, or the one it was asked from and didn't read. */
const mindRow = ({ answer, askedFrom }: StepsInput) =>
  answer.read
    ? ({ kind: "read", mind: answer.read } as const)
    : askedFrom
      ? ({ kind: "asked", mind: askedFrom } as const)
      : null;

/**
 * The rows of the unfolded timeline, in order: the model's reasoning (or
 * who answered), the Mind read, unless the folded line already says so, the
 * searches and other calls in the order they started (searches in a row as
 * one, a failed one where it happened), the figures calculated, and the check
 * of the Citations last. A call still waiting for the User has no row: its
 * approval card shows instead.
 */
export function stepRows(input: StepsInput): StepRow[] {
  const { answer } = input;
  const rows: StepRow[] = [];
  const first = modelRow(answer);
  if (first) rows.push(first);
  const mind = mindRow(input);
  if (mind && foldedLine(input)?.kind !== mind.kind) rows.push(mind);

  const searches: Extract<StepRow, { kind: "search" }>[] = [];
  const skills = new Map<string, Extract<StepRow, { kind: "skill" }>>();
  let joining: Extract<StepRow, { kind: "search" }> | null = null;
  let calculated = 0;
  for (const call of answer.toolCalls) {
    if (call.approval === "waiting") continue;
    if (isCalculated(call)) {
      calculated = call.resultCount ?? 0;
      continue;
    }
    if (isSearch(call)) {
      if (call.status === "failed") {
        rows.push({ kind: "failed-search", call });
        joining = null;
        continue;
      }
      if (!joining) {
        joining = { kind: "search", calls: [], found: 0, used: 0, sources: [] };
        rows.push(joining);
        searches.push(joining);
      }
      joining.calls.push(call);
      joining.found += call.resultCount ?? 0;
      continue;
    }
    joining = null;
    if (call.source === "skill" && call.tool !== RUN_SCRIPT) {
      const name = call.tool === "use_skill" ? field(call, "name") : field(call, "skill");
      let skill = skills.get(name);
      if (!skill) {
        skill = { kind: "skill", name, use: null, reads: [] };
        skills.set(name, skill);
        rows.push(skill);
      }
      if (call.tool === "use_skill") skill.use ??= call;
      else skill.reads.push(call);
      continue;
    }
    rows.push({ kind: "call", call });
  }

  // Each Citation goes under the first search that found its Passage. Searches made before
  // they kept what they found give theirs to the last search.
  const kept = searches.some((search) => search.calls.some((call) => call.passages));
  const used = new Map<Extract<StepRow, { kind: "search" }>, Set<string>>();
  for (const citation of answer.citations) {
    const search = kept
      ? searches.find((each) =>
          each.calls.some((call) => call.passages?.includes(citation.passageId)),
        )
      : searches.at(-1);
    if (!search) continue;
    const passages = used.get(search) ?? new Set<string>();
    passages.add(citation.passageId);
    used.set(search, passages);
    if (!search.sources.some((source) => placeOf(source) === placeOf(citation))) {
      search.sources.push(citation);
    }
  }
  for (const [search, passages] of used) search.used = passages.size;

  if (calculated > 0) rows.push({ kind: "calculated", count: calculated });
  if (answer.citations.length > 0) {
    rows.push({
      kind: "check",
      count: answer.citations.length,
      notFound: answer.citations.filter((citation) => citation.check === "not-found").length,
    });
  }
  return rows;
}

/** How many Documents the Answer cites. */
const sourceCount = (citations: readonly MarkedCitation[]) =>
  new Set(citations.map((citation) => citation.documentId)).size;

/**
 * Further scopes the Answer searched beside its Question's: each Connector
 * whose Tools it called, Find sources, and the web.
 */
function extraScopes(calls: readonly AnswerToolCall[]): number {
  const ran = calls.filter((call) => call.status === "done" && !call.signInRequired);
  const connectors = new Set(
    ran.flatMap((call) =>
      call.source === "connector" && call.connector ? [call.connector.id] : [],
    ),
  );
  const also = (source: AnswerToolCall["source"]) =>
    ran.some((call) => call.source === source) ? 1 : 0;
  return connectors.size + also("sources") + also("web");
}

/**
 * The folded line: what was searched (the Documents, or else the web) and
 * how many Documents the Answer cites; with no search, the Mind it read (or
 * was asked from), or the model that answered. Null for an Answer no model
 * wrote, asked from no Mind.
 */
export function foldedLine(input: StepsInput): FoldedLine | null {
  const { answer, meeting } = input;
  const sources = sourceCount(answer.citations);
  if (meeting) return { kind: "listened", sources };
  if (answer.toolCalls.some((call) => isSearch(call) && call.status === "done")) {
    return { kind: "searched", sources, extraScopes: extraScopes(answer.toolCalls) };
  }
  if (answer.toolCalls.some((call) => call.source === "web" && call.status === "done")) {
    return { kind: "searched-web", sources };
  }
  return (
    mindRow(input) ?? (answer.model ? { kind: "answered", model: answer.model.modelId } : null)
  );
}

/**
 * The step an Answer being written is on, from its phase, its calls, its
 * text and its Citations: waiting for the User first; then loading,
 * checking quotes or figures, searching or another call running; then writing, which
 * becomes checking once every marker in the text has its Citation; before
 * any text, searching once it has searched, else reading the Mind it reads.
 */
export function liveStep({
  answer,
  phase,
  waitingForApproval,
}: Pick<StepsInput, "answer"> & {
  phase: AnswerPhase | undefined;
  waitingForApproval: boolean;
}): LiveStep {
  if (waitingForApproval) return { kind: "waiting" };
  if (phase === "waiting-for-consent") return { kind: "consent" };
  if (phase === "loading") return { kind: "loading" };
  if (phase === "checking-quotes") return { kind: "checking", count: answer.citations.length };
  if (phase === "checking-figures") return { kind: "checking-figures" };
  const running = answer.toolCalls.filter(
    (call) => call.status === "running" && !isCalculated(call),
  );
  if (phase === "searching" || running.some(isSearch)) return { kind: "searching" };
  const call = running.find((each) => !isSearch(each));
  if (call) return { kind: "call", call };
  if (answer.text.trim()) {
    const markers = markersIn(answer.text);
    const cited = new Set(answer.citations.map((citation) => citation.marker));
    return markers.length > 0 && markers.every((marker) => cited.has(marker))
      ? { kind: "checking", count: markers.length }
      : { kind: "writing" };
  }
  if (answer.read && !answer.toolCalls.some(isSearch))
    return { kind: "reading", mind: answer.read };
  return { kind: "searching" };
}

/** Steps that stand for the same thing: a new count of Citations is the same step. */
export const liveStepKey = (step: LiveStep): string =>
  step.kind === "call" ? `call:${step.call.id}` : step.kind;
