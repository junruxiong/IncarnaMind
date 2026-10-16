/**
 * IncarnaMind's own record of what an Answer's Tool calls did: what
 * each one changed, and in which Mind, or why it changed nothing. It is
 * written from the calls' cards, never from the model's words, so a later
 * Question in the conversation knows what really happened before (./context),
 * and a reply that claims a change no call made is asked again with it
 * (./claimedChanges). In English, as the rest of the instructions are.
 */
import type { AnswerToolCall, MindChange, MindProblem } from "../api";

/** A Mind's name now, by its id; null for one deleted or unknown. */
export type MindNames = (mindId: string) => string | null;

const field = (call: AnswerToolCall, name: string): string => {
  const value = call.input[name];
  return typeof value === "string" ? value.trim() : "";
};

const quoted = (text: string) => `“${text.replace(/\s+/g, " ").slice(0, 120)}”`;

const counted = (count: number, one: string, many = `${one}s`) =>
  `${count} ${count === 1 ? one : many}`;

function whyNot(problem: MindProblem | undefined): string {
  switch (problem?.kind) {
    case "waiting":
      return problem.mixed
        ? "your own waiting changes sit beside the User's text there"
        : `${counted(problem.count, "change")} the assistant can't revise waited there for the User`;
    case "no-mind":
      return "no Mind was open and none was named";
    case "not-found":
      return `no Mind is called ${quoted(problem.name)}`;
    case "several":
      return `several Minds are called ${quoted(problem.name)}`;
    case "no-heading":
      return `it has no heading ${quoted(problem.heading)}`;
    case "no-place":
      return `it has no heading or paragraph ${quoted(problem.place)}`;
    case "no-text":
      return "there was no text there to change";
    case "too-long":
      return "it was too much text to change at once";
    case "changed":
      return "its text changed while it was being read";
    case "content-not-own":
      return "new text can only replace your own waiting changes";
    case "stopped":
      return "it was stopped";
    case "local-mode":
      return "“Keep everything on this computer” is on";
    case "not-ready":
      return "the model couldn't be used";
    case "model":
      return `the model failed: ${problem.message}`;
    case undefined:
      return "it failed";
  }
}

function citationsOf(change: MindChange): string {
  const cited = change.cited ? `, with ${counted(change.cited, "Citation")}` : "";
  const left = change.unrecorded
    ? `; ${counted(change.unrecorded, "marker")} had no Citation recorded with cite, so ${change.unrecorded === 1 ? "it was" : "they were"} left out`
    : "";
  return `${cited}${left}`;
}

function mindOf(call: AnswerToolCall, names: MindNames): string {
  const problem = call.mindProblem;
  const id = call.mindChange?.mindId ?? (problem && "mindId" in problem ? problem.mindId : null);
  const name = (id && names(id)) || field(call, "mind") || field(call, "title");
  return name ? quoted(name) : "the Mind the User was in";
}

function mindLine(call: AnswerToolCall, names: MindNames): string {
  const mind = mindOf(call, names);
  const change = call.mindChange;
  const section = field(call, "part") ? ` (section ${quoted(field(call, "part"))})` : "";
  if (call.status !== "done") {
    const verb = call.tool === "read_mind" ? "couldn't read it" : "nothing changed";
    return `${call.tool} ${mind}${section}: ${verb}: ${whyNot(call.mindProblem)}`;
  }
  const proposed = change
    ? `proposed ${counted(change.proposals, "change")}, waiting for the User${citationsOf(change)}`
    : "";
  switch (call.tool) {
    case "read_mind":
      return `read_mind ${mind}: read it`;
    case "create_mind":
      return change && change.proposals > 0
        ? `create_mind: made the Mind ${mind} and ${proposed}`
        : `create_mind: made the Mind ${mind}, empty`;
    case "add_to_mind":
      return `add_to_mind ${mind}: proposed ${counted(change?.proposals ?? 0, "new Block")}, waiting for the User${change ? citationsOf(change) : ""}`;
    default:
      if (!change) return `${call.tool} ${mind}${section}: no change was needed`;
      return change.revised
        ? `${call.tool} ${mind}${section}: revised your own waiting text in place, and ${proposed}`
        : `${call.tool} ${mind}${section}: ${proposed}`;
  }
}

/** A call's line in the record; null for one with nothing to say (a note that a Connector needs signing in). */
function lineOf(call: AnswerToolCall, names: MindNames): string | null {
  const done = call.status === "done";
  switch (call.source) {
    case "minds":
      return mindLine(call, names);
    case "documents":
      if (call.tool === "calculate") {
        return `calculate: ${counted(call.resultCount ?? 0, "figure")} worked out`;
      }
      return done
        ? `${call.tool} ${quoted(field(call, "query"))}: ${counted(call.resultCount ?? 0, "Passage")}`
        : `${call.tool} ${quoted(field(call, "query"))}: failed`;
    case "web":
      if (call.tool === "read_web_page") {
        const page = call.webPage;
        return page && "documentId" in page
          ? `read_web_page ${field(call, "url")}: saved as the Document ${quoted(page.name)}`
          : `read_web_page ${field(call, "url")}: couldn't be read`;
      }
      return `${call.tool} ${quoted(field(call, "query"))}: ${done ? "done" : "failed"}`;
    case "sources":
      return `${call.tool} ${quoted(field(call, "query"))}: ${done ? counted(call.resultCount ?? 0, "work") : "failed"}`;
    case "skill":
      if (call.tool === "run_skill_script") {
        return `run_skill_script ${field(call, "script")}: ${done ? "ran" : call.approval === "denied" ? "the User denied it" : "didn't run"}`;
      }
      return `${call.tool} ${quoted(field(call, "name") || field(call, "skill"))}: ${done ? "done" : "failed"}`;
    case "connector":
      if (call.signInRequired) return null;
      return `${call.tool} (the Connector ${quoted(call.connector?.name ?? "")}): ${
        done
          ? "done"
          : call.approval === "denied"
            ? "the User denied it, so it wasn't made"
            : "failed"
      }`;
  }
}

export function toolRecordLines(calls: readonly AnswerToolCall[], names: MindNames): string[] {
  return calls.flatMap((call) => {
    const line = lineOf(call, names);
    return line ? [`- ${line}`] : [];
  });
}

/**
 * The record of an earlier Answer's calls, as the message after it starts with; "" with none.
 * `cited`: how many Citations its reply holds, which the model reads as plain references.
 */
export function toolRecord(calls: readonly AnswerToolCall[], names: MindNames, cited = 0): string {
  const lines = [
    ...toolRecordLines(calls, names),
    ...(cited > 0
      ? [
          `- cite: ${counted(cited, "Citation")} in your reply; each reads as its source in brackets, like [Tides, p. 3]`,
        ]
      : []),
  ];
  if (lines.length === 0) return "";
  return [
    "<tool-record>",
    "What the Tools of your Answer above really did, as IncarnaMind recorded it:",
    ...lines,
    "</tool-record>",
  ].join("\n");
}

/** A record's block, as a message holds it: what reads it for anything but the model leaves it out. */
export const TOOL_RECORD = /<tool-record>[\s\S]*?<\/tool-record>\s*/g;
