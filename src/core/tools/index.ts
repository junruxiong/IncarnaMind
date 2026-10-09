/**
 * Tools (CONTEXT.md) and the Tool providers that hand them out
 * (docs/designs/agent-extensibility.md §4.1, ADR-0013). Every action an
 * Answer can call has this one shape, whichever provider it comes from:
 * Document search and `cite` from the Documents, `use_skill`,
 * `read_skill_file` and `run_skill_script` from the Skills, and each
 * Connector's own Tools from the Connectors. A later capability (Files, a
 * Browser, a Shell…) is a new provider, not a change to the Answer loop.
 *
 * Each Tool declares its Effects (CONTEXT.md): what a call with a given
 * input can do, and where (docs/designs/agent-extensibility.md §4.2).
 * IncarnaMind works them out from what it knows (the provider, the input,
 * the sandbox a program runs in); a Connector's hints only narrow them.
 * Approvals decide from them whether a call asks the User first (see
 * ../approvals).
 *
 * Nothing here depends on an agent library, or on Answers: the Answer engine
 * turns Tools into its library's own (see `toToolSet` in ../answers/engine),
 * so that library can change without the Tools or their providers.
 */
import type { Effect } from "../api";

/** v1's providers; later ones add their kind here. */
export type ToolProviderKind = "documents" | "skills" | "connector";

/** Which provider a Tool comes from: for its Tool-call card, its approvals and the log. */
export interface ToolProviderInfo {
  kind: ToolProviderKind;
  /** "documents" or "skills" for IncarnaMind's own; a Connector's id. */
  id: string;
  /** As the User knows it: "Documents", "Skills", or the Connector's name. */
  name: string;
}

/** One call of a Tool. */
export interface ToolCallContext {
  /** The id the model gave the call: its Tool-call card has the same. */
  toolCallId: string;
  /** Stops the call, e.g. when the User stops the Answer. */
  signal: AbortSignal;
}

/**
 * A Tool (CONTEXT.md) as a Run offers it to the model.
 *
 * Room is left for what comes next (docs/designs/agent-extensibility.md
 * §4.6): `untrustedResult`, for a result that carries text the User didn't
 * write.
 */
export interface Tool {
  /** The name the model calls it by, unique within the Run: "search_documents", "github__create_issue". */
  name: string;
  description: string;
  /** A JSON Schema for its arguments (an object). */
  inputSchema: Record<string, unknown>;
  provider: ToolProviderInfo;
  /** Its name at its provider: a Connector's "create_issue". The same as `name` for IncarnaMind's own. */
  providerTool: string;
  /** Its display name, when its provider gives one. */
  title?: string | null;
  /**
   * What a call with this input can do beyond the conversation it is part
   * of. None for one that only records something for its Answer (`cite`),
   * or that fails (it names a Skill the Answer can't use). A Connector's
   * hints only narrow these: they never widen what runs without asking.
   */
  effects(input: Record<string, unknown>): Effect[];
  /**
   * What a call's Tool-call card shows of its input, e.g. a search's query,
   * as text. Without it, the card shows the input as the model sent it.
   */
  shownInput?(input: Record<string, unknown>): Record<string, unknown>;
  /** Calls it; resolves with what the model reads, rejects when it fails (the model is told why). */
  call(input: Record<string, unknown>, context: ToolCallContext): Promise<string>;
}

/** Hands out Tools to a Run: the User's Connectors now; later Files, a Browser, a Shell. */
export interface ToolProvider {
  kind: ToolProviderKind;
  /** The Tools it offers now: none from a Connector that isn't ready. */
  tools(signal: AbortSignal): Promise<Tool[]>;
}

/** The names of the Documents' Tools. */
export const DOCUMENT_TOOLS = { search: "search_documents", cite: "cite" } as const;

/** The names of the Skills' Tools. */
export const SKILL_TOOLS = {
  use: "use_skill",
  readFile: "read_skill_file",
  runScript: "run_skill_script",
} as const;

/**
 * IncarnaMind's own Tools' names. They are reserved: a Tool from outside
 * IncarnaMind (a Connector's) never takes one (see `offeredTools`).
 */
export const RESERVED_TOOL_NAMES: ReadonlySet<string> = new Set<string>([
  ...Object.values(DOCUMENT_TOOLS),
  ...Object.values(SKILL_TOOLS),
]);

/** A Tool from outside IncarnaMind: a Connector's. Every other provider is IncarnaMind's own. */
const fromOutside = (tool: Tool): boolean => tool.provider.kind === "connector";

/**
 * The Tools one Run offers, from its providers' Tools, in the order the model
 * is offered them: those from outside IncarnaMind (Connectors') first, then
 * IncarnaMind's own, each in the order given. Each name is offered once.
 * IncarnaMind's own names are reserved: a Tool from outside named like one of
 * ours is left out, even in a Run that doesn't offer ours, so it never passes
 * for one of them. Of two Tools with one name, the first is kept.
 */
export function offeredTools(tools: readonly Tool[]): Tool[] {
  const ordered = [...tools.filter(fromOutside), ...tools.filter((tool) => !fromOutside(tool))];
  const names = new Set<string>();
  return ordered.filter((tool) => {
    if (names.has(tool.name) || (fromOutside(tool) && RESERVED_TOOL_NAMES.has(tool.name))) {
      return false;
    }
    names.add(tool.name);
    return true;
  });
}
