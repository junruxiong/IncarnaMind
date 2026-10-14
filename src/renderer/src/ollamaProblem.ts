import type { OllamaStatus } from "../../core/api";

/**
 * The one action under "Ollama isn't running on this computer", from what Ollama is now:
 * asking again when it runs by now, opening its app when that is here, getting it when it
 * isn't, and none while it is being opened or looked for.
 */
export type OllamaLineAction = "try-again" | "open" | "get" | null;

export function ollamaLineAction(status: OllamaStatus | null, opening: boolean): OllamaLineAction {
  if (status === null || opening) return null;
  if (status.running) return "try-again";
  return status.installed ? "open" : "get";
}
