import type { OllamaStatus } from "../../core/api";

export type OllamaLineAction = "try-again" | "open" | "get" | null;

export function ollamaLineAction(status: OllamaStatus | null, opening: boolean): OllamaLineAction {
  if (status === null || opening) return null;
  if (status.running) return "try-again";
  return status.installed ? "open" : "get";
}
