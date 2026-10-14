import type { AnswerToolCall } from "../core/api";

/** The Tool calls the core stored on an Answer (a JSON array), or none if unreadable. */
export function toolCallsOf(value: unknown): AnswerToolCall[] {
  if (typeof value !== "string") return [];
  try {
    const parsed: unknown = JSON.parse(value);
    return Array.isArray(parsed)
      ? parsed.filter(
          (call): call is AnswerToolCall =>
            typeof call === "object" && call !== null && typeof call.id === "string",
        )
      : [];
  } catch {
    return [];
  }
}
