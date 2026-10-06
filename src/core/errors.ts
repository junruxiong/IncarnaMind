/** Thrown when a caller of the core's public interface passes malformed input. */
export class InvalidInputError extends Error {
  override name = "InvalidInputError";
}

export function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
