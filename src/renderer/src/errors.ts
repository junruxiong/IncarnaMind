/**
 * The message of an error from the core, without Electron's IPC wrapping
 * ("Error invoking remote method 'core:x': InvalidInputError: …").
 */
export function errorMessage(error: unknown): string {
  const message = error instanceof Error ? error.message : String(error);
  return message.replace(/^Error invoking remote method '[^']*': /, "").replace(/^\w*Error: /, "");
}
