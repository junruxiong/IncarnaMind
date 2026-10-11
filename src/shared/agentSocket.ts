/**
 * How an outside agent's shim reaches the running app (#351): the stdio
 * command an agent starts connects to a local socket in the app, says which
 * pairing it is with one line, then carries MCP's own stdio framing both ways
 * unchanged (newline-delimited JSON-RPC, as the MCP spec asks a custom
 * transport over a byte stream to). Shared by the shim (src/main/agentShim.ts)
 * and the core's listener (src/core/agents/socket.ts), so it imports nothing.
 */

/** The environment variable that carries the pairing's token to the shim. */
export const AGENT_TOKEN_ENV = "INCARNAMIND_AGENT_TOKEN";

/** The environment variable that tells the shim where the app's socket is. */
export const AGENT_SOCKET_ENV = "INCARNAMIND_AGENT_SOCKET";

/** The version of the opening line; the app refuses one it doesn't know. */
export const AGENT_HANDSHAKE_VERSION = 1;

/** The longest opening line the app reads before giving up on a connection. */
export const AGENT_HANDSHAKE_MAX_BYTES = 4096;

/** The shim's opening line: the protocol's version and the pairing's token. */
export interface AgentHandshake {
  incarnamind: typeof AGENT_HANDSHAKE_VERSION;
  token: string;
}

export function handshakeLine(token: string): string {
  const handshake: AgentHandshake = { incarnamind: AGENT_HANDSHAKE_VERSION, token };
  return `${JSON.stringify(handshake)}\n`;
}

/** The token in an opening line, or null when the line isn't one this version reads. */
export function parseHandshake(line: string): string | null {
  try {
    const value: unknown = JSON.parse(line);
    if (typeof value !== "object" || value === null) return null;
    const { incarnamind, token } = value as Record<string, unknown>;
    return incarnamind === AGENT_HANDSHAKE_VERSION && typeof token === "string" && token !== ""
      ? token
      : null;
  } catch {
    return null;
  }
}
