/**
 * The connection a cloud transcription stream talks over (#141): a WebSocket carrying the
 * User's key in its Authorization header, as the services take it. Tests pass a socket that
 * plays back what a service sent.
 */
import { WebSocket } from "undici";

export interface CloudSocket {
  send(data: string | Uint8Array): void;
  close(): void;
}

/**
 * What `closed` says when the service turned the connection away at the handshake for the key
 * (HTTP 401 or 403): a WebSocket's own failure carries no status, so it is asked for.
 */
export const HANDSHAKE_REFUSED = 4401;

export interface CloudSocketEvents {
  open(): void;
  message(text: string): void;
  /** It closed, by either side, or never opened (`HANDSHAKE_REFUSED` when the key was turned away). */
  closed(code: number): void;
}

export type CloudConnect = (url: string, apiKey: string, events: CloudSocketEvents) => CloudSocket;

const PROBE_TIMEOUT_MS = 5000;

async function keyTurnedAway(url: string, apiKey: string): Promise<boolean> {
  try {
    const { status } = await fetch(url.replace(/^ws/, "http"), {
      headers: { Authorization: `Bearer ${apiKey}` },
      signal: AbortSignal.timeout(PROBE_TIMEOUT_MS),
    });
    return status === 401 || status === 403;
  } catch {
    return false;
  }
}

export const connectWebSocket: CloudConnect = (url, apiKey, events) => {
  const socket = new WebSocket(url, { headers: { Authorization: `Bearer ${apiKey}` } });
  socket.binaryType = "arraybuffer";
  let opened = false;
  let closing = false;
  socket.addEventListener("open", () => {
    opened = true;
    events.open();
  });
  socket.addEventListener("message", (event) => {
    const { data } = event;
    events.message(typeof data === "string" ? data : Buffer.from(data as ArrayBuffer).toString());
  });
  socket.addEventListener("close", (event) => {
    if (opened || closing) return events.closed(event.code);
    void keyTurnedAway(url, apiKey).then((turned) =>
      events.closed(turned ? HANDSHAKE_REFUSED : event.code),
    );
  });
  return {
    send: (data) => {
      if (socket.readyState === WebSocket.OPEN) socket.send(data);
    },
    close: () => {
      closing = true;
      socket.close();
    },
  };
};
