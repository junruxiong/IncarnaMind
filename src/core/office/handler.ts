/**
 * Every request an Office task pane can make to the app: its own static pages,
 * pairing, and the calls that read a document. The handler is told which Host
 * and Origin values are the app's own and refuses everything else before it
 * reads a body or touches a pairing: a Host it doesn't know is a DNS-rebinding
 * page, an Origin it doesn't know is another site in the User's browser, and a
 * call without a token is another program on this computer.
 */
import type { IncomingMessage, ServerResponse } from "node:http";
import type { Duplex } from "node:stream";
import type { Logger } from "../adapters";
import { isRecord } from "../errors";
import {
  OFFICE_HOSTS,
  type OfficeHost,
  type OfficePairing,
  PairingRefusedError,
  type Pairings,
} from "./pairing";
import { nameCell, nameSentence } from "./review";
import { acceptWebSocket, wantsWebSocket } from "./websocket";

export interface StaticFile {
  type: string;
  body: Buffer;
}

export interface OfficeHandlerOptions {
  /** The `Host` header values that reach this server: `localhost:47361` and the like. */
  hosts: ReadonlySet<string>;
  /** The `Origin` values allowed to call it, written as a browser sends them. */
  origins: ReadonlySet<string>;
  pairings: Pairings;
  file(path: string): Promise<StaticFile | null>;
  /** What the pane's head says it searches. */
  folderName: string;
  contentSecurityPolicy: string;
  log: Logger;
  heartbeatMs?: number;
}

const MAX_BODY_BYTES = 2_000_000;
const HELLO_WAIT_MS = 5_000;
const HEARTBEAT_MS = 15_000;

interface Listener {
  send(event: string, data: unknown): void;
  end(): void;
}

class Refusal extends Error {
  readonly status: number;
  readonly reason: string;

  constructor(status: number, reason: string) {
    super(reason);
    this.status = status;
    this.reason = reason;
  }
}

const bearer = (request: IncomingMessage): string | null => {
  const value = request.headers.authorization;
  return value?.startsWith("Bearer ") ? value.slice(7) : null;
};

function isHost(value: unknown): value is OfficeHost {
  return typeof value === "string" && (OFFICE_HOSTS as readonly string[]).includes(value);
}

async function readJson(request: IncomingMessage): Promise<Record<string, unknown>> {
  if (!request.headers["content-type"]?.toLowerCase().startsWith("application/json")) {
    throw new Refusal(415, "content-type");
  }
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of request) {
    size += (chunk as Buffer).length;
    if (size > MAX_BODY_BYTES) throw new Refusal(413, "too-large");
    chunks.push(chunk as Buffer);
  }
  let body: unknown;
  try {
    body = JSON.parse(Buffer.concat(chunks).toString("utf8"));
  } catch {
    throw new Refusal(400, "json");
  }
  if (!isRecord(body)) throw new Refusal(400, "json");
  return body;
}

const isTable = (value: unknown): value is unknown[][] =>
  Array.isArray(value) && value.length <= 5000 && value.every((row) => Array.isArray(row));

export function createOfficeHandler(options: OfficeHandlerOptions) {
  const listeners = new Map<string, Set<Listener>>();

  const listen = (pairingId: string, listener: Listener) => {
    const set = listeners.get(pairingId) ?? new Set<Listener>();
    set.add(listener);
    listeners.set(pairingId, set);
    return () => set.delete(listener);
  };

  const corsHeaders = (request: IncomingMessage): Record<string, string> => {
    const origin = request.headers.origin;
    if (origin === undefined || !options.origins.has(origin)) return {};
    return {
      "Access-Control-Allow-Origin": origin,
      Vary: "Origin",
    };
  };

  const baseHeaders = (request: IncomingMessage): Record<string, string> => ({
    "Cache-Control": "no-store",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    ...corsHeaders(request),
  });

  const sendJson = (
    request: IncomingMessage,
    response: ServerResponse,
    status: number,
    body?: unknown,
  ) => {
    response.writeHead(status, {
      ...baseHeaders(request),
      ...(body !== undefined && { "Content-Type": "application/json" }),
    });
    response.end(body === undefined ? undefined : JSON.stringify(body));
  };

  const checkWho = (request: IncomingMessage) => {
    const host = request.headers.host?.toLowerCase();
    if (!host || !options.hosts.has(host)) throw new Refusal(421, "host");
    const origin = request.headers.origin;
    if (origin !== undefined && !options.origins.has(origin)) throw new Refusal(403, "origin");
  };

  const authenticate = async (request: IncomingMessage): Promise<OfficePairing> => {
    const token = bearer(request);
    const pairing = token ? await options.pairings.authenticate(token) : null;
    if (!pairing) throw new Refusal(401, "token");
    return pairing;
  };

  const preflight = (request: IncomingMessage, response: ServerResponse) => {
    if (request.headers.origin === undefined) throw new Refusal(403, "origin");
    response.writeHead(204, {
      ...baseHeaders(request),
      "Access-Control-Allow-Headers": "authorization, content-type",
      "Access-Control-Allow-Methods": "GET, POST, DELETE",
      "Access-Control-Max-Age": "600",
      ...(request.headers["access-control-request-private-network"] === "true" && {
        "Access-Control-Allow-Private-Network": "true",
      }),
    });
    response.end();
  };

  const api = async (
    request: IncomingMessage,
    response: ServerResponse,
    method: string,
    path: string,
  ) => {
    if (method === "GET" && path === "/api/ping") {
      return sendJson(request, response, 200, { app: "IncarnaMind", protocol: 1 });
    }
    if (method === "POST" && path === "/api/pair") {
      if (request.headers.origin === undefined) throw new Refusal(403, "origin");
      const body = await readJson(request);
      if (typeof body.code !== "string" || !isHost(body.host)) throw new Refusal(400, "fields");
      try {
        const { token, pairing } = await options.pairings.redeem(body.code, body.host);
        return sendJson(request, response, 200, { token, pairingId: pairing.id });
      } catch (error) {
        if (error instanceof PairingRefusedError) {
          options.log.warn("office.pairingRefused", { reason: error.reason });
          return sendJson(request, response, 403, { error: error.reason });
        }
        throw error;
      }
    }
    const pairing = await authenticate(request);
    if (method === "GET" && path === "/api/session") {
      return sendJson(request, response, 200, { host: pairing.host, folder: options.folderName });
    }
    if (method === "DELETE" && path === "/api/pair") {
      await options.pairings.remove(pairing.id);
      return sendJson(request, response, 204);
    }
    if (method === "POST" && path === "/api/word/review") {
      const { paragraphs } = await readJson(request);
      if (!Array.isArray(paragraphs) || !paragraphs.every((text) => typeof text === "string")) {
        throw new Refusal(400, "fields");
      }
      return sendJson(request, response, 200, { finding: nameSentence(paragraphs) });
    }
    if (method === "POST" && path === "/api/excel/review") {
      const { values, formulas } = await readJson(request);
      if (!isTable(values) || !isTable(formulas)) throw new Refusal(400, "fields");
      return sendJson(request, response, 200, { finding: nameCell(values, formulas) });
    }
    if (method === "GET" && path === "/api/events") {
      response.writeHead(200, {
        ...baseHeaders(request),
        "Content-Type": "text/event-stream",
        Connection: "keep-alive",
      });
      const write = (event: string, data: unknown) =>
        response.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
      write("ready", { host: pairing.host });
      const beat = setInterval(
        () => response.write(": keepalive\n\n"),
        options.heartbeatMs ?? HEARTBEAT_MS,
      );
      beat.unref();
      const stop = listen(pairing.id, {
        send: write,
        end: () => response.end(),
      });
      request.on("close", () => {
        clearInterval(beat);
        stop();
      });
      return;
    }
    throw new Refusal(404, "route");
  };

  const request = async (incoming: IncomingMessage, response: ServerResponse) => {
    try {
      checkWho(incoming);
      const method = incoming.method ?? "GET";
      const path = new URL(incoming.url ?? "/", "https://localhost").pathname;
      if (method === "OPTIONS") return preflight(incoming, response);
      if (path.startsWith("/api/")) return await api(incoming, response, method, path);
      if (method !== "GET") throw new Refusal(405, "method");
      const file = await options.file(path === "/" ? "/taskpane.html" : path);
      if (!file) throw new Refusal(404, "route");
      response.writeHead(200, {
        ...baseHeaders(incoming),
        "Content-Type": file.type,
        ...(file.type.startsWith("text/html") && {
          "Content-Security-Policy": options.contentSecurityPolicy,
        }),
      });
      response.end(file.body);
    } catch (error) {
      if (error instanceof Refusal) {
        options.log.warn("office.refused", { reason: error.reason });
        return sendJson(incoming, response, error.status, { error: error.reason });
      }
      options.log.error("office.requestFailed", {
        error: error instanceof Error ? error.name : "unknown",
      });
      sendJson(incoming, response, 500, { error: "internal" });
    }
  };

  const upgrade = (incoming: IncomingMessage, socket: Duplex, head: Buffer) => {
    const refuse = (status: number, reason: string) => {
      options.log.warn("office.refused", { reason });
      socket.end(`HTTP/1.1 ${status} Refused\r\nConnection: close\r\n\r\n`);
    };
    const host = incoming.headers.host?.toLowerCase();
    if (!host || !options.hosts.has(host)) return refuse(421, "host");
    const origin = incoming.headers.origin;
    if (origin === undefined || !options.origins.has(origin)) return refuse(403, "origin");
    if (
      new URL(incoming.url ?? "/", "https://localhost").pathname !== "/ws" ||
      !wantsWebSocket(incoming)
    ) {
      return refuse(404, "route");
    }
    const peer = acceptWebSocket(incoming, socket, head);
    let stop: (() => void) | null = null;
    const waiting = setTimeout(() => peer.close(1008), HELLO_WAIT_MS);
    waiting.unref();
    peer.onClose(() => {
      clearTimeout(waiting);
      stop?.();
    });
    peer.onText((text) => {
      void (async () => {
        let message: unknown;
        try {
          message = JSON.parse(text);
        } catch {
          return peer.close(1003);
        }
        if (!isRecord(message)) return peer.close(1003);
        if (message.type === "ping") return peer.send(JSON.stringify({ type: "pong" }));
        if (message.type !== "hello" || stop || typeof message.token !== "string") {
          return peer.close(1008);
        }
        const pairing = await options.pairings.authenticate(message.token);
        if (!pairing) return peer.close(1008);
        clearTimeout(waiting);
        stop = listen(pairing.id, {
          send: (event, data) => peer.send(JSON.stringify({ type: event, data })),
          end: () => peer.close(1000),
        });
        peer.send(JSON.stringify({ type: "ready", data: { host: pairing.host } }));
      })();
    });
  };

  return {
    request,
    upgrade,
    /** Tells the pane its pairing ended, and hangs up on it. */
    revoke(pairingId: string): void {
      for (const listener of listeners.get(pairingId) ?? []) {
        listener.send("revoked", {});
        listener.end();
      }
      listeners.delete(pairingId);
    },
    closeAll(): void {
      for (const set of listeners.values()) for (const listener of set) listener.end();
      listeners.clear();
    },
  };
}

export type OfficeHandler = ReturnType<typeof createOfficeHandler>;
