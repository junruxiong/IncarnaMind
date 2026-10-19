import { randomBytes } from "node:crypto";
import { mkdir, mkdtemp, readFile, writeFile } from "node:fs/promises";
import { request as httpsRequest } from "node:https";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { TLSSocket } from "node:tls";
import { createOfficeBridge, type OfficeBridgeAddress } from "../../src/core/office/index";

export function memorySecrets() {
  const kept = new Map<string, string>();
  return {
    kept,
    set: async (name: string, secret: string) => void kept.set(name, secret),
    tryGet: async (name: string) => kept.get(name) ?? null,
    delete: async (name: string) => void kept.delete(name),
  };
}

export interface RunningBridge {
  address: OfficeBridgeAddress;
  authority: string;
  secrets: ReturnType<typeof memorySecrets>;
  bridge: ReturnType<typeof createOfficeBridge>;
  clock: { now: number };
}

export async function startBridge(
  options: {
    extraOrigins?: string[];
    secrets?: ReturnType<typeof memorySecrets>;
    staticDir?: string;
  } = {},
): Promise<RunningBridge> {
  const folder = await mkdtemp(join(tmpdir(), "office-bridge-"));
  const staticDir = options.staticDir ?? join(folder, "pages");
  if (!options.staticDir) {
    await mkdir(join(staticDir, "assets"), { recursive: true });
    await writeFile(join(staticDir, "taskpane.html"), "<!doctype html><title>pane</title>");
    await writeFile(join(staticDir, "taskpane.js"), "// pane");
  }
  const secrets = options.secrets ?? memorySecrets();
  const clock = { now: 1_000 };
  const bridge = createOfficeBridge({
    adapter: {
      port: 0,
      staticDir,
      authorityFile: join(folder, "authority.pem"),
      folderName: "Tides",
      ...(options.extraOrigins && { extraOrigins: options.extraOrigins }),
    },
    log: { info: () => undefined, warn: () => undefined, error: () => undefined },
    secrets,
    clock: () => clock.now,
    now: () => new Date(clock.now).toISOString(),
  });
  const address = await bridge.start();
  return {
    address,
    authority: await readFile(address.authorityFile, "utf8"),
    secrets,
    bridge,
    clock,
  };
}

export interface Reply {
  status: number;
  headers: Record<string, string | string[] | undefined>;
  text: string;
  json: unknown;
}

export function call(
  running: RunningBridge,
  request: {
    method?: string;
    path: string;
    headers?: Record<string, string>;
    body?: unknown;
    host?: string;
    address?: string;
  },
): Promise<Reply> {
  const { port } = running.address;
  const body =
    request.body === undefined
      ? undefined
      : typeof request.body === "string"
        ? request.body
        : JSON.stringify(request.body);
  return new Promise((done, fail) => {
    const outgoing = httpsRequest(
      {
        host: request.address ?? "127.0.0.1",
        port,
        ca: running.authority,
        servername: "localhost",
        method: request.method ?? "GET",
        path: request.path,
        headers: {
          Host: request.host ?? `127.0.0.1:${port}`,
          ...(body !== undefined && { "Content-Type": "application/json" }),
          ...request.headers,
        },
      },
      (response) => {
        const chunks: Buffer[] = [];
        response.on("data", (chunk: Buffer) => chunks.push(chunk));
        response.on("end", () => {
          const text = Buffer.concat(chunks).toString("utf8");
          let json: unknown = null;
          try {
            json = JSON.parse(text);
          } catch {
            json = null;
          }
          done({ status: response.statusCode ?? 0, headers: response.headers, text, json });
        });
      },
    );
    outgoing.on("error", fail);
    outgoing.end(body);
  });
}

export const own = (running: RunningBridge) => `https://127.0.0.1:${running.address.port}`;

export async function pair(
  running: RunningBridge,
  host: "word" | "excel" = "word",
): Promise<string> {
  const { code } = running.bridge.beginPairing();
  const reply = await call(running, {
    method: "POST",
    path: "/api/pair",
    headers: { Origin: own(running) },
    body: { code, host },
  });
  if (reply.status !== 200) throw new Error(`Pairing failed: ${reply.text}`);
  return (reply.json as { token: string }).token;
}

function clientFrame(text: string): Buffer {
  const payload = Buffer.from(text, "utf8");
  const mask = randomBytes(4);
  const masked = Buffer.from(payload.map((byte, index) => byte ^ (mask[index % 4] ?? 0)));
  return Buffer.concat([Buffer.from([0x81, 0x80 | payload.length]), mask, masked]);
}

export interface ClientSocket {
  status: number;
  send(message: unknown): void;
  next(): Promise<unknown>;
  closed: Promise<void>;
  end(): void;
}

export function openSocket(
  running: RunningBridge,
  headers: Record<string, string>,
): Promise<ClientSocket> {
  const { port } = running.address;
  return new Promise((done, fail) => {
    const outgoing = httpsRequest({
      host: "127.0.0.1",
      port,
      ca: running.authority,
      servername: "localhost",
      path: "/ws",
      headers: {
        Host: `127.0.0.1:${port}`,
        Connection: "Upgrade",
        Upgrade: "websocket",
        "Sec-WebSocket-Version": "13",
        "Sec-WebSocket-Key": randomBytes(16).toString("base64"),
        ...headers,
      },
    });
    const refused = (status: number) =>
      done({
        status,
        send: () => undefined,
        next: () => Promise.reject(new Error("refused")),
        closed: Promise.resolve(),
        end: () => undefined,
      });
    outgoing.on("response", (response) => {
      response.resume();
      refused(response.statusCode ?? 0);
    });
    outgoing.on("error", fail);
    outgoing.on("upgrade", (response, socket: TLSSocket) => {
      let buffered = Buffer.alloc(0);
      const waiting: ((message: unknown) => void)[] = [];
      const ready: unknown[] = [];
      const closed = new Promise<void>((finished) => socket.once("close", () => finished()));
      socket.on("data", (chunk: Buffer) => {
        buffered = Buffer.concat([buffered, chunk]);
        while (buffered.length >= 2) {
          const opcode = (buffered[0] ?? 0) & 0x0f;
          const size = (buffered[1] ?? 0) & 0x7f;
          if (buffered.length < 2 + size) return;
          const payload = buffered.subarray(2, 2 + size);
          buffered = buffered.subarray(2 + size);
          if (opcode !== 1) continue;
          const message = JSON.parse(payload.toString("utf8")) as unknown;
          const listener = waiting.shift();
          if (listener) listener(message);
          else ready.push(message);
        }
      });
      done({
        status: response.statusCode ?? 0,
        send: (message) => void socket.write(clientFrame(JSON.stringify(message))),
        next: () =>
          ready.length > 0
            ? Promise.resolve(ready.shift())
            : new Promise((resolve) => waiting.push(resolve)),
        closed,
        end: () => socket.destroy(),
      });
    });
    outgoing.end();
  });
}
