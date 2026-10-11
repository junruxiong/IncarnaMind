/**
 * Where outside agents' shims reach the running app (#351): a Unix domain
 * socket in the data folder (a named pipe on Windows), as the MCP security
 * guide advises for a local server, rather than an HTTP port a web page could
 * try. On macOS and Linux only the User's account can connect to it: the
 * socket file is made readable and writable by its owner only. The core runs
 * in the main process, so this listener does too; it needs only Node.
 */
import { createHash } from "node:crypto";
import { chmod, unlink } from "node:fs/promises";
import { createConnection, createServer, type Server, type Socket } from "node:net";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

/** The longest socket path every platform takes: macOS's limit is 104 bytes, Linux's 108. */
const MAX_SOCKET_PATH_BYTES = 100;

/**
 * The socket for a data folder: in it, or, when that path is too long for a
 * socket, in the temporary folder under a name made from the data folder's path.
 */
export function agentSocketPath(dataDir: string, platform: NodeJS.Platform = process.platform) {
  const id = createHash("sha256").update(resolve(dataDir)).digest("hex").slice(0, 16);
  if (platform === "win32") return `\\\\.\\pipe\\incarnamind-agents-${id}`;
  const inDataFolder = join(dataDir, "agents.sock");
  return Buffer.byteLength(inDataFolder) <= MAX_SOCKET_PATH_BYTES
    ? inDataFolder
    : join(tmpdir(), `incarnamind-${id}.sock`);
}

export interface AgentListener {
  readonly path: string;
  close(): Promise<void>;
}

/** Whether something is listening on the socket now. */
function answers(path: string): Promise<boolean> {
  return new Promise((done) => {
    const probe = createConnection(path);
    probe.once("connect", () => {
      probe.destroy();
      done(true);
    });
    probe.once("error", () => done(false));
  });
}

function listenOn(server: Server, path: string): Promise<void> {
  return new Promise((done, fail) => {
    const onError = (error: Error) => {
      server.off("listening", onListening);
      fail(error);
    };
    const onListening = () => {
      server.off("error", onError);
      done();
    };
    server.once("error", onError);
    server.once("listening", onListening);
    server.listen(path);
  });
}

/**
 * Listens on `path`, handing each connection to `onConnection`. A socket file
 * left by an app that quit without closing it is replaced; one another copy
 * of IncarnaMind is listening on is left alone, and this rejects.
 */
export async function listenForAgents(
  path: string,
  onConnection: (socket: Socket) => void,
): Promise<AgentListener> {
  const sockets = new Set<Socket>();
  const server = createServer((socket) => {
    sockets.add(socket);
    socket.once("close", () => sockets.delete(socket));
    onConnection(socket);
  });
  const windows = path.startsWith("\\\\.\\pipe\\");
  try {
    await listenOn(server, path);
  } catch (error) {
    const code = (error as NodeJS.ErrnoException).code;
    if (windows || code !== "EADDRINUSE" || (await answers(path))) throw error;
    await unlink(path);
    await listenOn(server, path);
  }
  if (!windows) await chmod(path, 0o600);
  return {
    path,
    close: () =>
      new Promise<void>((done) => {
        for (const socket of sockets) socket.destroy();
        server.close(() => done());
      }),
  };
}
