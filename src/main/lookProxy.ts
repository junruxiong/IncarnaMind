/**
 * The only way out of the hidden page that keeps a web page as it looks: an
 * HTTP proxy on this computer's loopback that connects only to the public
 * internet. Each request and tunnel is resolved here, with the guard web
 * search's fetcher uses (`publicAddresses`), and connected to an address it
 * checked, so a name that resolves, or rebinds, to this computer or the
 * local network is refused, as is any such address written as itself
 * (IPv4-mapped and NAT64 ones too), and a local name ("localhost.",
 * "*.local"). Only the page it serves can use it: every request must carry
 * this proxy's own secret. It serves one page's look, and is closed after it.
 */
import { randomBytes, timingSafeEqual } from "node:crypto";
import type { LookupAddress } from "node:dns";
import { createServer, type IncomingMessage, request, type ServerResponse } from "node:http";
import { connect, type Socket } from "node:net";
import type { Duplex } from "node:stream";
import type { Logger } from "../core";
import { publicAddresses, type Resolve } from "../core/tools/web/addresses";

const LOCAL_NAME = /^(?:localhost|.+\.localhost|.+\.local|.+\.internal|.+\.home\.arpa)$/i;

const HOP_BY_HOP = new Set([
  "connection",
  "keep-alive",
  "proxy-authenticate",
  "proxy-authorization",
  "proxy-connection",
  "te",
  "trailer",
  "transfer-encoding",
  "upgrade",
]);

export const LOOK_PROXY_USER = "look";

export interface LookProxyOptions {
  resolve: Resolve;
  log: Logger;
  /**
   * Test builds: the port of the local server standing in for the web. Each
   * checked address is this computer's instead, in the same family.
   */
  standIn?: number;
}

export interface LookProxy {
  /** "127.0.0.1:<port>", for `session.setProxy`. */
  address: string;
  host: string;
  port: number;
  /** What the page answers the proxy's sign-in with, as `LOOK_PROXY_USER`. */
  secret: string;
  close(): Promise<void>;
}

const bareHost = (host: string) =>
  host
    .replace(/^\[(.*)\]$/, "$1")
    .replace(/\.+$/, "")
    .toLowerCase();

const standInFor = ({ family }: LookupAddress): LookupAddress => ({
  address: family === 6 ? "::1" : "127.0.0.1",
  family,
});

export async function startLookProxy(options: LookProxyOptions): Promise<LookProxy> {
  const secret = randomBytes(24).toString("base64url");
  const expected = Buffer.from(
    `Basic ${Buffer.from(`${LOOK_PROXY_USER}:${secret}`).toString("base64")}`,
  );
  const signedIn = (incoming: IncomingMessage) => {
    const given = Buffer.from(incoming.headers["proxy-authorization"] ?? "");
    return given.length === expected.length && timingSafeEqual(given, expected);
  };
  let closed = false;
  const sockets = new Set<Duplex>();
  const track = (socket: Duplex) => {
    sockets.add(socket);
    socket.once("close", () => sockets.delete(socket));
  };

  function connectTo(addresses: readonly LookupAddress[], port: number): Promise<Socket> {
    return new Promise((resolve, reject) => {
      const socket = connect({
        // Any name: net.connect skips `lookup`, which gives it the checked addresses, for an IP.
        host: "checked.invalid",
        port,
        lookup: (_name, _options, callback) => callback(null, [...addresses]),
        // Moves on from an address that doesn't answer, rather than spending the page's time on it.
        autoSelectFamily: true,
      });
      track(socket);
      const stopped = () => reject(new Error("The proxy is closed."));
      socket.once("error", reject);
      socket.once("close", stopped);
      socket.once("connect", () => {
        socket.off("error", reject);
        socket.off("close", stopped);
        resolve(socket);
      });
    });
  }

  async function open(host: string, port: number): Promise<Socket> {
    const name = bareHost(host);
    if (!name || LOCAL_NAME.test(name) || !Number.isInteger(port) || port <= 0 || port > 65_535) {
      throw new Error(`${host} isn't on the public internet.`);
    }
    const checked = await publicAddresses(name, options.resolve);
    if (closed) throw new Error("The proxy is closed.");
    return options.standIn
      ? connectTo(checked.map(standInFor), options.standIn)
      : connectTo(checked, port);
  }
  const askToSignIn = { "Proxy-Authenticate": `Basic realm="${LOOK_PROXY_USER}"` };

  const server = createServer((incoming: IncomingMessage, outgoing: ServerResponse) => {
    if (!signedIn(incoming)) {
      outgoing.writeHead(407, askToSignIn).end();
      return;
    }
    const url = URL.parse(incoming.url ?? "");
    if (url?.protocol !== "http:") {
      outgoing.writeHead(400).end();
      return;
    }
    open(url.hostname, Number(url.port || 80)).then(
      (socket) => {
        const headers = Object.fromEntries(
          Object.entries(incoming.headers).filter(([name]) => !HOP_BY_HOP.has(name)),
        );
        const forwarded = request(
          {
            createConnection: () => socket,
            method: incoming.method,
            path: `${url.pathname}${url.search}`,
            headers: { ...headers, host: url.host, connection: "close" },
          },
          (answer) => {
            const kept = Object.fromEntries(
              Object.entries(answer.headers).filter(([name]) => !HOP_BY_HOP.has(name)),
            );
            outgoing.writeHead(answer.statusCode ?? 502, kept);
            answer.pipe(outgoing);
          },
        );
        forwarded.on("error", () => outgoing.destroy());
        incoming.pipe(forwarded);
      },
      () => outgoing.writeHead(403).end(),
    );
  });

  server.on("connect", (incoming: IncomingMessage, client: Duplex, head: Buffer) => {
    track(client);
    client.on("error", () => client.destroy());
    if (!signedIn(incoming)) {
      const challenge = Object.entries(askToSignIn)
        .map(([name, value]) => `${name}: ${value}\r\n`)
        .join("");
      client.end(`HTTP/1.1 407 Proxy Authentication Required\r\n${challenge}\r\n`);
      return;
    }
    const at = (incoming.url ?? "").lastIndexOf(":");
    const host = at > 0 ? (incoming.url ?? "").slice(0, at) : "";
    const port = Number((incoming.url ?? "").slice(at + 1));
    open(host, port).then(
      (socket) => {
        socket.on("error", () => client.destroy());
        client.write("HTTP/1.1 200 Connection Established\r\n\r\n");
        if (head.length > 0) socket.write(head);
        socket.pipe(client);
        client.pipe(socket);
      },
      () => client.end("HTTP/1.1 403 Forbidden\r\n\r\n"),
    );
  });
  server.on("error", (error) =>
    options.log.error("webPages.lookProxy", { message: error.message }),
  );

  await new Promise<void>((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", () => {
      server.off("error", reject);
      resolve();
    });
  });
  const listening = server.address();
  if (!listening || typeof listening === "string") throw new Error("The proxy isn't listening.");
  return {
    address: `127.0.0.1:${listening.port}`,
    host: "127.0.0.1",
    port: listening.port,
    secret,
    close: () =>
      new Promise((resolve) => {
        closed = true;
        for (const socket of sockets) socket.destroy();
        server.closeAllConnections();
        server.close(() => resolve());
      }),
  };
}
