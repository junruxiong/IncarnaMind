/**
 * The only way out of the window that keeps a web page as it looks: an HTTP
 * proxy on this computer's loopback that connects only to the public
 * internet. Each request and tunnel is resolved here, with the guard web
 * search's fetcher uses (`publicAddresses`), and connected to the address it
 * checked, so a name that resolves, or rebinds, to this computer or the
 * local network is refused, as is any such address written as itself
 * (IPv4-mapped and NAT64 ones too), and a local name ("localhost.",
 * "*.local"). It serves one page's look, and is closed after it.
 */
import { createServer, type IncomingMessage, request, type ServerResponse } from "node:http";
import { connect, type Socket } from "node:net";
import type { Duplex } from "node:stream";
import { publicAddresses, type Resolve } from "../core/tools/web/addresses";

const LOCAL_NAME = /^(?:localhost|.+\.localhost|.+\.local|.+\.internal|.+\.home\.arpa)$/i;

/** Headers for the proxy alone, not for the site. */
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

export interface LookProxyOptions {
  resolve: Resolve;
  /** Test builds: where every checked connection goes, the local server standing in for the web. */
  standIn?: number;
}

export interface LookProxy {
  /** "127.0.0.1:<port>", for `session.setProxy`. */
  address: string;
  close(): Promise<void>;
}

/** A host as written in a URL or a CONNECT line, without brackets or a trailing dot. */
const bareHost = (host: string) =>
  host
    .replace(/^\[(.*)\]$/, "$1")
    .replace(/\.$/, "")
    .toLowerCase();

export async function startLookProxy(options: LookProxyOptions): Promise<LookProxy> {
  /** Connects to `host`, if it is on the public internet, at the address the guard checked. */
  async function open(host: string, port: number): Promise<Socket> {
    const name = bareHost(host);
    if (!name || LOCAL_NAME.test(name) || !Number.isInteger(port) || port <= 0 || port > 65_535) {
      throw new Error(`${host} isn't on the public internet.`);
    }
    const [checked] = await publicAddresses(name, options.resolve);
    if (!checked) throw new Error(`${host} has no address.`);
    const target = options.standIn
      ? { host: "127.0.0.1", port: options.standIn }
      : { host: checked.address, port };
    return new Promise((resolve, reject) => {
      const socket = connect(target);
      socket.once("connect", () => {
        socket.off("error", reject);
        resolve(socket);
      });
      socket.once("error", reject);
    });
  }

  const sockets = new Set<Duplex>();
  const track = (socket: Duplex) => {
    sockets.add(socket);
    socket.once("close", () => sockets.delete(socket));
  };

  const server = createServer((incoming: IncomingMessage, outgoing: ServerResponse) => {
    const url = URL.parse(incoming.url ?? "");
    if (!url || url.protocol !== "http:") {
      outgoing.writeHead(400).end();
      return;
    }
    const port = Number(url.port || 80);
    open(url.hostname, port).then(
      (socket) => {
        track(socket);
        const headers = Object.fromEntries(
          Object.entries(incoming.headers).filter(([name]) => !HOP_BY_HOP.has(name)),
        );
        const forwarded = request(
          {
            createConnection: () => socket,
            method: incoming.method,
            path: `${url.pathname}${url.search}`,
            headers: { ...headers, connection: "close" },
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
    const at = (incoming.url ?? "").lastIndexOf(":");
    const host = at > 0 ? (incoming.url ?? "").slice(0, at) : "";
    const port = Number((incoming.url ?? "").slice(at + 1));
    open(host, port).then(
      (socket) => {
        track(socket);
        socket.on("error", () => client.destroy());
        client.write("HTTP/1.1 200 Connection Established\r\n\r\n");
        if (head.length > 0) socket.write(head);
        socket.pipe(client);
        client.pipe(socket);
      },
      () => client.end("HTTP/1.1 403 Forbidden\r\n\r\n"),
    );
  });

  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const listening = server.address();
  if (!listening || typeof listening === "string") throw new Error("The proxy isn't listening.");
  return {
    address: `127.0.0.1:${listening.port}`,
    close: () =>
      new Promise((resolve) => {
        for (const socket of sockets) socket.destroy();
        server.closeAllConnections();
        server.close(() => resolve());
      }),
  };
}
