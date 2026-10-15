/**
 * The proxy the hidden page that keeps a web page as it looks reaches the
 * network through: only with its own secret, so no other program on this
 * computer can use it, and only to the public internet.
 */
import { once } from "node:events";
import { createServer, type IncomingHttpHeaders, request, Server } from "node:http";
import { connect } from "node:net";
import { describe, expect, onTestFinished, test, vi } from "vitest";
import type { Resolve } from "../../src/core/tools/web/addresses";
import { LOOK_PROXY_USER, type LookProxy, startLookProxy } from "../../src/main/lookProxy";

const log = { info: () => undefined, warn: () => undefined, error: () => undefined };

const publicResolve: Resolve = async (host) =>
  host === "rebind.example"
    ? [{ address: "127.0.0.1", family: 4 }]
    : [{ address: "93.184.215.14", family: 4 }];

let site: Server;
let sitePort = 0;
let connections = 0;
const heard: string[] = [];
const heardHeaders: IncomingHttpHeaders[] = [];

/** The site, standing in for every public address, on this computer's IPv4 address only. */
async function start(resolve: Resolve = publicResolve): Promise<LookProxy> {
  heard.length = 0;
  heardHeaders.length = 0;
  connections = 0;
  site = createServer((incoming, outgoing) => {
    heard.push(`${incoming.headers.host}${incoming.url}`);
    heardHeaders.push(incoming.headers);
    outgoing.end("A page.");
  });
  site.on("connection", () => connections++);
  await new Promise<void>((listening) => site.listen(0, "127.0.0.1", listening));
  const address = site.address();
  if (!address || typeof address === "string") throw new Error("No port.");
  sitePort = address.port;
  const proxy = await startLookProxy({ resolve, log, standIn: sitePort });
  const closing = site;
  onTestFinished(async () => {
    await proxy.close();
    closing.closeAllConnections();
    await new Promise((closed) => closing.close(closed));
  });
  return proxy;
}

/** The connections the site has had, made before one made now. */
async function connectionsBefore(): Promise<number> {
  // A server takes connections in the order they were made: once this one is in, any before it are.
  const accepted = once(site, "connection");
  const control = connect(sitePort, "127.0.0.1");
  await accepted;
  control.destroy();
  return connections - 1;
}

const signIn = (secret: string) =>
  `Basic ${Buffer.from(`${LOOK_PROXY_USER}:${secret}`).toString("base64")}`;

/** An HTTP request through the proxy: its status and the challenge it answers with. */
function through(
  open: LookProxy,
  url: string,
  authorization?: string,
): Promise<{ status: number; challenge: string | undefined }> {
  return new Promise((resolve, reject) => {
    const asked = request(
      {
        host: open.host,
        port: open.port,
        path: url,
        headers: authorization ? { "proxy-authorization": authorization } : {},
      },
      (answer) => {
        answer.resume();
        resolve({
          status: answer.statusCode ?? 0,
          challenge: answer.headers["proxy-authenticate"],
        });
      },
    );
    asked.on("error", reject);
    asked.end();
  });
}

/** A tunnel asked of the proxy: the status line it answers with. */
function tunnel(open: LookProxy, authority: string, authorization?: string): Promise<string> {
  return new Promise((resolve, reject) => {
    const socket = connect(open.port, open.host, () => {
      socket.write(
        `CONNECT ${authority} HTTP/1.1\r\nHost: ${authority}\r\n${
          authorization ? `Proxy-Authorization: ${authorization}\r\n` : ""
        }\r\n`,
      );
    });
    socket.once("data", (data) => {
      resolve(data.toString("latin1").split("\r\n")[0] ?? "");
      socket.destroy();
    });
    socket.on("error", reject);
  });
}

describe("the proxy of the hidden page that keeps a web page as it looks", () => {
  test("without its secret, a request or a tunnel is asked to sign in, and nothing is fetched", async () => {
    const open = await start();
    expect(await through(open, "http://news.example/page")).toEqual({
      status: 407,
      challenge: `Basic realm="${LOOK_PROXY_USER}"`,
    });
    expect((await through(open, "http://news.example/page", signIn("guessed"))).status).toBe(407);
    expect(await tunnel(open, "news.example:443")).toMatch(/^HTTP\/1\.1 407 /);
    expect(heard).toEqual([]);
  });

  test("with it, a public host is reached; this computer, by name or by address, is not", async () => {
    const open = await start();
    const secret = signIn(open.secret);
    expect((await through(open, "http://news.example/page", secret)).status).toBe(200);
    expect(heard).toEqual(["news.example/page"]);
    for (const url of [
      "http://localhost./page",
      "http://printer.local../page",
      "http://[::ffff:7f00:1]/page",
      "http://[64:ff9b::7f00:1]/page",
      "http://100.64.0.1/page",
      "http://rebind.example/page",
    ]) {
      expect((await through(open, url, secret)).status, url).toBe(403);
    }
    expect(await tunnel(open, "localhost.:443", secret)).toMatch(/^HTTP\/1\.1 403 /);
    expect(await tunnel(open, "news.example:443", secret)).toMatch(/^HTTP\/1\.1 200 /);
    expect(heard).toEqual(["news.example/page"]);
  });

  test("the site never sees the proxy's secret", async () => {
    const open = await start();

    expect((await through(open, "http://news.example/page", signIn(open.secret))).status).toBe(200);

    expect(heardHeaders).toHaveLength(1);
    expect(heardHeaders[0]).not.toHaveProperty("proxy-authorization");
    expect(JSON.stringify(heardHeaders[0])).not.toContain(open.secret);
  });

  test("a host's next checked address is reached when the first can't be", async () => {
    // The IPv6 address stands in as this computer's, where the site doesn't listen.
    const open = await start(async () => [
      { address: "2606:2800:21f:cb07:6820:80da:af6b:8b2c", family: 6 },
      { address: "93.184.215.14", family: 4 },
    ]);

    expect((await through(open, "http://two.example/page", signIn(open.secret))).status).toBe(200);
    expect(heard).toEqual(["two.example/page"]);
  });

  test("once it is closed, a request it was looking up a name for goes nowhere", async () => {
    let looking: () => void = () => undefined;
    const lookingUp = new Promise<void>((resolve) => {
      looking = resolve;
    });
    let answer: () => void = () => undefined;
    const answered = new Promise<void>((resolve) => {
      answer = resolve;
    });
    const open = await start(async (host) => {
      looking();
      await answered;
      return publicResolve(host);
    });
    const asked = through(open, "http://slow.example/page", signIn(open.secret)).catch(
      () => "closed",
    );
    await lookingUp;

    await open.close();
    answer();

    expect(await asked).toBe("closed");
    await new Promise((resolve) => setImmediate(resolve));
    expect(await connectionsBefore()).toBe(0);
    expect(heard).toEqual([]);
  });

  test("one that can't listen says so, rather than never starting", async () => {
    const listen = vi.spyOn(Server.prototype, "listen").mockImplementationOnce(function (
      this: Server,
    ) {
      process.nextTick(() =>
        this.emit("error", Object.assign(new Error("Too many open files."), { code: "EMFILE" })),
      );
      return this;
    });
    onTestFinished(() => listen.mockRestore());

    await expect(startLookProxy({ resolve: publicResolve, log })).rejects.toThrow(
      "Too many open files.",
    );
  });
});
