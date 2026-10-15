/**
 * The proxy the hidden page that keeps a web page as it looks reaches the
 * network through: only with its own secret, so no other program on this
 * computer can use it, and only to the public internet.
 */
import { createServer, request, type Server } from "node:http";
import { connect } from "node:net";
import { afterEach, describe, expect, test } from "vitest";
import { LOOK_PROXY_USER, type LookProxy, startLookProxy } from "../../src/main/lookProxy";

const log = { info: () => undefined, warn: () => undefined, error: () => undefined };

let site: Server;
let proxy: LookProxy | undefined;
const heard: string[] = [];

async function start(): Promise<LookProxy> {
  heard.length = 0;
  site = createServer((incoming, outgoing) => {
    heard.push(`${incoming.headers.host}${incoming.url}`);
    outgoing.end("A page.");
  });
  await new Promise<void>((resolve) => site.listen(0, "127.0.0.1", resolve));
  const address = site.address();
  if (!address || typeof address === "string") throw new Error("No port.");
  proxy = await startLookProxy({
    resolve: async (host) =>
      host === "rebind.example"
        ? [{ address: "127.0.0.1", family: 4 }]
        : [{ address: "93.184.215.14", family: 4 }],
    log,
    standIn: address.port,
  });
  return proxy;
}

afterEach(async () => {
  await proxy?.close();
  await new Promise((resolve) => site.close(resolve));
});

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
});
