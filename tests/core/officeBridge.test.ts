/**
 * The Office add-in bridge (spike for #1272): another program on this
 * computer, a web page in the User's browser and a DNS-rebinding page can't
 * use what an add-in pairs for, and the certificate the app makes for itself
 * vouches for loopback names only.
 */
import { X509Certificate } from "node:crypto";
import { existsSync } from "node:fs";
import { mkdir, mkdtemp, writeFile } from "node:fs/promises";
import { request as httpsRequest } from "node:https";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { createAuthority, issueLoopbackCertificate } from "../../src/core/office/certificate";
import { createOfficeBridge } from "../../src/core/office/index";
import {
  CODE_LIFETIME_MS,
  createPairings,
  PairingRefusedError,
  WRONG_TRIES_ALLOWED,
} from "../../src/core/office/pairing";
import {
  call,
  memorySecrets,
  openSocket,
  own,
  pair,
  type RunningBridge,
  startBridge,
} from "../helpers/officeBridge";

const running: RunningBridge[] = [];
afterEach(async () => {
  await Promise.all(running.splice(0).map((each) => each.bridge.close()));
});

async function start(options?: Parameters<typeof startBridge>[0]): Promise<RunningBridge> {
  const started = await startBridge(options);
  running.push(started);
  return started;
}

function handshake(
  authority: string,
  port: number,
  host: string,
  servername?: string,
): Promise<number> {
  return new Promise((done, fail) => {
    const outgoing = httpsRequest(
      {
        host,
        port,
        ca: authority,
        path: "/api/ping",
        ...(servername && { servername }),
        headers: { Host: `127.0.0.1:${port}` },
      },
      (response) => {
        response.resume();
        done(response.statusCode ?? 0);
      },
    );
    outgoing.on("error", fail);
    outgoing.end();
  });
}

describe("the certificate the app makes for itself", () => {
  const now = new Date("2026-10-19T10:00:00Z");

  it("is signed by an authority that may only vouch for loopback names", () => {
    const authority = createAuthority(now);
    const leaf = issueLoopbackCertificate(authority, now);
    const authorityCert = new X509Certificate(authority.certificatePem);
    const leafCert = new X509Certificate(leaf.certificatePem);

    expect(authorityCert.ca).toBe(true);
    expect(leafCert.ca).toBe(false);
    expect(leafCert.checkIssued(authorityCert)).toBe(true);
    expect(leafCert.verify(authorityCert.publicKey)).toBe(true);
    expect(leafCert.checkHost("localhost")).toBe("localhost");
    expect(leafCert.checkIP("127.0.0.1")).toBe("127.0.0.1");
    expect(leafCert.checkIP("::1")).toBeDefined();
    expect(leafCert.checkHost("example.com")).toBeUndefined();
    expect(new Date(leafCert.validTo).getTime() - now.getTime()).toBeLessThanOrEqual(
      91 * 86_400_000,
    );
  });

  it("is accepted by a TLS client that trusts only the authority, on every loopback name", async () => {
    const live = await start();
    const { port, addresses } = live.address;
    expect(await handshake(live.authority, port, "127.0.0.1")).toBe(200);
    expect(await handshake(live.authority, port, "127.0.0.1", "localhost")).toBe(200);
    if (addresses.includes("::1")) {
      expect(await handshake(live.authority, port, "::1")).toBe(200);
    }
  });

  it("is refused for a name the authority may not vouch for", async () => {
    const authority = createAuthority(now);
    const other = issueLoopbackCertificate(authority, now, {
      dns: ["bank.example"],
      addresses: ["127.0.0.1"],
    });
    const { createServer } = await import("node:https");
    const server = createServer({
      key: other.keyPem,
      cert: `${other.certificatePem}${authority.certificatePem}`,
    });
    await new Promise<void>((done) => server.listen(0, "127.0.0.1", done));
    const port = (server.address() as { port: number }).port;
    try {
      await expect(
        new Promise((done, fail) => {
          const outgoing = httpsRequest(
            { host: "127.0.0.1", port, ca: authority.certificatePem, servername: "bank.example" },
            (response) => {
              response.resume();
              done(response.statusCode);
            },
          );
          outgoing.on("error", fail);
          outgoing.end();
        }),
      ).rejects.toThrow(/permitted subtree violation/);
    } finally {
      server.close();
    }
  });
});

describe("pairing", () => {
  function pairings(clock: { now: number }) {
    const secrets = memorySecrets();
    const removed: string[] = [];
    const made = createPairings({
      secrets,
      clock: () => clock.now,
      now: () => new Date(clock.now).toISOString(),
      removed: (id) => removed.push(id),
    });
    return { made, secrets, removed };
  }

  const refusal = async (attempt: Promise<unknown>) =>
    attempt.then(
      () => null,
      (error: unknown) => (error instanceof PairingRefusedError ? error.reason : "other"),
    );

  it("swaps the code for a token once, and keeps only the token's hash", async () => {
    const clock = { now: 0 };
    const { made, secrets } = pairings(clock);
    const { code } = made.start();
    const { token } = await made.redeem(code, "word");

    expect((await made.authenticate(token))?.host).toBe("word");
    expect(await refusal(made.redeem(code, "word"))).toBe("no-code");
    expect([...secrets.kept.values()].join("")).not.toContain(token);
  });

  it("gives up on the code after three wrong tries, even for the right one", async () => {
    const { made } = pairings({ now: 0 });
    const { code } = made.start();
    const wrong = code === "000000" ? "111111" : "000000";
    for (let attempt = 0; attempt < WRONG_TRIES_ALLOWED; attempt += 1) {
      expect(await refusal(made.redeem(wrong, "word"))).toBe("wrong");
    }
    expect(await refusal(made.redeem(code, "word"))).toBe("no-code");
  });

  it("expires the code, and a new code replaces the one on screen", async () => {
    const clock = { now: 0 };
    const { made } = pairings(clock);
    const first = made.start();
    clock.now = CODE_LIFETIME_MS;
    expect(await refusal(made.redeem(first.code, "word"))).toBe("expired");

    const older = made.start();
    const newer = made.start();
    if (older.code !== newer.code) {
      expect(await refusal(made.redeem(older.code, "word"))).toBe("wrong");
    }
  });

  it("forgets a token when it is removed, and says which pairing went", async () => {
    const { made, removed } = pairings({ now: 0 });
    const { token, pairing } = await made.redeem(made.start().code, "excel");
    await made.remove(pairing.id);

    expect(await made.authenticate(token)).toBeNull();
    expect(removed).toEqual([pairing.id]);
  });

  it("keeps a pairing across a restart", async () => {
    const live = await start();
    const token = await pair(live);
    await live.bridge.close();
    running.length = 0;

    const again = await start({ secrets: live.secrets });
    const reply = await call(again, {
      path: "/api/session",
      headers: { Authorization: `Bearer ${token}` },
    });
    expect(reply.status).toBe(200);
    expect(reply.json).toEqual({ host: "word", folder: "Tides" });
  });
});

describe("a request that isn't from a paired pane is refused", () => {
  it("has no token", async () => {
    const live = await start();
    for (const path of ["/api/session", "/api/events"]) {
      expect((await call(live, { path })).status).toBe(401);
    }
    for (const path of ["/api/word/review", "/api/excel/review"]) {
      expect((await call(live, { method: "POST", path, body: {} })).status).toBe(401);
    }
  });

  it("has a wrong token", async () => {
    const live = await start();
    await pair(live);
    const reply = await call(live, {
      path: "/api/session",
      headers: { Authorization: "Bearer not-the-token" },
    });
    expect(reply.status).toBe(401);
  });

  it("comes from a page on another origin, even with the right token", async () => {
    const live = await start();
    const token = await pair(live);
    const reply = await call(live, {
      method: "POST",
      path: "/api/word/review",
      headers: { Authorization: `Bearer ${token}`, Origin: "https://evil.example" },
      body: { paragraphs: ["The loan is 12 million."] },
    });
    expect(reply.status).toBe(403);
    expect(reply.headers["access-control-allow-origin"]).toBeUndefined();
  });

  it("is a preflight from another origin", async () => {
    const live = await start();
    const reply = await call(live, {
      method: "OPTIONS",
      path: "/api/word/review",
      headers: {
        Origin: "https://evil.example",
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "authorization, content-type",
      },
    });
    expect(reply.status).toBe(403);
  });

  it("names a Host that isn't a loopback address, as a DNS-rebinding page does", async () => {
    const live = await start();
    const token = await pair(live);
    for (const host of ["evil.example", `evil.example:${live.address.port}`, "localhost:1"]) {
      const reply = await call(live, {
        path: "/api/session",
        host,
        headers: { Authorization: `Bearer ${token}` },
      });
      expect(reply.status).toBe(421);
    }
  });

  it("pairs without an Origin, which only a program that isn't a page sends", async () => {
    const live = await start();
    const { code } = live.bridge.beginPairing();
    const reply = await call(live, {
      method: "POST",
      path: "/api/pair",
      body: { code, host: "word" },
    });
    expect(reply.status).toBe(403);
    expect(await live.bridge.pairings()).toEqual([]);
  });

  it("guesses the code: three wrong ones end it", async () => {
    const live = await start();
    const { code } = live.bridge.beginPairing();
    const wrong = code === "000000" ? "111111" : "000000";
    for (let attempt = 0; attempt < WRONG_TRIES_ALLOWED; attempt += 1) {
      const reply = await call(live, {
        method: "POST",
        path: "/api/pair",
        headers: { Origin: own(live) },
        body: { code: wrong, host: "word" },
      });
      expect(reply.status).toBe(403);
    }
    const right = await call(live, {
      method: "POST",
      path: "/api/pair",
      headers: { Origin: own(live) },
      body: { code, host: "word" },
    });
    expect(right.status).toBe(403);
    expect(await live.bridge.pairings()).toEqual([]);
  });

  it("posts something a page could send without asking first", async () => {
    const live = await start();
    const token = await pair(live);
    const reply = await call(live, {
      method: "POST",
      path: "/api/word/review",
      headers: { Authorization: `Bearer ${token}`, "Content-Type": "text/plain" },
      body: JSON.stringify({ paragraphs: [] }),
    });
    expect(reply.status).toBe(415);
  });

  it("asks for a file outside the add-in's pages", async () => {
    const live = await start();
    for (const path of ["/../../etc/passwd", "/%2e%2e/secret", "/package.json", "/api/nothing"]) {
      expect((await call(live, { path })).status).toBe(path === "/api/nothing" ? 401 : 404);
    }
  });
});

describe("a paired pane", () => {
  it("reads a Word document and is told which sentence to comment on", async () => {
    const live = await start();
    const token = await pair(live, "word");
    const reply = await call(live, {
      method: "POST",
      path: "/api/word/review",
      headers: { Authorization: `Bearer ${token}`, Origin: own(live) },
      body: {
        paragraphs: ["Tides rise and fall.", "The facilities end on 14 February 2029. Fine."],
      },
    });
    expect(reply.status).toBe(200);
    expect(reply.json).toEqual({
      finding: {
        paragraph: 1,
        sentence: "The facilities end on 14 February 2029.",
        comment: "Checked by IncarnaMind: The facilities end on 14 February 2029.",
      },
    });
  });

  it("reads an Excel range and is told which cell to add a note to", async () => {
    const live = await start();
    const token = await pair(live, "excel");
    const reply = await call(live, {
      method: "POST",
      path: "/api/excel/review",
      headers: { Authorization: `Bearer ${token}` },
      body: {
        values: [
          ["Revenue", 10],
          ["Total", 30],
        ],
        formulas: [
          ["Revenue", 10],
          ["Total", "=SUM(B1:B2)"],
        ],
      },
    });
    expect(reply.json).toEqual({
      finding: { row: 1, column: 1, note: "Checked by IncarnaMind: =SUM(B1:B2) gives 30" },
    });
  });

  it("is served the add-in's pages with a policy that keeps it to this computer", async () => {
    const live = await start();
    const page = await call(live, { path: "/taskpane.html" });
    expect(page.status).toBe(200);
    expect(page.headers["content-security-policy"]).toContain("default-src 'none'");
    expect(page.headers["cache-control"]).toBe("no-store");
  });

  it("answers a preflight from the hosted pane's origin, and no other, when one is allowed", async () => {
    const hosted = "https://addin.example";
    const live = await start({ extraOrigins: [hosted] });
    const reply = await call(live, {
      method: "OPTIONS",
      path: "/api/session",
      headers: {
        Origin: hosted,
        "Access-Control-Request-Method": "GET",
        "Access-Control-Request-Private-Network": "true",
      },
    });
    expect(reply.status).toBe(204);
    expect(reply.headers["access-control-allow-origin"]).toBe(hosted);
    expect(reply.headers["access-control-allow-private-network"]).toBe("true");
  });

  it("stops at once when the User unpairs it, on the stream and the socket", async () => {
    const live = await start();
    const token = await pair(live);
    const socket = await openSocket(live, { Origin: own(live) });
    socket.send({ type: "hello", token });
    expect(await socket.next()).toEqual({ type: "ready", data: { host: "word" } });

    const [pairing] = await live.bridge.pairings();
    await live.bridge.unpair(pairing?.id ?? "");

    expect(await socket.next()).toEqual({ type: "revoked", data: {} });
    await socket.closed;
    const after = await call(live, {
      path: "/api/session",
      headers: { Authorization: `Bearer ${token}` },
    });
    expect(after.status).toBe(401);
  });
});

describe("the WebSocket", () => {
  it("is refused to a page on another origin and to a client with no Origin", async () => {
    const live = await start();
    expect((await openSocket(live, { Origin: "https://evil.example" })).status).toBe(403);
    expect((await openSocket(live, {})).status).toBe(403);
  });

  it("is refused for a Host that isn't loopback", async () => {
    const live = await start();
    expect((await openSocket(live, { Origin: own(live), Host: "evil.example" })).status).toBe(421);
  });

  it("hangs up on a client whose first message is not a valid token", async () => {
    const live = await start();
    const socket = await openSocket(live, { Origin: own(live) });
    expect(socket.status).toBe(101);
    socket.send({ type: "hello", token: "wrong" });
    await socket.closed;
  });

  it("answers a ping once paired", async () => {
    const live = await start();
    const token = await pair(live);
    const socket = await openSocket(live, { Origin: own(live) });
    socket.send({ type: "hello", token });
    await socket.next();
    socket.send({ type: "ping" });
    expect(await socket.next()).toEqual({ type: "pong" });
    socket.end();
  });
});

describe("over plain loopback HTTP, for the probe of whether Office loads an http://localhost pane", () => {
  // protects=the HTTP probe still refuses what HTTPS refuses: no token, another origin, and the https origin of the same port; it serves the pane with a policy that allows its own ws:// socket; fails_when=plain HTTP drops the pairing or the origin check; no_claim=what Office allows, which only a sideload shows.
  it("serves the pane, pairs only with the code, and refuses an unpaired request", async () => {
    const folder = await mkdtemp(join(tmpdir(), "office-bridge-http-"));
    await mkdir(join(folder, "assets"), { recursive: true });
    await writeFile(join(folder, "taskpane.html"), "<!doctype html><title>pane</title>");
    const bridge = createOfficeBridge({
      adapter: {
        port: 0,
        staticDir: folder,
        authorityFile: join(folder, "authority.pem"),
        scheme: "http",
      },
      log: { info: () => undefined, warn: () => undefined, error: () => undefined },
      secrets: memorySecrets(),
      clock: () => 1_000,
      now: () => new Date(1_000).toISOString(),
    });
    const { port } = await bridge.start();
    try {
      const origin = `http://127.0.0.1:${port}`;
      const pane = await fetch(`${origin}/taskpane.html`);
      expect(pane.status).toBe(200);
      expect(pane.headers.get("content-security-policy")).toContain(`ws://localhost:${port}`);
      expect(existsSync(join(folder, "authority.pem"))).toBe(false);

      expect((await fetch(`${origin}/api/session`, { headers: { Origin: origin } })).status).toBe(
        401,
      );
      const wrong = await fetch(`${origin}/api/pair`, {
        method: "POST",
        headers: { Origin: origin, "Content-Type": "application/json" },
        body: JSON.stringify({ code: "000000", host: "word" }),
      });
      expect(wrong.status).not.toBe(200);

      const { code } = bridge.beginPairing();
      const asHttps = await fetch(`${origin}/api/pair`, {
        method: "POST",
        headers: { Origin: `https://127.0.0.1:${port}`, "Content-Type": "application/json" },
        body: JSON.stringify({ code, host: "word" }),
      });
      expect(asHttps.status).toBe(403);
      const paired = await fetch(`${origin}/api/pair`, {
        method: "POST",
        headers: { Origin: origin, "Content-Type": "application/json" },
        body: JSON.stringify({ code, host: "word" }),
      });
      expect(paired.status).toBe(200);
      const { token } = (await paired.json()) as { token: string };
      const session = await fetch(`${origin}/api/session`, {
        headers: { Origin: origin, Authorization: `Bearer ${token}` },
      });
      expect(session.status).toBe(200);
    } finally {
      await bridge.close();
    }
  });
});
