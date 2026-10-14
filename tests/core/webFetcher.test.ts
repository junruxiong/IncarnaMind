/**
 * Web search's fetcher (#128) reaches only the public internet. A local
 * server stands in for the web: the guard's lookup decides which hosts are
 * reached, and an allowed connection goes to the server. Every refusal is
 * made before the server hears a request.
 */
import { describe, expect, onTestFinished, test } from "vitest";
import { WebPageError } from "../../src/core/documents/web/errors";
import { fetchPage } from "../../src/core/documents/web/fetch";
import {
  isPublicAddress,
  publicOnlyLookup,
  type Resolve,
} from "../../src/core/tools/web/addresses";
import { createPublicFetch, MAX_REDIRECTS } from "../../src/core/tools/web/fetcher";
import { connectorTo, PUBLIC_ADDRESS as PUBLIC_V4, startWeb } from "../helpers/web";

const PUBLIC_V6 = "2606:4700:4700::1111";

const PAGE = `<!doctype html><html><head><title>Parks at night</title></head><body><article>
<h1>Parks at night</h1><p>Large parks stay cooler than the streets around them after sunset.</p>
</article></body></html>`;

/** A resolver from a table of host names; every other name doesn't exist. */
const resolving =
  (table: Record<string, string | string[] | (() => string)>): Resolve =>
  async (hostname) => {
    const entry = table[hostname];
    if (entry === undefined)
      throw Object.assign(new Error(`getaddrinfo ENOTFOUND ${hostname}`), { code: "ENOTFOUND" });
    const addresses =
      typeof entry === "function" ? [entry()] : Array.isArray(entry) ? entry : [entry];
    return addresses.map((address) => ({ address, family: address.includes(":") ? 6 : 4 }));
  };

function fetcher(port: number, resolve: Resolve) {
  const made = createPublicFetch({ resolve, connector: connectorTo(port) });
  onTestFinished(() => made.close());
  return made;
}

/** Fetches a page as web search does, resolving with why it was refused (or "fetched"). */
async function outcome(
  fetch: ReturnType<typeof fetcher>["fetch"],
  address: string,
): Promise<string> {
  try {
    await fetchPage(address, { fetch, limits: { timeoutMs: 5_000, maxBytes: 100_000 } });
    return "fetched";
  } catch (error) {
    if (error instanceof WebPageError) return error.reason;
    throw error;
  }
}

describe("Which addresses are public", () => {
  test.each([PUBLIC_V4, "8.8.8.8", "1.1.1.1", PUBLIC_V6, "2a00:1450:4009:81f::200e"])(
    "%s is public",
    (address) => expect(isPublicAddress(address)).toBe(true),
  );

  test.each([
    ["loopback", "127.0.0.1"],
    ["loopback, elsewhere in 127/8", "127.8.9.10"],
    ["this network", "0.0.0.0"],
    ["private 10/8", "10.0.0.5"],
    ["private 172.16/12", "172.20.1.1"],
    ["private 192.168/16", "192.168.1.10"],
    ["link-local, a cloud's metadata service", "169.254.169.254"],
    ["carrier-grade NAT", "100.64.0.1"],
    ["documentation", "203.0.113.7"],
    ["benchmarking", "198.18.0.1"],
    ["multicast", "224.0.0.1"],
    ["broadcast", "255.255.255.255"],
    ["reserved", "240.0.0.1"],
    ["IPv6 loopback", "::1"],
    ["IPv6 unspecified", "::"],
    ["IPv6 link-local", "fe80::1"],
    ["IPv6 link-local with a zone", "fe80::1%en0"],
    ["IPv6 unique local", "fd12:3456:789a::1"],
    ["IPv4-mapped loopback", "::ffff:127.0.0.1"],
    ["IPv4-mapped private, in hex", "::ffff:a00:5"],
    ["IPv4-mapped public", "::ffff:8.8.8.8"],
    ["IPv4-compatible loopback", "::127.0.0.1"],
    ["NAT64 of loopback", "64:ff9b::7f00:1"],
    ["6to4 of loopback", "2002:7f00:1::1"],
    ["Teredo", "2001::1"],
    ["IPv6 documentation", "2001:db8::1"],
    ["IPv6 multicast", "ff02::1"],
    ["a host name", "localhost"],
    ["nothing", ""],
  ])("%s (%s) isn't", (_what, address) => expect(isPublicAddress(address)).toBe(false));
});

describe("Web search's fetcher", { timeout: 20_000 }, () => {
  test("a public page is fetched", async () => {
    const web = await startWeb({ "/parks": () => ({ status: 200, body: PAGE }) });
    const { fetch } = fetcher(web.port, resolving({ "news.example": PUBLIC_V4 }));

    const page = await fetchPage("http://news.example/parks", { fetch });

    expect(page.html).toContain("Large parks stay cooler");
    expect(page.url).toBe("http://news.example/parks");
    expect(web.heard).toEqual(["news.example/parks"]);
  });

  test.each([
    ["a literal IPv4 loopback", "http://127.0.0.1/admin"],
    ["a private IPv4", "http://192.168.1.1/"],
    ["a cloud's metadata service", "http://169.254.169.254/latest/meta-data/"],
    ["IPv4 written as one number", "http://2130706433/"],
    ["IPv4 written in hex", "http://0x7f.1/"],
    ["a literal IPv6 loopback", "http://[::1]/"],
    ["IPv6 unique local", "http://[fd00::1]/"],
    ["IPv4-mapped IPv6", "http://[::ffff:127.0.0.1]/"],
    ["IPv4-mapped IPv6 of a private address", "http://[::ffff:10.0.0.1]/"],
  ])("%s is refused before anything is sent (%s)", async (_what, address) => {
    const web = await startWeb({});
    const { fetch } = fetcher(web.port, resolving({}));

    expect(await outcome(fetch, address)).toBe("not-public");
    expect(web.heard).toEqual([]);
  });

  test("a name that resolves to a local or private address is refused", async () => {
    const web = await startWeb({});
    const { fetch } = fetcher(
      web.port,
      resolving({
        "intranet.example": "10.0.0.5",
        "localhost.example": "127.0.0.1",
        "mapped.example": "::ffff:127.0.0.1",
        "mixed.example": [PUBLIC_V4, "192.168.0.2"],
      }),
    );

    for (const host of ["intranet", "localhost", "mapped", "mixed"]) {
      expect(await outcome(fetch, `http://${host}.example/`)).toBe("not-public");
    }
    expect(web.heard).toEqual([]);
  });

  test("DNS rebinding: a name public when first looked up and local when next connected to is refused", async () => {
    const web = await startWeb({ "/parks": () => ({ status: 200, body: PAGE }) });
    const answers = [PUBLIC_V4, "127.0.0.1"];
    const { fetch } = fetcher(
      web.port,
      resolving({ "rebind.example": () => answers.shift() ?? "127.0.0.1" }),
    );

    expect(await outcome(fetch, "http://rebind.example/parks")).toBe("fetched");
    // Each answer closes its connection, so the next request connects, and looks up, again.
    expect(await outcome(fetch, "http://rebind.example/parks")).toBe("not-public");
    expect(web.heard).toEqual(["rebind.example/parks"]);
  });

  test("a redirect to a local-network address is refused, written as an address or as a name", async () => {
    const web = await startWeb({
      "/to-loopback": (port) => ({
        status: 302,
        headers: { Location: `http://127.0.0.1:${port}/secret` },
      }),
      "/to-router": () => ({ status: 301, headers: { Location: "http://router.example/admin" } }),
      "/to-metadata": () => ({
        status: 307,
        headers: { Location: "http://169.254.169.254/latest/meta-data/" },
      }),
      "/secret": () => ({ status: 200, body: PAGE }),
    });
    const { fetch } = fetcher(
      web.port,
      resolving({ "news.example": PUBLIC_V4, "router.example": "192.168.1.1" }),
    );

    expect(await outcome(fetch, "http://news.example/to-loopback")).toBe("not-public");
    expect(await outcome(fetch, "http://news.example/to-router")).toBe("not-public");
    expect(await outcome(fetch, "http://news.example/to-metadata")).toBe("not-public");
    expect(web.heard).toEqual([
      "news.example/to-loopback",
      "news.example/to-router",
      "news.example/to-metadata",
    ]);
  });

  test(`at most ${MAX_REDIRECTS} redirects are followed, each checked`, async () => {
    const hop = (to: string) => () => ({ status: 302, headers: { Location: to } });
    const web = await startWeb({
      "/1": hop("/2"),
      "/2": hop("/3"),
      "/3": hop("/4"),
      "/4": hop("/5"),
      "/5": hop("/parks"),
      "/0": hop("/1"),
      "/parks": () => ({ status: 200, body: PAGE }),
    });
    const { fetch } = fetcher(web.port, resolving({ "news.example": PUBLIC_V4 }));

    expect(await outcome(fetch, "http://news.example/1")).toBe("fetched");
    expect(await outcome(fetch, "http://news.example/0")).toBe("too-many-redirects");
  });

  test("a proxy set in the environment isn't used", async () => {
    const web = await startWeb({ "/parks": () => ({ status: 200, body: PAGE }) });
    const before = { http: process.env.HTTP_PROXY, https: process.env.HTTPS_PROXY };
    process.env.HTTP_PROXY = "http://127.0.0.1:9";
    process.env.HTTPS_PROXY = "http://127.0.0.1:9";
    onTestFinished(() => {
      if (before.http === undefined) delete process.env.HTTP_PROXY;
      else process.env.HTTP_PROXY = before.http;
      if (before.https === undefined) delete process.env.HTTPS_PROXY;
      else process.env.HTTPS_PROXY = before.https;
    });
    const { fetch } = fetcher(web.port, resolving({ "news.example": PUBLIC_V4 }));

    expect(await outcome(fetch, "http://news.example/parks")).toBe("fetched");
  });

  test("undici's own connections go through the guard: a name for this computer is refused", async () => {
    const web = await startWeb({ "/": () => ({ status: 200, body: PAGE }) });
    const made = createPublicFetch({ resolve: resolving({ "home.example": "127.0.0.1" }) });
    onTestFinished(() => made.close());

    expect(await outcome(made.fetch, `http://home.example:${web.port}/`)).toBe("not-public");
    expect(web.heard).toEqual([]);
  });

  test("the guarded lookup gives a public name's addresses, one or all as asked", async () => {
    const lookup = publicOnlyLookup(resolving({ "dual.example": [PUBLIC_V4, PUBLIC_V6] }));
    const one = await new Promise<unknown[]>((resolve) =>
      lookup("dual.example", { family: 6 }, (...args) => resolve(args)),
    );
    const all = await new Promise<unknown[]>((resolve) =>
      lookup("dual.example", { all: true }, (...args) => resolve(args)),
    );

    expect(one).toEqual([null, PUBLIC_V6, 6]);
    expect(all).toEqual([
      null,
      [
        { address: PUBLIC_V4, family: 4 },
        { address: PUBLIC_V6, family: 6 },
      ],
    ]);
  });
});
