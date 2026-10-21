/**
 * The Office add-in bridge (spike for #1272, never on by default): the app
 * serves the add-in's own pages and a small API over HTTPS on this computer's
 * loopback addresses only, with a certificate it made for itself. An add-in in
 * Word or Excel pairs once with a code the app shows, then reads the open
 * document and gets back what to comment on. Nothing else listens, and nothing
 * leaves the computer.
 */
import { readFile, writeFile } from "node:fs/promises";
import { createServer as createHttpServer } from "node:http";
import { createServer as createHttpsServer, type Server } from "node:https";
import { join } from "node:path";
import type { Logger } from "../adapters";
import type { Secrets } from "../secrets";
import {
  createAuthority,
  issueLoopbackCertificate,
  type LocalAuthority,
  type LoopbackCertificate,
} from "./certificate";
import { createOfficeHandler, type StaticFile } from "./handler";
import { createPairings, type OfficePairing, type PairingCode } from "./pairing";

export type { OfficeHost, OfficePairing, PairingCode } from "./pairing";

export interface OfficeBridgeAdapter {
  /** The one port the manifest names. 0 takes any free one, for tests. */
  port: number;
  /** The add-in's pages: taskpane.html, taskpane.js, styles.css and assets/. */
  staticDir: string;
  /** Where the public part of the local authority is written, for the User to trust. */
  authorityFile: string;
  folderName?: string;
  /** Other origins allowed to call the API, such as a hosted task pane. */
  extraOrigins?: readonly string[];
  /**
   * "http" serves the pane and the API over plain loopback HTTP, with no certificate: the probe
   * of whether Office for Mac loads an http://localhost pane at all (#1272). Default "https".
   */
  scheme?: "http" | "https";
}

export interface OfficeBridgeOptions {
  adapter: OfficeBridgeAdapter;
  log: Logger;
  secrets: Pick<Secrets, "set" | "tryGet" | "delete">;
  clock(): number;
  now(): string;
}

export interface OfficeBridgeAddress {
  port: number;
  /** The loopback addresses it listens on: both, or only 127.0.0.1 where there is no ::1. */
  addresses: string[];
  authorityFile: string;
}

const AUTHORITY_SECRET = "office-bridge:authority";

const PAGES: Readonly<Record<string, string>> = {
  "/taskpane.html": "text/html; charset=utf-8",
  "/taskpane.js": "text/javascript; charset=utf-8",
  "/styles.css": "text/css; charset=utf-8",
  "/assets/icon-16.png": "image/png",
  "/assets/icon-32.png": "image/png",
  "/assets/icon-64.png": "image/png",
  "/assets/icon-80.png": "image/png",
  "/assets/icon-128.png": "image/png",
};

const LOOPBACK_HOSTNAMES = ["localhost", "127.0.0.1", "[::1]"] as const;

export function loopbackHosts(port: number): Set<string> {
  return new Set(LOOPBACK_HOSTNAMES.map((name) => `${name}:${port}`));
}

export function loopbackOrigins(
  port: number,
  extra: readonly string[] = [],
  scheme: "http" | "https" = "https",
): Set<string> {
  return new Set([...LOOPBACK_HOSTNAMES.map((name) => `${scheme}://${name}:${port}`), ...extra]);
}

export function pageSecurityPolicy(port: number, scheme: "http" | "https" = "https"): string {
  // A bracketed IPv6 host is not a valid CSP source (WebKit ignores it), and 'self' covers ::1.
  const socket = scheme === "http" ? "ws" : "wss";
  const sockets = ["localhost", "127.0.0.1"].map((name) => `${socket}://${name}:${port}`).join(" ");
  return [
    "default-src 'none'",
    "script-src 'self' https://officeapis.public.onecdn.static.microsoft",
    "style-src 'self' 'unsafe-inline'",
    "img-src 'self' data:",
    `connect-src 'self' ${sockets} https://officeapis.public.onecdn.static.microsoft`,
    "frame-ancestors 'self' https://*.office.com https://*.officeapps.live.com https://*.office365.com https://*.microsoft365.com https://*.sharepoint.com",
  ].join("; ");
}

export function createOfficeBridge(options: OfficeBridgeOptions) {
  const { adapter, log } = options;
  const servers: Server[] = [];
  let started: OfficeBridgeAddress | null = null;
  const handlers: { revoke(pairingId: string): void; closeAll(): void }[] = [];

  const pairings = createPairings({
    secrets: options.secrets,
    clock: options.clock,
    now: options.now,
    removed: (pairingId) => {
      for (const handler of handlers) handler.revoke(pairingId);
    },
  });

  const authority = async (): Promise<LocalAuthority> => {
    const kept = await options.secrets.tryGet(AUTHORITY_SECRET);
    if (kept) return JSON.parse(kept) as LocalAuthority;
    const made = createAuthority(new Date());
    await options.secrets.set(AUTHORITY_SECRET, JSON.stringify(made));
    return made;
  };

  const file = async (path: string): Promise<StaticFile | null> => {
    const type = PAGES[path];
    if (!type) return null;
    try {
      return { type, body: await readFile(join(adapter.staticDir, path)) };
    } catch {
      return null;
    }
  };

  const scheme = adapter.scheme ?? "https";
  const listen = (
    tls: { certificate: LoopbackCertificate; local: LocalAuthority } | null,
    address: string,
    port: number,
  ) =>
    new Promise<Server>((done, fail) => {
      const server = tls
        ? createHttpsServer({
            key: tls.certificate.keyPem,
            cert: `${tls.certificate.certificatePem}${tls.local.certificatePem}`,
            minVersion: "TLSv1.2",
          })
        : (createHttpServer() as unknown as Server);
      server.once("error", fail);
      server.listen(port, address, () => {
        server.off("error", fail);
        done(server);
      });
    });

  const open = async (): Promise<OfficeBridgeAddress> => {
    let tls: { certificate: LoopbackCertificate; local: LocalAuthority } | null = null;
    if (scheme === "https") {
      const local = await authority();
      tls = { certificate: issueLoopbackCertificate(local, new Date()), local };
      await writeFile(adapter.authorityFile, local.certificatePem, { mode: 0o644 });
    }

    const first = await listen(tls, "127.0.0.1", adapter.port);
    servers.push(first);
    const address = first.address();
    const port = typeof address === "object" && address ? address.port : adapter.port;
    try {
      servers.push(await listen(tls, "::1", port));
    } catch (error) {
      log.warn("office.noIpv6Loopback", { code: (error as NodeJS.ErrnoException).code ?? null });
    }

    const handler = createOfficeHandler({
      hosts: loopbackHosts(port),
      origins: loopbackOrigins(port, adapter.extraOrigins, scheme),
      pairings,
      file,
      folderName: adapter.folderName ?? "Example folder",
      contentSecurityPolicy: pageSecurityPolicy(port, scheme),
      log,
    });
    handlers.push(handler);
    for (const server of servers) {
      server.on("request", (request, response) => void handler.request(request, response));
      server.on("upgrade", handler.upgrade);
    }
    started = {
      port,
      addresses: servers.map((server) => String((server.address() as { address: string }).address)),
      authorityFile: adapter.authorityFile,
    };
    log.info("office.listening", { port, scheme });
    return started;
  };
  let opening: Promise<OfficeBridgeAddress> | null = null;

  return {
    // The core starts it, and the pairing dialog waits for it: both get the one start.
    start(): Promise<OfficeBridgeAddress> {
      if (started) return Promise.resolve(started);
      opening ??= open().finally(() => {
        opening = null;
      });
      return opening;
    },

    beginPairing(): PairingCode {
      return pairings.start();
    },

    pairings(): Promise<OfficePairing[]> {
      return pairings.list();
    },

    unpair(pairingId: string): Promise<void> {
      return pairings.remove(pairingId);
    },

    async close(): Promise<void> {
      for (const handler of handlers) handler.closeAll();
      await Promise.all(
        servers.map(
          (server) =>
            new Promise<void>((done) => {
              server.closeAllConnections();
              server.close(() => done());
            }),
        ),
      );
      servers.length = 0;
      handlers.length = 0;
      started = null;
    },
  };
}

export type OfficeBridge = ReturnType<typeof createOfficeBridge>;
