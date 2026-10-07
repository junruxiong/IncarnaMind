/**
 * OAuth 2.0 sign-in for a desktop app: the authorization code flow with PKCE
 * (RFC 7636) in the User's browser, redirecting to a small HTTP server on the
 * loopback interface (RFC 8252). Used by the ChatGPT plan provider and, later,
 * by remote Connectors that sign in with OAuth (#39), so nothing here knows
 * about a particular service: the caller passes every URL, the client ID, the
 * port and the path.
 *
 * The browser is opened through the core's `Browser` adapter; this module
 * never imports Electron.
 */
import { createHash, randomBytes } from "node:crypto";
import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import type { AddressInfo } from "node:net";
import { isRecord } from "./errors";

/** The browser sign-in gives up after this long unless the caller sets its own limit. */
export const DEFAULT_SIGN_IN_TIMEOUT_MS = 5 * 60_000;

/** The server always listens here, never on other interfaces. */
const LOOPBACK_HOST = "127.0.0.1";

export interface OAuthPage {
  title: string;
  body: string;
}

export interface LoopbackAuthorizationOptions {
  /** The authorization endpoint, e.g. "https://auth.example.com/oauth/authorize". */
  authorizeUrl: string;
  /** The token endpoint, for exchanging the code. */
  tokenUrl: string;
  clientId: string;
  /** Space-separated scopes, if the server wants them. */
  scope?: string;
  /**
   * The loopback port to listen on. Some servers allow only the redirect URI
   * they registered, so a fixed port; 0 picks any free port, for servers that
   * accept any loopback port (RFC 8252 §7.3).
   */
  port: number;
  /** The redirect path, e.g. "/auth/callback". */
  path: string;
  /** Extra query parameters for the authorization request (e.g. RFC 8707 `resource`). */
  authorizeParams?: Readonly<Record<string, string>>;
  /** Extra form parameters for the code exchange. */
  tokenParams?: Readonly<Record<string, string>>;
  /** Opens the authorization URL in the User's browser. */
  openBrowser(url: string): Promise<void>;
  /** What the browser tab shows when sign-in finishes, or fails. */
  pages: { success: OAuthPage; failure: OAuthPage };
  /** Defaults to five minutes. */
  timeoutMs?: number;
  /** Aborting cancels the sign-in and frees the port. */
  signal?: AbortSignal;
  fetch?: typeof globalThis.fetch;
}

/** A token endpoint's answer (RFC 6749 §5.1), in camelCase. */
export interface OAuthTokens {
  accessToken: string;
  /** Absent when the server issues no refresh token, or keeps the old one on refresh. */
  refreshToken: string | null;
  /** OpenID Connect servers send one when the "openid" scope was requested. */
  idToken: string | null;
  /** Lifetime of the access token in seconds, as the server sent it. */
  expiresIn: number | null;
  scope: string | null;
}

export type OAuthFlowFailure =
  /** Another program is listening on the port (for example another app signing in). */
  | "port-in-use"
  /** Nothing came back from the browser in time. */
  | "timed-out"
  /** The caller cancelled. */
  | "cancelled"
  /** The authorization server redirected back with an error, e.g. the User declined. */
  | "denied";

/** The browser part of a sign-in failed. */
export class OAuthFlowError extends Error {
  override name = "OAuthFlowError";
  constructor(
    readonly reason: OAuthFlowFailure,
    message: string,
  ) {
    super(message);
  }
}

/** The token endpoint refused a code exchange or a refresh, or sent something unusable. */
export class OAuthTokenError extends Error {
  override name = "OAuthTokenError";
  constructor(
    /** The HTTP status, or null when the response itself was unusable. */
    readonly status: number | null,
    /** The OAuth error code, e.g. "invalid_grant". */
    readonly code: string | null,
    message: string,
  ) {
    super(message);
  }
}

const base64Url = (bytes: Buffer) => bytes.toString("base64url");

/** A PKCE verifier and its S256 challenge (RFC 7636). */
export function createPkcePair(): { verifier: string; challenge: string } {
  const verifier = base64Url(randomBytes(32));
  const challenge = base64Url(createHash("sha256").update(verifier).digest());
  return { verifier, challenge };
}

/**
 * The claims in a JWT's payload, without verifying its signature: only for
 * reading what a token we just received from the server says about itself.
 */
export function decodeJwtClaims(token: string): Record<string, unknown> | null {
  const parts = token.split(".");
  if (parts.length !== 3 || !parts[1]) return null;
  try {
    const claims: unknown = JSON.parse(Buffer.from(parts[1], "base64url").toString("utf8"));
    return isRecord(claims) ? claims : null;
  } catch {
    return null;
  }
}

const escapeHtml = (text: string) =>
  text.replace(
    /[&<>"']/g,
    (char) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[char] ?? char,
  );

function renderPage({ title, body }: OAuthPage, detail?: string): string {
  return `<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>${escapeHtml(title)}</title>
<style>body{font:16px/1.5 system-ui,sans-serif;color:#1f2937;background:#f3f4f6;margin:0;display:grid;place-items:center;min-height:100vh}main{background:#fff;border-radius:9px;padding:2rem;max-width:28rem;box-shadow:0 1px 3px #0002}h1{font-size:1.25rem;margin:0 0 .5rem}p{margin:0}small{display:block;margin-top:1rem;color:#6b7280;word-break:break-word}</style>
</head><body><main><h1>${escapeHtml(title)}</h1><p>${escapeHtml(body)}</p>${
    detail ? `<small>${escapeHtml(detail)}</small>` : ""
  }</main></body></html>`;
}

/** Resolves once the page has been handed to the OS, so closing the server can't cut it off. */
function sendPage(response: ServerResponse, status: number, html: string): Promise<void> {
  return new Promise((resolve) => {
    response.writeHead(status, {
      "content-type": "text/html; charset=utf-8",
      "cache-control": "no-store",
      connection: "close",
    });
    response.end(html, () => resolve());
  });
}

const text = (value: unknown) => (typeof value === "string" && value !== "" ? value : null);

async function readTokenResponse(response: Response, action: string): Promise<OAuthTokens> {
  const raw = await response.text();
  let body: unknown;
  try {
    body = JSON.parse(raw);
  } catch {
    body = undefined;
  }
  if (!response.ok) {
    const code = isRecord(body) ? text(body.error) : null;
    const description = isRecord(body) ? text(body.error_description) : null;
    throw new OAuthTokenError(
      response.status,
      code,
      `The ${action} was refused (${response.status}): ${description ?? code ?? (raw.slice(0, 200) || response.statusText)}`,
    );
  }
  const accessToken = isRecord(body) ? text(body.access_token) : null;
  if (!isRecord(body) || !accessToken) {
    throw new OAuthTokenError(null, null, `The ${action} response has no access token.`);
  }
  return {
    accessToken,
    refreshToken: text(body.refresh_token),
    idToken: text(body.id_token),
    expiresIn: typeof body.expires_in === "number" ? body.expires_in : null,
    scope: text(body.scope),
  };
}

async function postForm(
  fetcher: typeof globalThis.fetch,
  url: string,
  form: Record<string, string>,
  signal?: AbortSignal,
): Promise<Response> {
  return fetcher(url, {
    method: "POST",
    headers: {
      "content-type": "application/x-www-form-urlencoded",
      accept: "application/json",
    },
    body: new URLSearchParams(form),
    ...(signal ? { signal } : {}),
  });
}

/** Trades a refresh token for new tokens (RFC 6749 §6). Network failures reject as fetch does. */
export async function refreshOAuthTokens(options: {
  tokenUrl: string;
  clientId: string;
  refreshToken: string;
  scope?: string;
  params?: Readonly<Record<string, string>>;
  fetch?: typeof globalThis.fetch;
  signal?: AbortSignal;
}): Promise<OAuthTokens> {
  const response = await postForm(
    options.fetch ?? globalThis.fetch,
    options.tokenUrl,
    {
      grant_type: "refresh_token",
      client_id: options.clientId,
      refresh_token: options.refreshToken,
      ...(options.scope ? { scope: options.scope } : {}),
      ...options.params,
    },
    options.signal,
  );
  return readTokenResponse(response, "token refresh");
}

/**
 * Runs the whole browser sign-in: listens on the loopback port, opens the
 * authorization URL, waits for the redirect, checks its state, exchanges the
 * code with the PKCE verifier, shows a "you can close this tab" page, and
 * frees the port. Rejects with `OAuthFlowError` or `OAuthTokenError`.
 */
export async function authorizeWithLoopback(
  options: LoopbackAuthorizationOptions,
): Promise<OAuthTokens> {
  const { signal } = options;
  if (signal?.aborted) throw new OAuthFlowError("cancelled", "The sign-in was cancelled.");
  const fetcher = options.fetch ?? globalThis.fetch;
  const pkce = createPkcePair();
  const state = base64Url(randomBytes(24));

  let settle!: (tokens: OAuthTokens) => void;
  let fail!: (error: Error) => void;
  const result = new Promise<OAuthTokens>((resolve, reject) => {
    settle = resolve;
    fail = reject;
  });
  // A failure before anyone awaits (e.g. while the browser opens) must not go unhandled.
  result.catch(() => undefined);

  let redirectUri = "";
  let claimed = false;

  const handle = async (request: IncomingMessage, response: ServerResponse) => {
    const url = new URL(request.url ?? "/", `http://${LOOPBACK_HOST}`);
    if (request.method !== "GET" || url.pathname !== options.path) {
      response.writeHead(404, { connection: "close" }).end();
      return;
    }
    // A redirect that didn't come from this sign-in: ignore it and keep waiting.
    const failurePage = (detail: string) => renderPage(options.pages.failure, detail);
    if (url.searchParams.get("state") !== state) {
      await sendPage(response, 400, failurePage("The sign-in state didn't match."));
      return;
    }
    if (claimed) {
      await sendPage(response, 409, failurePage("This sign-in already finished."));
      return;
    }
    claimed = true;
    const error = url.searchParams.get("error");
    if (error) {
      const description = url.searchParams.get("error_description") ?? error;
      await sendPage(response, 400, failurePage(description));
      fail(new OAuthFlowError("denied", `The sign-in was refused: ${description}`));
      return;
    }
    const code = url.searchParams.get("code");
    if (!code) {
      await sendPage(response, 400, failurePage("No authorization code came back."));
      fail(new OAuthFlowError("denied", "The sign-in returned no authorization code."));
      return;
    }
    try {
      const exchange = await postForm(
        fetcher,
        options.tokenUrl,
        {
          grant_type: "authorization_code",
          client_id: options.clientId,
          code,
          redirect_uri: redirectUri,
          code_verifier: pkce.verifier,
          ...options.tokenParams,
        },
        signal,
      );
      const tokens = await readTokenResponse(exchange, "sign-in");
      await sendPage(response, 200, renderPage(options.pages.success));
      settle(tokens);
    } catch (failure) {
      const reason = failure instanceof Error ? failure : new Error(String(failure));
      await sendPage(response, 502, failurePage(reason.message));
      fail(reason);
    }
  };

  const server = createServer((request, response) => {
    handle(request, response).catch((error: unknown) => {
      if (!response.headersSent) response.writeHead(500, { connection: "close" }).end();
      fail(error instanceof Error ? error : new Error(String(error)));
    });
  });

  await new Promise<void>((resolve, reject) => {
    server.once("error", (error: NodeJS.ErrnoException) => {
      reject(
        error.code === "EADDRINUSE"
          ? new OAuthFlowError(
              "port-in-use",
              `Port ${options.port} on this computer is in use by another program, so the sign-in can't receive its answer.`,
            )
          : error,
      );
    });
    server.listen(options.port, LOOPBACK_HOST, () => resolve());
  });
  server.on("error", (error) => fail(error));

  const { port } = server.address() as AddressInfo;
  redirectUri = `http://${LOOPBACK_HOST}:${port}${options.path}`;

  const timeoutMs = options.timeoutMs ?? DEFAULT_SIGN_IN_TIMEOUT_MS;
  const timer = setTimeout(
    () => fail(new OAuthFlowError("timed-out", "The sign-in didn't finish in time.")),
    timeoutMs,
  );
  const onAbort = () => fail(new OAuthFlowError("cancelled", "The sign-in was cancelled."));
  signal?.addEventListener("abort", onAbort, { once: true });
  // Cancelled while the server was starting: the listener above came too late to hear it.
  if (signal?.aborted) onAbort();

  try {
    if (!signal?.aborted) {
      const authorize = new URL(options.authorizeUrl);
      const params: Record<string, string> = {
        response_type: "code",
        client_id: options.clientId,
        redirect_uri: redirectUri,
        ...(options.scope ? { scope: options.scope } : {}),
        code_challenge: pkce.challenge,
        code_challenge_method: "S256",
        state,
        ...options.authorizeParams,
      };
      for (const [name, value] of Object.entries(params)) authorize.searchParams.set(name, value);
      await options.openBrowser(authorize.href);
    }
    return await result;
  } finally {
    clearTimeout(timer);
    signal?.removeEventListener("abort", onAbort);
    // Responses are sent with "connection: close"; drop anything still open so the port is free now.
    await new Promise<void>((resolve) => {
      server.close(() => resolve());
      server.closeAllConnections();
    });
  }
}
