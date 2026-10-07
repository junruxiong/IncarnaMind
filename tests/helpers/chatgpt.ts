/**
 * A local stand-in for OpenAI, for the ChatGPT plan provider: an OAuth
 * authorization server (code exchange with PKCE, refresh) and the Codex model
 * endpoint, on one HTTP server on 127.0.0.1. Plus a fake browser that plays
 * the User signing in. No real account and no network.
 */
import { createHash } from "node:crypto";
import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import type { AddressInfo } from "node:net";
import { onTestFinished } from "vitest";
import type { Browser, ChatGptPlanEndpoints } from "../../src/core";

export const ACCOUNT_ID = "chatgpt-account-0001";
export const EMAIL = "reader@example.com";
export const ANSWER = "Hello from your plan.";

const AUTH_CLAIM = "https://api.openai.com/auth";

/** An unsigned JWT: the app only reads the claims of tokens it got straight from the server. */
function jwt(claims: Record<string, unknown>): string {
  const part = (value: object) => Buffer.from(JSON.stringify(value)).toString("base64url");
  return `${part({ alg: "none", typ: "JWT" })}.${part(claims)}.signature`;
}

const readBody = (request: IncomingMessage) =>
  new Promise<string>((resolve, reject) => {
    let body = "";
    request.on("data", (chunk) => {
      body += chunk;
    });
    request.on("end", () => resolve(body));
    request.on("error", reject);
  });

const sendJson = (response: ServerResponse, status: number, body: unknown) => {
  response.writeHead(status, { "content-type": "application/json" });
  response.end(JSON.stringify(body));
};

export interface IssuedTokens {
  accessToken: string;
  refreshToken: string;
  idToken: string;
}

export interface CodexRequest {
  headers: Record<string, string | undefined>;
  body: Record<string, unknown>;
}

/** What the fake Codex endpoint answers with, when the token is good. */
export type CodexReply =
  | { kind: "answer"; text: string }
  | { kind: "error"; status: number; body: unknown };

/** Server-sent events of a streamed Responses API answer, as the Codex endpoint sends them. */
function answerEvents(text: string, model: string): object[] {
  const response = { id: "resp_1", object: "response", created_at: 1_790_000_000, model };
  const words = text.split(/(?<= )/);
  return [
    { type: "response.created", response: { ...response, status: "in_progress", output: [] } },
    {
      type: "response.output_item.added",
      output_index: 0,
      item: { id: "msg_1", type: "message", role: "assistant", status: "in_progress", content: [] },
    },
    {
      type: "response.content_part.added",
      item_id: "msg_1",
      output_index: 0,
      content_index: 0,
      part: { type: "output_text", text: "", annotations: [] },
    },
    ...words.map((delta) => ({
      type: "response.output_text.delta",
      item_id: "msg_1",
      output_index: 0,
      content_index: 0,
      delta,
    })),
    {
      type: "response.output_item.done",
      output_index: 0,
      item: {
        id: "msg_1",
        type: "message",
        role: "assistant",
        status: "completed",
        content: [{ type: "output_text", text, annotations: [] }],
      },
    },
    {
      // Like the real endpoint, the final event can come without the output items.
      type: "response.completed",
      response: {
        ...response,
        status: "completed",
        output: [],
        usage: {
          input_tokens: 12,
          input_tokens_details: { cached_tokens: 0 },
          output_tokens: 5,
          output_tokens_details: { reasoning_tokens: 0 },
          total_tokens: 17,
        },
      },
    },
  ];
}

export interface FakeOpenAI {
  /** Point the core's `chatGptPlan` adapter here. The callback port is any free one. */
  endpoints: ChatGptPlanEndpoints;
  /** Every request to the token endpoint, as its form fields. */
  tokenRequests: Record<string, string>[];
  /** Every request that reached the Codex endpoint. */
  codexRequests: CodexRequest[];
  /** Tokens handed out, oldest first. */
  issued: IssuedTokens[];
  /** Access-token lifetime in seconds for the next tokens. */
  expiresIn: number;
  /** Milliseconds each refresh takes. */
  refreshDelayMs: number;
  /** When set, refreshes are refused with this status and body. */
  refreshRefusal: { status: number; body: unknown } | null;
  /** How the Codex endpoint answers a request with a valid token. */
  codexReply: CodexReply;
  /** Makes every access token issued so far invalid, as if OpenAI revoked them. */
  revokeAccessTokens(): void;
  /** What OpenAI's authorization page does when the User approves: a code bound to the request. */
  approve(authorizeUrl: string): string;
}

/** Starts the fake on a free port; stopped when the test finishes. */
export async function startFakeOpenAI(): Promise<FakeOpenAI> {
  const codes = new Map<string, { challenge: string; redirectUri: string; clientId: string }>();
  const validAccessTokens = new Set<string>();
  let codeCount = 0;

  const issue = (): IssuedTokens => {
    const n = fake.issued.length + 1;
    const tokens = {
      accessToken: jwt({
        n,
        exp: Math.floor(Date.now() / 1000) + fake.expiresIn,
        [AUTH_CLAIM]: { chatgpt_account_id: ACCOUNT_ID },
      }),
      refreshToken: `refresh-token-${n}-${"r".repeat(24)}`,
      idToken: jwt({
        n,
        email: EMAIL,
        [AUTH_CLAIM]: { chatgpt_account_id: ACCOUNT_ID, chatgpt_plan_type: "plus" },
      }),
    };
    fake.issued.push(tokens);
    validAccessTokens.add(tokens.accessToken);
    return tokens;
  };

  const tokenResponse = (tokens: IssuedTokens) => ({
    access_token: tokens.accessToken,
    refresh_token: tokens.refreshToken,
    id_token: tokens.idToken,
    token_type: "Bearer",
    expires_in: fake.expiresIn,
  });

  const handleToken = async (request: IncomingMessage, response: ServerResponse) => {
    const form = Object.fromEntries(new URLSearchParams(await readBody(request)));
    fake.tokenRequests.push(form);
    if (form.grant_type === "authorization_code") {
      const pending = codes.get(form.code ?? "");
      codes.delete(form.code ?? "");
      const challenge = createHash("sha256")
        .update(form.code_verifier ?? "")
        .digest("base64url");
      if (
        !pending ||
        pending.challenge !== challenge ||
        pending.redirectUri !== form.redirect_uri ||
        pending.clientId !== form.client_id
      ) {
        sendJson(response, 400, { error: "invalid_grant" });
        return;
      }
      sendJson(response, 200, tokenResponse(issue()));
      return;
    }
    if (form.grant_type === "refresh_token") {
      await new Promise((resolve) => setTimeout(resolve, fake.refreshDelayMs));
      if (fake.refreshRefusal) {
        sendJson(response, fake.refreshRefusal.status, fake.refreshRefusal.body);
        return;
      }
      // Refresh tokens rotate: only the newest one works.
      if (form.refresh_token !== fake.issued.at(-1)?.refreshToken) {
        sendJson(response, 400, { error: "invalid_grant", error_description: "reused" });
        return;
      }
      sendJson(response, 200, tokenResponse(issue()));
      return;
    }
    sendJson(response, 400, { error: "unsupported_grant_type" });
  };

  const handleCodex = async (request: IncomingMessage, response: ServerResponse) => {
    const body = JSON.parse(await readBody(request)) as Record<string, unknown>;
    const headers: Record<string, string | undefined> = {};
    for (const [name, value] of Object.entries(request.headers)) {
      headers[name] = Array.isArray(value) ? value.join(", ") : value;
    }
    fake.codexRequests.push({ headers, body });
    const token = headers.authorization?.replace(/^Bearer /, "") ?? "";
    if (!validAccessTokens.has(token) || headers["chatgpt-account-id"] !== ACCOUNT_ID) {
      sendJson(response, 401, { error: { message: "Your authentication token is invalid." } });
      return;
    }
    const reply = fake.codexReply;
    if (reply.kind === "error") {
      sendJson(response, reply.status, reply.body);
      return;
    }
    response.writeHead(200, { "content-type": "text/event-stream" });
    for (const event of answerEvents(reply.text, String(body.model))) {
      response.write(
        `event: ${(event as { type: string }).type}\ndata: ${JSON.stringify(event)}\n\n`,
      );
    }
    response.end();
  };

  const server = createServer((request, response) => {
    const path = new URL(request.url ?? "/", "http://127.0.0.1").pathname;
    const handler =
      request.method === "POST" && path === "/oauth/token"
        ? handleToken
        : request.method === "POST" && path === "/backend-api/codex/responses"
          ? handleCodex
          : null;
    if (!handler) {
      response.writeHead(404).end();
      return;
    }
    handler(request, response).catch((error: unknown) => {
      response.writeHead(500).end(String(error));
    });
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  onTestFinished(
    () =>
      new Promise<void>((resolve) => {
        server.close(() => resolve());
        server.closeAllConnections();
      }),
  );
  const origin = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;

  const fake: FakeOpenAI = {
    endpoints: {
      authorizeUrl: `${origin}/oauth/authorize`,
      tokenUrl: `${origin}/oauth/token`,
      callbackPort: 0,
      signInTimeoutMs: 10_000,
      codexBaseUrl: `${origin}/backend-api/codex`,
    },
    tokenRequests: [],
    codexRequests: [],
    issued: [],
    expiresIn: 3600,
    refreshDelayMs: 0,
    refreshRefusal: null,
    codexReply: { kind: "answer", text: ANSWER },
    revokeAccessTokens: () => validAccessTokens.clear(),
    approve(authorizeUrl) {
      const params = new URL(authorizeUrl).searchParams;
      codeCount += 1;
      const code = `code-${codeCount}`;
      codes.set(code, {
        challenge: params.get("code_challenge") ?? "",
        redirectUri: params.get("redirect_uri") ?? "",
        clientId: params.get("client_id") ?? "",
      });
      return code;
    },
  };
  return fake;
}

export interface BrowserVisit {
  status: number;
  html: string;
}

export interface FakeBrowser extends Browser {
  /** Every URL the app opened. */
  opened: string[];
  /** The page the loopback server showed for each redirect, once it arrives. */
  visits: Promise<BrowserVisit>[];
  /**
   * What the User does on OpenAI's page: "approve" signs in, "decline"
   * refuses, "ignore" walks away.
   */
  behaviour: "approve" | "decline" | "ignore";
}

/** A browser in which the User signs in to the fake OpenAI, redirecting back like the real page. */
export function createFakeBrowser(fake: FakeOpenAI): FakeBrowser {
  const browser: FakeBrowser = {
    opened: [],
    visits: [],
    behaviour: "approve",
    async open(url) {
      browser.opened.push(url);
      if (browser.behaviour === "ignore") return;
      const params = new URL(url).searchParams;
      const redirect = new URL(params.get("redirect_uri") ?? "");
      redirect.searchParams.set("state", params.get("state") ?? "");
      if (browser.behaviour === "approve") {
        redirect.searchParams.set("code", fake.approve(url));
      } else {
        redirect.searchParams.set("error", "access_denied");
        redirect.searchParams.set("error_description", "The User declined.");
      }
      // Like a real browser, the redirect happens after `open` returns.
      browser.visits.push(
        new Promise((resolve) => setTimeout(resolve, 5)).then(async () => {
          const response = await fetch(redirect);
          return { status: response.status, html: await response.text() };
        }),
      );
    },
  };
  return browser;
}

/** A free port that something else is listening on, until the test finishes. */
export async function occupiedPort(): Promise<number> {
  const server = createServer((_request, response) => response.end());
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  onTestFinished(() => new Promise<void>((resolve) => server.close(() => resolve())));
  return (server.address() as AddressInfo).port;
}
