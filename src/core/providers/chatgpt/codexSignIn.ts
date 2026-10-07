/**
 * Signing in to ChatGPT with the public OAuth registration of OpenAI's Codex
 * CLI: its client ID, its fixed loopback callback (port 1455, path
 * /auth/callback) and its authorization parameters. OpenAI hasn't approved
 * this for other apps and may block it, which is why the provider using it is
 * experimental and off by default.
 *
 * This is the part the official "Sign in with ChatGPT" replaces once OpenAI
 * admits IncarnaMind: the endpoint adapter only needs `ChatGptCredentials`.
 *
 * Tokens (access, refresh and ID) are kept only in the secret store, as one
 * secret, never in SQLite. An access token is refreshed when it is within
 * five minutes of expiring, and concurrent requests share one refresh. A
 * refresh the server refuses ends the sign-in: the tokens are deleted and the
 * User is asked to sign in again.
 *
 * Sources, checked 2026-10-07: the Codex CLI (github.com/openai/codex at
 * 24edd7b: `codex-rs/login/src/server.rs`, `codex-rs/login/src/auth/manager.rs`,
 * `codex-rs/login/src/token_data.rs`) and the MIT-licensed pi-ai package
 * (@earendil-works/pi-ai 1.0.4, `dist/auth/oauth/openai-codex.js`). No code is
 * copied from either.
 */
import type { Browser } from "../../adapters";
import type { ChatGptAccount } from "../../api";
import { SecretStorageError } from "../../errors";
import {
  authorizeWithLoopback,
  decodeJwtClaims,
  type OAuthPage,
  OAuthTokenError,
  type OAuthTokens,
  refreshOAuthTokens,
} from "../../oauth";
import type { Secrets } from "../../secrets";
import type { SettingsStore } from "../../settings";
import type { ChatGptCredentials } from "./codexEndpoint";
import { ORIGINATOR } from "./codexEndpoint";
import { ChatGptSignInRequiredError } from "./errors";

/** The Codex CLI's OAuth client ID (`CLIENT_ID` in `codex-rs/login/src/auth/manager.rs`). */
export const CODEX_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann";

export const CODEX_AUTHORIZE_URL = "https://auth.openai.com/oauth/authorize";
export const CODEX_TOKEN_URL = "https://auth.openai.com/oauth/token";

/** The only callback port the registration allows besides its fallback, 1457, so it can't change. */
export const CODEX_CALLBACK_PORT = 1455;
export const CODEX_CALLBACK_PATH = "/auth/callback";

/** What the Codex CLI asks for, minus its Connector scopes. `offline_access` brings a refresh token. */
const SCOPE = "openid profile email offline_access";

/** Asks for the ChatGPT account in the ID token, and the short consent screen the Codex CLI gets. */
const AUTHORIZE_PARAMS = {
  id_token_add_organizations: "true",
  codex_cli_simplified_flow: "true",
  originator: ORIGINATOR,
};

/** The claim holding ChatGPT details in OpenAI's tokens. */
const AUTH_CLAIM = "https://api.openai.com/auth";
const PROFILE_CLAIM = "https://api.openai.com/profile";

/** Refresh this long before the access token expires, as the Codex CLI does. */
const REFRESH_WINDOW_MS = 5 * 60_000;

/** When neither the server nor the token says how long it lasts. */
const ASSUMED_LIFETIME_MS = 60 * 60_000;

const TOKENS_SECRET = "chatgpt-plan:tokens";

/** Per-device: the last sign-in ended because a refresh was refused. */
const EXPIRED_FLAG = "chatGptSignInExpired";

/** Where the sign-in happens. Tests point these at a local fake server. */
export interface CodexSignInEndpoints {
  authorizeUrl: string;
  tokenUrl: string;
  /** 0 picks a free port; only tests do that, since OpenAI accepts only the registered one. */
  callbackPort: number;
  signInTimeoutMs: number;
}

interface StoredTokens {
  accessToken: string;
  refreshToken: string | null;
  idToken: string | null;
  /** Milliseconds since the epoch. */
  expiresAt: number;
  accountId: string;
  email: string | null;
  plan: string | null;
}

const text = (value: unknown) => (typeof value === "string" && value !== "" ? value : null);
const record = (value: unknown): Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {};

function parseStored(json: string | null): StoredTokens | null {
  if (!json) return null;
  try {
    const value = record(JSON.parse(json));
    const accessToken = text(value.accessToken);
    const accountId = text(value.accountId);
    if (!accessToken || !accountId || typeof value.expiresAt !== "number") return null;
    return {
      accessToken,
      refreshToken: text(value.refreshToken),
      idToken: text(value.idToken),
      expiresAt: value.expiresAt,
      accountId,
      email: text(value.email),
      plan: text(value.plan),
    };
  } catch {
    return null;
  }
}

/**
 * What the tokens say: the ChatGPT account (from the ID token, else the
 * access token), the email and plan, and when the access token expires.
 * Refreshes may leave out the refresh and ID tokens; the previous ones stay.
 */
function toStored(tokens: OAuthTokens, previous: StoredTokens | null, now: number): StoredTokens {
  const idToken = tokens.idToken ?? previous?.idToken ?? null;
  const idClaims = record(idToken ? decodeJwtClaims(idToken) : null);
  const accessClaims = record(decodeJwtClaims(tokens.accessToken));
  const idAuth = record(idClaims[AUTH_CLAIM]);
  const accessAuth = record(accessClaims[AUTH_CLAIM]);
  const accountId =
    text(idAuth.chatgpt_account_id) ??
    text(accessAuth.chatgpt_account_id) ??
    previous?.accountId ??
    null;
  if (!accountId) throw new Error("The sign-in didn't say which ChatGPT account to use.");
  const expiresAt =
    tokens.expiresIn !== null
      ? now + tokens.expiresIn * 1000
      : typeof accessClaims.exp === "number"
        ? accessClaims.exp * 1000
        : now + ASSUMED_LIFETIME_MS;
  return {
    accessToken: tokens.accessToken,
    refreshToken: tokens.refreshToken ?? previous?.refreshToken ?? null,
    idToken,
    expiresAt,
    accountId,
    email:
      text(idClaims.email) ??
      text(record(idClaims[PROFILE_CLAIM]).email) ??
      previous?.email ??
      null,
    plan: text(idAuth.chatgpt_plan_type) ?? previous?.plan ?? null,
  };
}

/** The token endpoint refused the refresh token itself, as opposed to a passing failure. */
const isRefusedRefresh = (error: unknown) =>
  error instanceof OAuthTokenError &&
  error.status !== null &&
  error.status >= 400 &&
  error.status < 500 &&
  error.status !== 408 &&
  error.status !== 429;

export type CodexSignIn = ReturnType<typeof createCodexSignIn>;

export function createCodexSignIn(options: {
  endpoints: CodexSignInEndpoints;
  secrets: Secrets;
  settings: SettingsStore;
  browser: Browser;
  /** Milliseconds since the epoch. */
  now: () => number;
  /** The browser tab's pages, in the User's language. */
  pages: () => { success: OAuthPage; failure: OAuthPage };
  /** The sign-in changed: started, finished, signed out, or expired. */
  onChange: () => void;
}) {
  const { endpoints, secrets, settings, now, onChange } = options;

  /** Bumped by every sign-in and sign-out, so a refresh that finishes afterwards changes nothing. */
  let generation = 0;
  let refreshing: Promise<StoredTokens> | null = null;
  let running: { controller: AbortController; done: Promise<unknown> } | null = null;

  const load = async () => parseStored(await secrets.tryGet(TOKENS_SECRET));

  const expire = async (from: number) => {
    if (from !== generation) return;
    generation += 1;
    await secrets.delete(TOKENS_SECRET);
    settings.writeDeviceValue(EXPIRED_FLAG, true);
    onChange();
  };

  /** One refresh at a time: everyone who needs a fresh token while it runs waits for it. */
  const refresh = (current: StoredTokens): Promise<StoredTokens> => {
    if (refreshing) return refreshing;
    const from = generation;
    const run = (async () => {
      if (!current.refreshToken) {
        await expire(from);
        throw new ChatGptSignInRequiredError("expired");
      }
      let fresh: OAuthTokens;
      try {
        fresh = await refreshOAuthTokens({
          tokenUrl: endpoints.tokenUrl,
          clientId: CODEX_CLIENT_ID,
          refreshToken: current.refreshToken,
        });
      } catch (error) {
        if (!isRefusedRefresh(error)) throw error;
        await expire(from);
        throw new ChatGptSignInRequiredError("expired");
      }
      if (from !== generation) throw new ChatGptSignInRequiredError("signed-out");
      const next = toStored(fresh, current, now());
      await secrets.set(TOKENS_SECRET, JSON.stringify(next));
      return next;
    })();
    refreshing = run;
    const clear = () => {
      if (refreshing === run) refreshing = null;
    };
    run.then(clear, clear);
    return run;
  };

  const credentials: ChatGptCredentials = {
    async access(request = {}) {
      // Join a refresh already under way: its result replaces what is stored.
      const current = refreshing ? await refreshing : await load();
      if (!current) {
        throw new ChatGptSignInRequiredError(
          settings.readDeviceValue(EXPIRED_FLAG) === true ? "expired" : "signed-out",
        );
      }
      const stale =
        current.expiresAt - now() <= REFRESH_WINDOW_MS ||
        (request.rejected !== undefined && request.rejected === current.accessToken);
      const usable = stale ? await refresh(current) : current;
      return { accessToken: usable.accessToken, accountId: usable.accountId };
    },
  };

  const cancel = async () => {
    const current = running;
    if (!current) return;
    current.controller.abort();
    await current.done.catch(() => undefined);
  };

  return {
    credentials,

    async account(): Promise<ChatGptAccount> {
      const stored = await load();
      if (stored) return { state: "signed-in", email: stored.email, plan: stored.plan };
      return settings.readDeviceValue(EXPIRED_FLAG) === true
        ? { state: "expired" }
        : { state: "signed-out" };
    },

    hasSignIn: async () => (await load()) !== null,

    signingIn: () => running !== null,

    /**
     * Opens the browser and waits for the User. Starting again cancels a
     * sign-in still waiting, which frees the port first. Rejects with
     * `OAuthFlowError`, `OAuthTokenError` or `SecretStorageError`.
     */
    async signIn(): Promise<void> {
      const storage = secrets.status();
      // Check before the User signs in, not after: the tokens would have nowhere to go.
      if (!storage.canSave) throw new SecretStorageError(storage.protection);
      await cancel();
      const controller = new AbortController();
      const done = (async () => {
        const tokens = await authorizeWithLoopback({
          authorizeUrl: endpoints.authorizeUrl,
          tokenUrl: endpoints.tokenUrl,
          clientId: CODEX_CLIENT_ID,
          scope: SCOPE,
          port: endpoints.callbackPort,
          path: CODEX_CALLBACK_PATH,
          authorizeParams: AUTHORIZE_PARAMS,
          openBrowser: (url) => options.browser.open(url),
          pages: options.pages(),
          timeoutMs: endpoints.signInTimeoutMs,
          signal: controller.signal,
        });
        const stored = toStored(tokens, null, now());
        generation += 1;
        await secrets.set(TOKENS_SECRET, JSON.stringify(stored));
        settings.writeDeviceValue(EXPIRED_FLAG, false);
      })();
      running = { controller, done };
      onChange();
      try {
        await done;
      } finally {
        if (running?.controller === controller) running = null;
        onChange();
      }
    },

    cancelSignIn: cancel,

    /** Deletes the tokens. Safe to call when signed out. */
    async signOut(): Promise<void> {
      await cancel();
      generation += 1;
      await secrets.delete(TOKENS_SECRET);
      settings.writeDeviceValue(EXPIRED_FLAG, false);
      onChange();
    },
  };
}
