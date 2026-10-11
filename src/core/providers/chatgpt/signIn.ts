/**
 * Sign in with ChatGPT, OpenAI's documented flow for open-source apps that
 * run on the User's computer to use the User's ChatGPT plan
 * (https://developers.openai.com/siwc/token-sharing-open-source/sign-in).
 *
 * - The first sign-in registers IncarnaMind with the User's ChatGPT account:
 *   `client_id=dynamic_agent_client`, the app's name as `agent_name_hint`,
 *   and a host ID chosen once for this computer (`ext_agent_host_id`). The
 *   browser comes back with the client ID OpenAI issued, which every later
 *   sign-in and refresh uses. No client secret, and no other app's client.
 * - PKCE, a nonce and a loopback redirect on 127.0.0.1 at any free port.
 * - The ID token's signature, issuer, audience, expiry and nonce are checked,
 *   and the granted scopes must include `chatgpt.tokens.use.direct`: without
 *   it the User didn't allow plan use, and the next sign-in asks again.
 * - Tokens live only in the secret store, as one record with the
 *   registration. An access token is refreshed when it is within five
 *   minutes of expiring; refresh tokens rotate, so refreshes are serialised,
 *   across processes too when the store is shared (see `RefreshLock`).
 * - Signing out revokes the refresh token at OpenAI, then forgets the tokens
 *   but keeps the registration, so the next sign-in reuses the same client.
 *
 * Written from OpenAI's documentation; no code from OpenAI's devkit, whose
 * licence is non-commercial.
 */
import { randomBytes, randomUUID } from "node:crypto";
import type { Browser } from "../../adapters";
import { SecretStorageError } from "../../errors";
import {
  authorizeWithLoopback,
  type OAuthPage,
  OAuthTokenError,
  type OAuthTokens,
  refreshOAuthTokens,
  revokeOAuthToken,
} from "../../oauth";
import type { Secrets } from "../../secrets";
import {
  ChatGptOtherAccountError,
  ChatGptPlanNotAllowedError,
  ChatGptSignInRequiredError,
} from "./errors";
import { createJwks, verifyIdToken } from "./idToken";

/** OpenAI's production OpenID configuration (https://auth.openai.com/.well-known/openid-configuration). */
export const OPENAI_AUTH = {
  issuer: "https://auth.openai.com",
  authorizeUrl: "https://auth.openai.com/api/accounts/authorize",
  tokenUrl: "https://auth.openai.com/api/accounts/oauth/token",
  revokeUrl: "https://auth.openai.com/api/accounts/oauth/revoke",
  jwksUrl: "https://auth.openai.com/.well-known/jwks.json",
} as const;

/** The API the tokens are for (the OAuth `resource`), whatever URL tests send requests to. */
export const CHATGPT_PLAN_RESOURCE = "https://api.openai.com/v1";

/** The client ID for a first sign-in: OpenAI issues IncarnaMind's own in return. */
const DYNAMIC_CLIENT = "dynamic_agent_client";

/** What the User sees on OpenAI's page as the app asking; used the same way in every installation. */
const APP_NAME = "IncarnaMind";

const CALLBACK_PATH = "/auth/callback";

const IDENTITY_SCOPES = "openid profile email";
/** The permission to use the User's ChatGPT plan; without it nothing can be asked. */
const PLAN_SCOPE = "chatgpt.tokens.use.direct";
const SCOPE = `${IDENTITY_SCOPES} offline_access resource.invoke ${PLAN_SCOPE}`;

/** Refresh this long before the access token expires. */
const REFRESH_WINDOW_MS = 5 * 60_000;

/** When neither the server nor the token says how long it lasts; OpenAI's are an hour. */
const ASSUMED_LIFETIME_MS = 60 * 60_000;

/** Waits between tries to revoke the session at sign-out, when OpenAI can't be reached. */
const REVOKE_RETRIES_MS = [250, 1000];

const ACCOUNT_SECRET = "chatgpt:account";

/** Refresh-token errors that mean the token can't be used again (OpenAI's errors and recovery). */
const UNUSABLE_REFRESH = new Set([
  "invalid_grant",
  "invalid_refresh_token",
  "token_expired",
  "refresh_token_expired",
  "refresh_token_invalidated",
  "refresh_token_reused",
]);

/** Where the sign-in happens. Tests point these at a local fake server. */
export interface ChatGptSignInEndpoints {
  issuer: string;
  authorizeUrl: string;
  tokenUrl: string;
  revokeUrl: string;
  jwksUrl: string;
  signInTimeoutMs: number;
}

/**
 * Runs a refresh so that no other process sharing the same store refreshes
 * at the same time. Within one process, refreshes already run one at a time.
 */
export interface RefreshLock {
  run<T>(task: () => Promise<T>): Promise<T>;
}

const NO_LOCK: RefreshLock = { run: (task) => task() };

interface StoredTokens {
  accessToken: string;
  refreshToken: string | null;
  /** Milliseconds since the epoch. */
  expiresAt: number;
}

/** Everything about the ChatGPT sign-in on this device, kept as one secret. */
interface StoredAccount {
  /** This computer's `ext_agent_host_id`, chosen before its first sign-in and kept. */
  hostId: string;
  /** The client OpenAI issued at the first sign-in; null until then. */
  clientId: string | null;
  /** The account the client was registered with (the ID token's `sub`). */
  subject: string | null;
  email: string | null;
  /** The last sign-in's ID token, sent back as `id_token_hint`; dropped at sign-out. */
  idToken: string | null;
  tokens: StoredTokens | null;
  /** Why there are no tokens: the User signed out, or a refresh failed. */
  ended: "signed-out" | "expired" | null;
  /** The last sign-in didn't allow plan use, so the next one asks for consent again. */
  planRefused: boolean;
}

const text = (value: unknown) => (typeof value === "string" && value !== "" ? value : null);

function parseStored(json: string | null): StoredAccount | null {
  if (!json) return null;
  try {
    const value: unknown = JSON.parse(json);
    if (typeof value !== "object" || value === null) return null;
    const record = value as Record<string, unknown>;
    const hostId = text(record.hostId);
    if (!hostId) return null;
    const tokens = record.tokens as Record<string, unknown> | null | undefined;
    const accessToken = tokens ? text(tokens.accessToken) : null;
    return {
      hostId,
      clientId: text(record.clientId),
      subject: text(record.subject),
      email: text(record.email),
      idToken: text(record.idToken),
      tokens:
        tokens && accessToken && typeof tokens.expiresAt === "number"
          ? {
              accessToken,
              refreshToken: text(tokens.refreshToken),
              expiresAt: tokens.expiresAt,
            }
          : null,
      ended:
        record.ended === "expired"
          ? "expired"
          : record.ended === "signed-out"
            ? "signed-out"
            : null,
      planRefused: record.planRefused === true,
    };
  } catch {
    return null;
  }
}

function toTokens(tokens: OAuthTokens, previous: StoredTokens | null, now: number): StoredTokens {
  return {
    accessToken: tokens.accessToken,
    // A refresh that doesn't rotate the refresh token keeps the old one.
    refreshToken: tokens.refreshToken ?? previous?.refreshToken ?? null,
    expiresAt: now + (tokens.expiresIn ?? ASSUMED_LIFETIME_MS / 1000) * 1000,
  };
}

/** What a refresh refusal means: the token is unusable, the client is gone, or it may pass. */
function refusal(error: unknown): "unusable" | "client-gone" | "passing" {
  if (!(error instanceof OAuthTokenError) || error.status === null) return "passing";
  if (error.code === "invalid_client") return "client-gone";
  if (error.code && UNUSABLE_REFRESH.has(error.code)) return "unusable";
  // A refusal without a known code: only an overloaded or slow server is worth waiting for.
  const { status } = error;
  return status >= 400 && status < 500 && status !== 408 && status !== 429 ? "unusable" : "passing";
}

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

export type ChatGptAccountState =
  | { state: "signed-out" }
  | { state: "expired" }
  | { state: "signed-in"; email: string | null };

export interface ChatGptCredentials {
  /**
   * A current access token, refreshed first when it is about to expire.
   * `rejected` is a token the API refused: if it is still the current one, it
   * is refreshed even so. Throws `ChatGptSignInRequiredError` when there is no
   * sign-in or it can't be refreshed.
   */
  access(options?: { rejected?: string }): Promise<{ accessToken: string }>;
}

export function createChatGptSignIn(options: {
  endpoints: ChatGptSignInEndpoints;
  /** Where the sign-in is kept: the device's secret store, or a shared one (see `RefreshLock`). */
  secrets: Secrets;
  refreshLock?: RefreshLock;
  browser: Browser;
  /** Milliseconds since the epoch. */
  now: () => number;
  /** The browser tab's pages, in the User's language. */
  pages: () => { success: OAuthPage; failure: OAuthPage };
  /** The sign-in changed: started, finished, signed out, or expired. */
  onChange: () => void;
  fetch?: typeof globalThis.fetch;
}) {
  const { endpoints, secrets, now, onChange } = options;
  const lock = options.refreshLock ?? NO_LOCK;
  const jwks = createJwks(endpoints.jwksUrl, options.fetch);

  /** Bumped by every sign-in and sign-out, so a refresh that finishes afterwards changes nothing. */
  let generation = 0;
  let refreshing: Promise<StoredAccount> | null = null;
  let running: { controller: AbortController; done: Promise<unknown> } | null = null;
  let signingOut = false;

  const load = async () => parseStored(await secrets.tryGet(ACCOUNT_SECRET));
  const save = (account: StoredAccount) => secrets.set(ACCOUNT_SECRET, JSON.stringify(account));

  const expire = async (from: number, account: StoredAccount, clientGone: boolean) => {
    if (from !== generation) return;
    generation += 1;
    await save({
      ...account,
      // A client OpenAI no longer knows can't be signed in with again: the next sign-in registers anew.
      ...(clientGone && { clientId: null, subject: null, idToken: null }),
      tokens: null,
      ended: "expired",
    });
    onChange();
  };

  const isStale = (tokens: StoredTokens) => tokens.expiresAt - now() <= REFRESH_WINDOW_MS;

  /** One refresh at a time: everyone who needs a fresh token while it runs waits for it. */
  const refresh = (current: StoredAccount, rejected?: string): Promise<StoredAccount> => {
    if (refreshing) return refreshing;
    const from = generation;
    const run = lock.run(async () => {
      // Another process sharing the store may have refreshed while this one waited for the lock.
      const latest = (await load()) ?? current;
      const tokens = latest.tokens;
      if (!tokens || !latest.clientId) {
        throw new ChatGptSignInRequiredError(latest.ended === "expired" ? "expired" : "signed-out");
      }
      const replaced = tokens.accessToken !== current.tokens?.accessToken;
      if (replaced && !isStale(tokens) && tokens.accessToken !== rejected) return latest;
      if (!tokens.refreshToken) {
        await expire(from, latest, false);
        throw new ChatGptSignInRequiredError("expired");
      }
      let fresh: OAuthTokens;
      try {
        fresh = await refreshOAuthTokens({
          tokenUrl: endpoints.tokenUrl,
          clientId: latest.clientId,
          refreshToken: tokens.refreshToken,
          // The grant keeps its scopes: `scope` is left out, as OpenAI asks.
          params: { resource: CHATGPT_PLAN_RESOURCE },
          ...(options.fetch && { fetch: options.fetch }),
        });
      } catch (error) {
        const meaning = refusal(error);
        if (meaning === "passing") throw error;
        await expire(from, latest, meaning === "client-gone");
        throw new ChatGptSignInRequiredError("expired");
      }
      if (from !== generation) throw new ChatGptSignInRequiredError("signed-out");
      const next: StoredAccount = { ...latest, tokens: toTokens(fresh, tokens, now()) };
      // The access token, its expiry and the rotated refresh token are replaced together.
      await save(next);
      return next;
    });
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
      if (signingOut || !current?.tokens) {
        throw new ChatGptSignInRequiredError(
          current?.ended === "expired" ? "expired" : "signed-out",
        );
      }
      const stale =
        isStale(current.tokens) ||
        (request.rejected !== undefined && request.rejected === current.tokens.accessToken);
      const usable = stale ? await refresh(current, request.rejected) : current;
      if (!usable.tokens) throw new ChatGptSignInRequiredError("signed-out");
      return { accessToken: usable.tokens.accessToken };
    },
  };

  const cancel = async () => {
    const current = running;
    if (!current) return;
    current.controller.abort();
    await current.done.catch(() => undefined);
  };

  /** Ends the session at OpenAI. True once OpenAI confirms; false if it couldn't be confirmed. */
  const revoke = async (clientId: string, refreshToken: string): Promise<boolean> => {
    for (let attempt = 0; ; attempt += 1) {
      try {
        await revokeOAuthToken({
          revokeUrl: endpoints.revokeUrl,
          clientId,
          token: refreshToken,
          tokenTypeHint: "refresh_token",
          ...(options.fetch && { fetch: options.fetch }),
        });
        return true;
      } catch (error) {
        // A refusal won't change by asking again; a server or network failure might.
        const passing =
          !(error instanceof OAuthTokenError) || error.status === null || error.status >= 500;
        const wait = REVOKE_RETRIES_MS[attempt];
        if (!passing || wait === undefined) {
          console.error("The ChatGPT sign-out couldn't be confirmed by OpenAI.", error);
          return false;
        }
        await sleep(wait);
      }
    }
  };

  /** The record to sign in with, a host ID chosen and kept before the first sign-in. */
  const prepare = async (): Promise<StoredAccount> => {
    const existing = await load();
    if (existing) return existing;
    const fresh: StoredAccount = {
      hostId: `urn:uuid:${randomUUID()}`,
      clientId: null,
      subject: null,
      email: null,
      idToken: null,
      tokens: null,
      ended: null,
      planRefused: false,
    };
    await save(fresh);
    return fresh;
  };

  const signInOnce = async (signal: AbortSignal, anotherAccount: boolean) => {
    const account = await prepare();
    const registered = anotherAccount ? null : account.clientId;
    const nonce = randomBytes(24).toString("base64url");
    /** The client the code is for: the registered one, or the one OpenAI issues now. */
    let clientId = registered;
    let tokens: OAuthTokens;
    try {
      tokens = await authorizeWithLoopback({
        authorizeUrl: endpoints.authorizeUrl,
        tokenUrl: endpoints.tokenUrl,
        clientId: registered ?? DYNAMIC_CLIENT,
        scope: SCOPE,
        port: 0,
        path: CALLBACK_PATH,
        authorizeParams: {
          resource: CHATGPT_PLAN_RESOURCE,
          nonce,
          ext_agent_host_id: account.hostId,
          ...(registered
            ? {
                ...(account.email && { login_hint: account.email }),
                ...(account.idToken && account.ended !== "signed-out"
                  ? { id_token_hint: account.idToken }
                  : {}),
              }
            : { agent_name_hint: APP_NAME }),
          // Asked again only after the User didn't allow plan use: not on every sign-in.
          ...(account.planRefused && registered && { prompt: "consent" }),
        },
        tokenParams: { resource: CHATGPT_PLAN_RESOURCE },
        verify(params) {
          if (!params.has("code")) return;
          const returned = params.get("client_id");
          if (registered) {
            if (returned && returned !== registered) {
              throw new Error("OpenAI answered for another registration of IncarnaMind.");
            }
          } else if (!returned || returned === DYNAMIC_CLIENT) {
            throw new Error("OpenAI didn't finish registering IncarnaMind. Try again.");
          } else {
            clientId = returned;
          }
        },
        exchangeClientId: () => clientId ?? DYNAMIC_CLIENT,
        openBrowser: (url) => options.browser.open(url),
        pages: options.pages(),
        timeoutMs: endpoints.signInTimeoutMs,
        signal,
        ...(options.fetch && { fetch: options.fetch }),
      });
    } catch (error) {
      // A registration that went through stays, so the next attempt reuses its client.
      if (!registered && clientId) {
        await save({ ...account, clientId, subject: null, idToken: null });
      }
      throw error;
    }
    const issued = clientId ?? DYNAMIC_CLIENT;
    if (!tokens.idToken) throw new Error("OpenAI's sign-in returned no ID token.");
    const claims = await verifyIdToken({
      token: tokens.idToken,
      jwks,
      issuer: endpoints.issuer,
      audience: issued,
      nonce,
      now: now(),
      signal,
    });
    const scopes = (tokens.scope ?? "").split(/\s+/);
    const abandon = async () => {
      if (tokens.refreshToken) await revoke(issued, tokens.refreshToken);
    };
    if (registered && account.subject && claims.subject !== account.subject) {
      await abandon();
      throw new ChatGptOtherAccountError(account.email);
    }
    const signedIn: StoredAccount = {
      hostId: account.hostId,
      clientId: issued,
      subject: claims.subject,
      email: claims.email ?? account.email,
      idToken: tokens.idToken,
      tokens: null,
      ended: "signed-out",
      planRefused: false,
    };
    if (!scopes.includes(PLAN_SCOPE)) {
      // Signed in, but not allowed to use the plan: nothing IncarnaMind could use is kept.
      await abandon();
      generation += 1;
      await save({ ...signedIn, planRefused: true });
      throw new ChatGptPlanNotAllowedError();
    }
    generation += 1;
    await save({ ...signedIn, tokens: toTokens(tokens, null, now()), ended: null });
    return { firstTime: registered === null };
  };

  return {
    credentials,

    async account(): Promise<ChatGptAccountState> {
      const stored = await load();
      if (stored?.tokens) return { state: "signed-in", email: stored.email };
      return stored?.ended === "expired" ? { state: "expired" } : { state: "signed-out" };
    },

    hasSignIn: async () => (await load())?.tokens != null,

    /** Whether this device has signed in with ChatGPT, or is signing in: its traffic can happen. */
    used: async () => running !== null || (await load())?.clientId != null,

    signingIn: () => running !== null,

    /**
     * Opens the browser and waits for the User. Starting again cancels a
     * sign-in still waiting. `anotherAccount` registers a new client instead
     * of signing in to the one this device has. Rejects with `OAuthFlowError`,
     * `OAuthTokenError`, `IdTokenError`, `SecretStorageError`,
     * `ChatGptPlanNotAllowedError` or `ChatGptOtherAccountError`.
     */
    async signIn(
      signInOptions: { anotherAccount?: boolean } = {},
    ): Promise<{ firstTime: boolean }> {
      const storage = secrets.status();
      // Check before the User signs in, not after: the tokens would have nowhere to go.
      if (!storage.canSave) throw new SecretStorageError(storage.protection);
      await cancel();
      const controller = new AbortController();
      const done = signInOnce(controller.signal, signInOptions.anotherAccount === true);
      running = { controller, done };
      onChange();
      try {
        return await done;
      } finally {
        if (running?.controller === controller) running = null;
        onChange();
      }
    },

    cancelSignIn: cancel,

    /**
     * Stops using the sign-in, ends its session at OpenAI and forgets its
     * tokens; the registration stays for the next sign-in. `revoked` is false
     * when OpenAI couldn't confirm the session ended. Safe when signed out.
     */
    async signOut(): Promise<{ revoked: boolean }> {
      await cancel();
      generation += 1;
      signingOut = true;
      try {
        const stored = await load();
        if (!stored?.tokens) return { revoked: true };
        const { clientId } = stored;
        const { refreshToken } = stored.tokens;
        const revoked = clientId && refreshToken ? await revoke(clientId, refreshToken) : true;
        await save({ ...stored, tokens: null, idToken: null, ended: "signed-out" });
        return { revoked };
      } finally {
        signingOut = false;
        onChange();
      }
    },
  };
}

export type ChatGptSignIn = ReturnType<typeof createChatGptSignIn>;
