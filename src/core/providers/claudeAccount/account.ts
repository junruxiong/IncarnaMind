/**
 * Sign in with a Claude account: Anthropic's OAuth with the client, redirect
 * and token requests the route's profile names (./config), kept as
 * `../../oauthAccount` keeps a sign-in. Signing in doesn't choose the route;
 * saving a "claude-account" chat provider does, and that choice allows its
 * flows (ADR-0022).
 */
import { translate } from "../../../shared/i18n";
import type { Browser } from "../../adapters";
import type {
  ClaudeAccountStatus,
  ClaudeSignInErrorKind,
  ClaudeSignInResult,
  PastedAddressResult,
} from "../../api";
import { SecretStorageError } from "../../errors";
import type { Language } from "../../language";
import {
  DEFAULT_SIGN_IN_TIMEOUT_MS,
  OAuthFlowError,
  OAuthTokenError,
  refreshOAuthTokens,
} from "../../oauth";
import {
  createOAuthAccount,
  type OAuthAccountRecord,
  parseAccountRecord,
  pastedAddressResult,
  revokeRefreshToken,
} from "../../oauthAccount";
import type { Secrets } from "../../secrets";
import { ClaudeAccountError } from "../providerErrors";
import type { AnthropicOAuthConfig } from "./config";

const ACCOUNT_SECRET = "claude:account";

function signInError(error: unknown): { kind: ClaudeSignInErrorKind; message: string } {
  const message = error instanceof Error ? error.message : String(error);
  if (error instanceof SecretStorageError) return { kind: "secret-storage", message };
  if (error instanceof OAuthFlowError) {
    if (error.reason === "cancelled" || error.reason === "timed-out") {
      return { kind: "sign-in-cancelled", message };
    }
    return { kind: error.reason === "denied" ? "refused" : "failed", message };
  }
  if (error instanceof OAuthTokenError) {
    return { kind: error.status === null ? "unexpected-response" : "refused", message };
  }
  return { kind: "failed", message };
}

export type ClaudeSignIn = ReturnType<typeof createClaudeAccount>;

export function createClaudeAccount(options: {
  /** The route's client profile; null where this build has none, and the route isn't offered. */
  config: Readonly<AnthropicOAuthConfig> | null;
  secrets: Secrets;
  /** The User's language, for the browser tab's pages. */
  language: () => Language;
  browser: Browser;
  /** Milliseconds since the epoch. */
  now: () => number;
  /** The sign-in changed: started, finished, signed out, or expired. */
  onChange: () => void;
  fetch?: typeof globalThis.fetch;
}) {
  const { config } = options;
  const fetcher = options.fetch && { fetch: options.fetch };
  const format = config?.tokenRequest?.format ?? "form";
  const revokeUrl = config?.revokeUrl;

  const account = createOAuthAccount<OAuthAccountRecord>({
    secretName: ACCOUNT_SECRET,
    secrets: options.secrets,
    parse: parseAccountRecord,
    browser: options.browser,
    pages: () => {
      const language = options.language();
      return {
        success: {
          title: translate(language, "claudeAccount.page.success.title"),
          body: translate(language, "claudeAccount.page.success.body"),
        },
        failure: {
          title: translate(language, "claudeAccount.page.failure.title"),
          body: translate(language, "claudeAccount.page.failure.body"),
        },
      };
    },
    signInTimeoutMs: config?.signInTimeoutMs ?? DEFAULT_SIGN_IN_TIMEOUT_MS,
    now: options.now,
    renew(_stored, refreshToken, signal) {
      if (!config)
        throw new ClaudeAccountError("signed-out", "This build doesn't offer Claude sign-in.");
      return refreshOAuthTokens({
        tokenUrl: config.tokenUrl,
        clientId: config.clientId,
        refreshToken,
        format,
        signal,
        ...fetcher,
      });
    },
    ...(config &&
      revokeUrl && {
        revoke: (_stored: OAuthAccountRecord, refreshToken: string) =>
          revokeRefreshToken({
            revokeUrl,
            clientId: config.clientId,
            refreshToken,
            provider: "Anthropic",
            ...fetcher,
          }),
      }),
    signInRequired: (reason) =>
      reason === "expired"
        ? new ClaudeAccountError("sign-in-expired", "Your Claude sign-in expired. Sign in again.")
        : new ClaudeAccountError("signed-out", "Sign in with your Claude account first."),
    onChange: options.onChange,
    ...fetcher,
  });

  const hasSignIn = async () => config !== null && (await account.load())?.tokens != null;

  const status = async (): Promise<ClaudeAccountStatus> => {
    // Read before awaiting the keychain, so the status describes the moment it was asked for.
    const signingIn = account.signingIn();
    const stored = config ? await account.load() : null;
    const state = stored?.tokens
      ? "signed-in"
      : stored?.ended === "expired"
        ? "expired"
        : "signed-out";
    return {
      available: config !== null,
      experimental: config?.experimental === true,
      account: { state },
      signingIn,
    };
  };

  return {
    status,
    credentials: account.credentials,
    apiBaseUrl: config?.apiBaseUrl ?? null,
    request: config?.request ?? {},
    hasSignIn,
    /** Whether the sign-in's traffic can happen on this device: it has signed in, or is signing in. */
    used: async () => account.signingIn() || (await hasSignIn()),

    /**
     * Opens Anthropic's sign-in in the browser and waits until the User
     * finishes there, or pastes what the page shows. Starting again cancels
     * a sign-in still waiting.
     */
    async signIn(): Promise<ClaudeSignInResult> {
      if (!config) {
        return {
          ok: false,
          error: { kind: "unavailable", message: "This build doesn't offer Claude sign-in." },
        };
      }
      try {
        await account.signIn(async (authorize) => {
          const tokens = await authorize({
            authorizeUrl: config.authorizeUrl,
            tokenUrl: config.tokenUrl,
            clientId: config.clientId,
            scope: config.scopes.join(" "),
            redirect: config.redirect,
            ...(config.authorizeParams && { authorizeParams: config.authorizeParams }),
            tokenFormat: format,
            stateInExchange: config.tokenRequest?.stateInExchange ?? false,
            ...(config.stateBytes !== undefined && { stateBytes: config.stateBytes }),
          });
          await account.commit({ tokens: account.storedTokens(tokens, null), ended: null });
        });
        return { ok: true, status: await status() };
      } catch (error) {
        return { ok: false, error: signInError(error) };
      }
    },

    cancelSignIn: account.cancelSignIn,

    /** Forgets the tokens on this device, and ends the session at Anthropic where the profile says how. */
    async signOut(): Promise<ClaudeAccountStatus> {
      await account.signOut();
      return status();
    },

    pasteSignInAddress: (address: string): PastedAddressResult =>
      pastedAddressResult(account.pasteAddress(address)),
  };
}
