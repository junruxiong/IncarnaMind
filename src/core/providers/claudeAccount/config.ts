/** The only place that names Anthropic's side of the route, so a change there touches only this file. */
import { EXPERIMENTAL_SIGN_IN_ROUTES } from "../../buildFlags";
import type { RedirectShape, TokenRequestFormat } from "../../oauth";
import type { RequestTransform } from "../signedFetch";

export interface AnthropicOAuthConfig {
  clientId: string;
  authorizeUrl: string;
  tokenUrl: string;
  /** Ends the session at sign-out (RFC 7009); without one, signing out forgets the tokens here. */
  revokeUrl?: string;
  scopes: readonly string[];
  authorizeParams?: Readonly<Record<string, string>>;
  redirect: RedirectShape;
  /** How the token endpoint takes a code exchange or a refresh. Defaults to a form, without the state. */
  tokenRequest?: { format: TokenRequestFormat; stateInExchange: boolean };
  /** The Messages API's base; requests go to `<base>/messages`. */
  apiBaseUrl: string;
  request?: RequestTransform;
  /** Defaults to five minutes. */
  signInTimeoutMs?: number;
  stateBytes?: number;
  /** An experimental sign-in route, which the User is told about before signing in. */
  experimental?: boolean;
}

const SUBSCRIPTION_SYSTEM_PREFIX = "You are Claude Code, Anthropic's official CLI for Claude.";

export const ANTHROPIC_OAUTH: Readonly<AnthropicOAuthConfig> | null = EXPERIMENTAL_SIGN_IN_ROUTES
  ? {
      experimental: true,
      clientId: "9d1c250a-e61b-44d9-88ed-5944d1962f5e",
      authorizeUrl: "https://claude.ai/oauth/authorize",
      tokenUrl: "https://platform.claude.com/v1/oauth/token",
      scopes: ["user:profile", "user:inference"],
      // Asks Anthropic's page to show the code for pasting.
      authorizeParams: { code: "true" },
      redirect: { kind: "page", uri: "https://platform.claude.com/oauth/code/callback" },
      tokenRequest: { format: "json", stateInExchange: true },
      // Its authorization server refuses a shorter state.
      stateBytes: 32,
      apiBaseUrl: "https://api.anthropic.com/v1",
      request: {
        systemBlocks: [SUBSCRIPTION_SYSTEM_PREFIX],
        headers: { "anthropic-beta": "oauth-2025-04-20" },
      },
    }
  : null;
