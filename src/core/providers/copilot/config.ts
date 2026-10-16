/** The only place that names GitHub's side of the route, so a change there touches only this file. */

import type { CopilotProfileId } from "../../api";
import { EXPERIMENTAL_SIGN_IN_ROUTES } from "../../buildFlags";
import type { TokenRequestFormat } from "../../oauth";

export interface CopilotProfile {
  clientId: string;
  deviceCodeUrl: string;
  /** Polled during sign-in; also where a refresh token is traded. */
  tokenUrl: string;
  scopes: readonly string[];
  format: TokenRequestFormat;
  signInHeaders?: Readonly<Record<string, string>>;
  /** The Copilot API's base; requests go to `<base>/chat/completions`, the model list to `<base>/models`. */
  apiBaseUrl: string;
  /** Headers every Copilot request carries, set after IncarnaMind's own. */
  requestHeaders?: Readonly<Record<string, string>>;
  /**
   * Where the GitHub token is traded for a short-lived Copilot token before
   * requests, and traded again when that expires; none sends the GitHub token itself.
   */
  tokenExchange?: { url: string; headers?: Readonly<Record<string, string>> };
  /** An experimental sign-in route, which the User is told about before signing in. */
  experimental?: boolean;
}

/** IncarnaMind's own GitHub OAuth App, signing in with the device flow. */
const OWN: Readonly<CopilotProfile> = {
  clientId: "Ov23liqRRjAsHYMWax18",
  deviceCodeUrl: "https://github.com/login/device/code",
  tokenUrl: "https://github.com/login/oauth/access_token",
  // `offline_access` asks for a token that expires, with a refresh token.
  scopes: ["read:user", "offline_access"],
  format: "json",
  apiBaseUrl: "https://api.githubcopilot.com",
};

const EDITOR_HEADERS = {
  "User-Agent": "GitHubCopilotChat/0.26.7",
  "Editor-Version": "vscode/1.99.3",
  "Editor-Plugin-Version": "copilot-chat/0.26.7",
  "Copilot-Integration-Id": "vscode-chat",
} as const;

const EDITOR: Readonly<CopilotProfile> | null = EXPERIMENTAL_SIGN_IN_ROUTES
  ? {
      experimental: true,
      clientId: "Iv1.b507a08c87ecfe98",
      deviceCodeUrl: "https://github.com/login/device/code",
      tokenUrl: "https://github.com/login/oauth/access_token",
      scopes: ["read:user"],
      format: "json",
      signInHeaders: { "User-Agent": EDITOR_HEADERS["User-Agent"] },
      apiBaseUrl: "https://api.githubcopilot.com",
      requestHeaders: EDITOR_HEADERS,
      tokenExchange: {
        url: "https://api.github.com/copilot_internal/v2/token",
        headers: EDITOR_HEADERS,
      },
    }
  : null;

export const COPILOT_PROFILES: Readonly<
  Partial<Record<CopilotProfileId, Readonly<CopilotProfile>>>
> = {
  own: OWN,
  ...(EDITOR && { editor: EDITOR }),
};
