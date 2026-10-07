/**
 * The experimental "ChatGPT plan (via Codex sign-in)" provider: the switch
 * that turns it on (per device, off by default), the sign-in, and its status.
 * Chat providers (`../chat.ts`) use it for readiness, the connection test and
 * the model factory; the endpoint adapter does the requests.
 */
import { translate } from "../../../shared/i18n";
import type { Browser } from "../../adapters";
import type { ChatGptPlanStatus, ChatGptSignInErrorKind, ChatGptSignInResult } from "../../api";
import { InvalidInputError, SecretStorageError } from "../../errors";
import { DEFAULT_SIGN_IN_TIMEOUT_MS, OAuthFlowError } from "../../oauth";
import type { Secrets } from "../../secrets";
import type { SettingsStore } from "../../settings";
import { CHATGPT_PLAN_MODELS, CODEX_BASE_URL } from "./codexEndpoint";
import {
  CODEX_AUTHORIZE_URL,
  CODEX_CALLBACK_PORT,
  CODEX_TOKEN_URL,
  type CodexSignInEndpoints,
  createCodexSignIn,
} from "./codexSignIn";

/** Where the provider signs in and sends Questions. */
export interface ChatGptPlanEndpoints extends CodexSignInEndpoints {
  /** The model endpoint's base URL; requests go to `<base>/responses`. */
  codexBaseUrl: string;
}

export const CHATGPT_PLAN_ENDPOINTS: Readonly<ChatGptPlanEndpoints> = {
  authorizeUrl: CODEX_AUTHORIZE_URL,
  tokenUrl: CODEX_TOKEN_URL,
  callbackPort: CODEX_CALLBACK_PORT,
  signInTimeoutMs: DEFAULT_SIGN_IN_TIMEOUT_MS,
  codexBaseUrl: CODEX_BASE_URL,
};

/** Per-device: the User turned the experimental provider on. */
const ENABLED = "chatGptPlanEnabled";

function signInError(error: unknown): { kind: ChatGptSignInErrorKind; message: string } {
  const message = error instanceof Error ? error.message : String(error);
  if (error instanceof OAuthFlowError) return { kind: error.reason, message };
  if (error instanceof SecretStorageError) return { kind: "secret-storage", message };
  return { kind: "failed", message };
}

export type ChatGptPlan = ReturnType<typeof createChatGptPlan>;

export function createChatGptPlan(options: {
  endpoints: ChatGptPlanEndpoints;
  secrets: Secrets;
  settings: SettingsStore;
  browser: Browser;
  /** Milliseconds since the epoch. */
  now: () => number;
  /** The provider was turned on or off, or its sign-in changed. */
  onChange: () => void;
}) {
  const { endpoints, settings, onChange } = options;

  const signIn = createCodexSignIn({
    endpoints,
    secrets: options.secrets,
    settings,
    browser: options.browser,
    now: options.now,
    onChange,
    pages: () => {
      const { language } = settings.get();
      return {
        success: {
          title: translate(language, "codex.page.success.title"),
          body: translate(language, "codex.page.success.body"),
        },
        failure: {
          title: translate(language, "codex.page.failure.title"),
          body: translate(language, "codex.page.failure.body"),
        },
      };
    },
  });

  const enabled = () => settings.readDeviceValue(ENABLED) === true;

  const status = async (): Promise<ChatGptPlanStatus> => {
    // Read before awaiting the keychain, so the status describes the moment it was asked for.
    const on = enabled();
    const signingIn = signIn.signingIn();
    return {
      enabled: on,
      account: await signIn.account(),
      signingIn,
      models: CHATGPT_PLAN_MODELS.map((model) => ({ ...model })),
    };
  };

  return {
    enabled,
    status,
    credentials: signIn.credentials,
    codexBaseUrl: endpoints.codexBaseUrl,
    hasSignIn: signIn.hasSignIn,

    /** Turning it off signs out, so no tokens stay behind. The caller removes the provider. */
    async setEnabled(value: unknown): Promise<void> {
      if (typeof value !== "boolean") throw new InvalidInputError("enabled must be true or false.");
      settings.writeDeviceValue(ENABLED, value);
      if (value) onChange();
      else await signIn.signOut();
    },

    async signIn(): Promise<ChatGptSignInResult> {
      if (!enabled()) {
        throw new InvalidInputError("Turn on the ChatGPT plan provider in Settings first.");
      }
      try {
        await signIn.signIn();
        return { ok: true, status: await status() };
      } catch (error) {
        return { ok: false, error: signInError(error) };
      }
    },

    cancelSignIn: () => signIn.cancelSignIn(),

    async signOut(): Promise<ChatGptPlanStatus> {
      await signIn.signOut();
      return status();
    },
  };
}
