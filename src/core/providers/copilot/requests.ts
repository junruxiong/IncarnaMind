/** Copilot's OpenAI-compatible chat API, signed as `../signedFetch` signs requests, with the profile's headers. */
import { createOpenAICompatible } from "@ai-sdk/openai-compatible";
import { APICallError } from "ai";
import type { CopilotModel } from "../../api";
import { isRecord } from "../../errors";
import type { OAuthCredentials } from "../../oauthAccount";
import type { ChatLanguageModel } from "../models";
import { SignInRouteError } from "../providerErrors";
import { createSignedFetch, errorOf, isAnswer } from "../signedFetch";
import { NO_COPILOT_ACCESS } from "./service";

const POLICY = /polic|organi[sz]ation|enterprise|administrator/i;
const MODEL_UNAVAILABLE = /model.*not (supported|available|enabled)|unsupported model/i;
const QUOTA = /quota|allowance|premium requests?/i;

/** Null for what the AI SDK handles as it does any server's: an answer, a passing rate limit, a server error. */
async function refusalOf(response: Response): Promise<Error | null> {
  const { status } = response;
  if (status === 401) {
    const { message } = await errorOf(response);
    return new SignInRouteError("refused", `GitHub refused the Copilot sign-in (401): ${message}`);
  }
  if (status === 403) {
    const { message, code } = await errorOf(response);
    const said = `${code ?? ""} ${message}`;
    const problem = POLICY.test(said)
      ? "policy-blocked"
      : NO_COPILOT_ACCESS.test(said)
        ? "no-plan"
        : "refused";
    return new SignInRouteError(problem, `GitHub Copilot refused the request (403): ${message}`);
  }
  if (status === 402 || status === 429) {
    const { message, code } = await errorOf(response);
    if (status === 402 || code === "quota_exceeded" || QUOTA.test(message)) {
      return new SignInRouteError("plan-limit", `Your Copilot plan's limit is reached: ${message}`);
    }
    // Retryable, so the AI SDK waits and asks again as for any provider; its cause names the route's problem.
    return new APICallError({
      message: `GitHub Copilot asked to slow down: ${message}`,
      url: response.url,
      requestBodyValues: {},
      statusCode: status,
      responseHeaders: Object.fromEntries(response.headers),
      isRetryable: true,
      cause: new SignInRouteError("rate-limited", `GitHub Copilot asked to slow down: ${message}`),
    });
  }
  if (status === 400 || status === 404 || status === 422) {
    const { message, code } = await errorOf(response);
    if (code === "model_not_supported" || MODEL_UNAVAILABLE.test(message)) {
      return new SignInRouteError("model-unavailable", `GitHub Copilot: ${message}`);
    }
    return new SignInRouteError(
      "unexpected-response",
      `GitHub Copilot didn't accept the request (${status}): ${message}`,
    );
  }
  if (response.ok && !isAnswer(response)) {
    await response.body?.cancel();
    return new SignInRouteError(
      "unexpected-response",
      `GitHub Copilot answered with ${response.headers.get("content-type") || "no content type"}, not an answer.`,
    );
  }
  return null;
}

export function createCopilotFetch(options: {
  baseUrl: string;
  credentials: OAuthCredentials;
  headers: Readonly<Record<string, string>>;
  fetch?: typeof globalThis.fetch;
}): typeof globalThis.fetch {
  return createSignedFetch({
    credentials: options.credentials,
    apiBaseUrl: options.baseUrl,
    sign(headers, accessToken) {
      headers.set("authorization", `Bearer ${accessToken}`);
      for (const [name, value] of Object.entries(options.headers)) headers.set(name, value);
    },
    refusal: refusalOf,
    ...(options.fetch && { fetch: options.fetch }),
  });
}

export function createCopilotChatModel(options: {
  baseUrl: string;
  modelId: string;
  credentials: OAuthCredentials;
  headers: Readonly<Record<string, string>>;
  fetch?: typeof globalThis.fetch;
}): ChatLanguageModel {
  return createOpenAICompatible({
    name: "copilot",
    baseURL: options.baseUrl,
    includeUsage: true,
    fetch: createCopilotFetch(options),
  })(options.modelId);
}

/** Chat models meant for a picker that the plan's policy hasn't turned off, in Copilot's order. */
export async function listCopilotModels(options: {
  baseUrl: string;
  credentials: OAuthCredentials;
  headers: Readonly<Record<string, string>>;
  fetch?: typeof globalThis.fetch;
  signal?: AbortSignal;
}): Promise<CopilotModel[]> {
  const response = await createCopilotFetch(options)(`${options.baseUrl}/models`, {
    headers: { accept: "application/json" },
    ...(options.signal && { signal: options.signal }),
  });
  if (!response.ok) throw new Error(`Copilot's model list answered ${response.status}.`);
  const body: unknown = await response.json();
  const models = isRecord(body) && Array.isArray(body.data) ? body.data : [];
  return models.flatMap((model): CopilotModel[] => {
    if (!isRecord(model) || typeof model.id !== "string") return [];
    if (model.model_picker_enabled === false) return [];
    if (isRecord(model.policy) && model.policy.state === "disabled") return [];
    if (isRecord(model.capabilities) && model.capabilities.type !== undefined) {
      if (model.capabilities.type !== "chat") return [];
    }
    return [{ id: model.id, name: typeof model.name === "string" ? model.name : model.id }];
  });
}
