/** What each kind of chat provider needs, and where its data goes. */
import { type ChatProviderKind, chatProviderKinds, type ExternalService } from "../api";
import { InvalidInputError } from "../errors";
import { CHATGPT_SERVICE } from "./chatgpt/codexEndpoint";

/** Ollama's default local address. 127.0.0.1 rather than localhost, which may resolve to IPv6 first. */
export const OLLAMA_DEFAULT_URL = "http://127.0.0.1:11434";

interface KindInfo {
  /** The hosted API, for kinds that don't take a base URL. */
  hosted?: ExternalService;
  baseUrl: "none" | "required" | "optional";
  apiKey: "required" | "optional" | "none";
}

const kinds: Record<ChatProviderKind, KindInfo> = {
  openai: {
    hosted: { id: "https://api.openai.com", name: "OpenAI" },
    baseUrl: "none",
    apiKey: "required",
  },
  anthropic: {
    hosted: { id: "https://api.anthropic.com", name: "Anthropic" },
    baseUrl: "none",
    apiKey: "required",
  },
  google: {
    hosted: { id: "https://generativelanguage.googleapis.com", name: "Google" },
    baseUrl: "none",
    apiKey: "required",
  },
  "openai-compatible": { baseUrl: "required", apiKey: "optional" },
  ollama: { baseUrl: "optional", apiKey: "none" },
  // Signs in instead of taking a key (see ./chatgpt).
  chatgpt: { hosted: CHATGPT_SERVICE, baseUrl: "none", apiKey: "none" },
};

export function isChatProviderKind(value: unknown): value is ChatProviderKind {
  return chatProviderKinds.some((kind) => kind === value);
}

export const requiresApiKey = (kind: ChatProviderKind) => kinds[kind].apiKey === "required";
export const acceptsApiKey = (kind: ChatProviderKind) => kinds[kind].apiKey !== "none";

/**
 * Checks a server URL and returns it without a trailing slash. Credentials,
 * query strings and fragments are refused: the URL is stored in the database,
 * and keys belong in the API key field.
 */
export function normalizeBaseUrl(raw: string): string {
  let url: URL;
  try {
    url = new URL(raw.trim());
  } catch {
    throw new InvalidInputError(`"${raw}" isn't a valid server URL.`);
  }
  if (url.protocol !== "http:" && url.protocol !== "https:") {
    throw new InvalidInputError("The server URL must start with http:// or https://.");
  }
  if (url.username || url.password || url.search || url.hash) {
    throw new InvalidInputError(
      "The server URL can't contain a user name, password, query or fragment. Put an API key in the key field.",
    );
  }
  return `${url.origin}${url.pathname.replace(/\/+$/, "")}`;
}

/** The base URL to store for a kind, from what the User entered. */
export function baseUrlFor(kind: ChatProviderKind, raw: unknown): string | null {
  const rule = kinds[kind].baseUrl;
  const given = typeof raw === "string" && raw.trim() !== "" ? raw : undefined;
  if (raw !== undefined && raw !== null && typeof raw !== "string") {
    throw new InvalidInputError("The server URL must be text.");
  }
  if (rule === "none") {
    if (given)
      throw new InvalidInputError(`${kinds[kind].hosted?.name} doesn't take a server URL.`);
    return null;
  }
  if (!given) {
    if (rule === "required") throw new InvalidInputError("Enter the server's URL.");
    return OLLAMA_DEFAULT_URL;
  }
  return normalizeBaseUrl(given);
}

export const ollamaBaseUrl = (raw: unknown): string =>
  baseUrlFor("ollama", raw) ?? OLLAMA_DEFAULT_URL;

/** Loopback addresses: data sent there stays on this computer. */
export function isLoopbackHost(hostname: string): boolean {
  const host = hostname.toLowerCase().replace(/^\[|\]$/g, "");
  return (
    host === "localhost" ||
    host.endsWith(".localhost") ||
    host === "::1" ||
    host === "0.0.0.0" ||
    /^127(\.\d{1,3}){3}$/.test(host)
  );
}

/** The service a server URL sends data to, or null when the server runs on this computer. */
export function serviceForUrl(baseUrl: string): ExternalService | null {
  const url = new URL(baseUrl);
  return isLoopbackHost(url.hostname) ? null : { id: url.origin, name: url.host };
}

/** Where a provider's requests go, or null when the server runs on this computer. */
export function serviceFor(kind: ChatProviderKind, baseUrl: string | null): ExternalService | null {
  const hosted = kinds[kind].hosted;
  if (hosted) return hosted;
  return baseUrl ? serviceForUrl(baseUrl) : null;
}
