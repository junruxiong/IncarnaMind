/** What each kind of chat provider needs, and where its data goes. */
import { type ChatProviderKind, chatProviderKinds, type ExternalService } from "../api";
import { InvalidInputError } from "../errors";
import { catalogProviderOfKind, endpointOf } from "./catalog/providers";
import type { CatalogProvider, Endpoint } from "./catalog/types";
import { CHATGPT_SERVICE } from "./chatgpt/codexEndpoint";

/** Ollama's default local address. 127.0.0.1 rather than localhost, which may resolve to IPv6 first. */
export const OLLAMA_DEFAULT_URL = "http://127.0.0.1:11434";

/**
 * The address in `OLLAMA_HOST`, read the way Ollama's own client reads it
 * (`envconfig.Host`): without a scheme, http and port 11434; with "http://"
 * or "https://" and no port, 80 or 443; a path is kept. The address a server
 * listens on for every interface (0.0.0.0, ::) is reached on loopback. Null
 * when it is unset or can't be read.
 */
export function ollamaHostUrl(value: string | undefined): string | null {
  const raw = value?.trim();
  if (!raw) return null;
  const [scheme, rest] = raw.includes("://")
    ? (raw.split("://", 2) as [string, string])
    : ["", raw];
  if (scheme && scheme !== "http" && scheme !== "https") return null;
  const defaultPort = scheme === "https" ? "443" : scheme === "http" ? "80" : "11434";
  const slash = rest.indexOf("/");
  const hostPort = slash === -1 ? rest : rest.slice(0, slash);
  const path = slash === -1 ? "" : rest.slice(slash).replace(/\/+$/, "");
  const bracketed = /^\[([^\]]+)\](?::(\d*))?$/.exec(hostPort);
  const plain = bracketed ? null : /^([^:]*)(?::(\d*))?$/.exec(hostPort);
  let host = bracketed?.[1] ?? plain?.[1] ?? "";
  let port = bracketed?.[2] ?? plain?.[2] ?? "";
  // More than one colon and no brackets: a bare IPv6 address, without a port.
  if (!bracketed && !plain) {
    host = hostPort;
    port = "";
  }
  if (!/^\d+$/.test(port) || Number(port) > 65535) port = defaultPort;
  if (host === "" || host === "0.0.0.0" || host === "::") host = "127.0.0.1";
  const shown = host.includes(":") ? `[${host}]` : host;
  try {
    return normalizeBaseUrl(`${scheme || "http"}://${shown}:${port}${path}`);
  } catch {
    return null;
  }
}

/** Where Ollama is when the User gave no URL: `OLLAMA_HOST` if set, else the default address. */
export const defaultOllamaUrl = (env: NodeJS.ProcessEnv = process.env): string =>
  ollamaHostUrl(env.OLLAMA_HOST) ?? OLLAMA_DEFAULT_URL;

/**
 * What a provider kind needs: a key, a server URL. A kind the catalog has
 * takes its rules from there (see ./catalog); the ChatGPT plan signs in
 * instead of taking a key (see ./chatgpt).
 */
interface KindInfo {
  /** The hosted API, for a provider whose endpoints the catalog names. */
  hosted?: ExternalService;
  baseUrl: "none" | "required" | "optional";
  apiKey: "required" | "optional" | "none";
}

/** The service an endpoint sends data to: its origin, named after its provider. */
const endpointService = (provider: CatalogProvider, endpoint: Endpoint): ExternalService => ({
  id: new URL(endpoint.baseUrl).origin,
  name: provider.name.en,
});

function kindInfo(kind: ChatProviderKind, endpointId?: string | null): KindInfo {
  if (kind === "chatgpt") return { hosted: CHATGPT_SERVICE, baseUrl: "none", apiKey: "none" };
  const provider = catalogProviderOfKind(kind);
  if (!provider) throw new Error(`The catalog has no provider for ${kind}.`);
  const endpoint = endpointOf(provider, endpointId);
  return {
    ...(endpoint && { hosted: endpointService(provider, endpoint) }),
    baseUrl: endpoint ? "none" : (provider.serverUrl ?? "required"),
    apiKey: provider.apiKey,
  };
}

export function isChatProviderKind(value: unknown): value is ChatProviderKind {
  return chatProviderKinds.some((kind) => kind === value);
}

/**
 * The base URL of a hosted provider's API, as its AI SDK provider takes it:
 * its endpoint with this id, else its first. Null for a server the User gives.
 */
export function endpointUrl(kind: ChatProviderKind, endpointId?: string | null): string | null {
  const provider = catalogProviderOfKind(kind);
  return (provider && endpointOf(provider, endpointId)?.baseUrl) ?? null;
}

export const requiresApiKey = (kind: ChatProviderKind) => kindInfo(kind).apiKey === "required";
export const acceptsApiKey = (kind: ChatProviderKind) => kindInfo(kind).apiKey !== "none";

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
  const { baseUrl: rule, hosted } = kindInfo(kind);
  const given = typeof raw === "string" && raw.trim() !== "" ? raw : undefined;
  if (raw !== undefined && raw !== null && typeof raw !== "string") {
    throw new InvalidInputError("The server URL must be text.");
  }
  if (rule === "none") {
    if (given) throw new InvalidInputError(`${hosted?.name} doesn't take a server URL.`);
    return null;
  }
  if (!given) {
    if (rule === "required") throw new InvalidInputError("Enter the server's URL.");
    return defaultOllamaUrl();
  }
  return normalizeBaseUrl(given);
}

/** Ollama's address: the URL the User gave, else `OLLAMA_HOST`, else the default. */
export const ollamaBaseUrl = (raw: unknown): string =>
  baseUrlFor("ollama", raw) ?? defaultOllamaUrl();

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

/**
 * Where a provider's requests go: its endpoint (the one with this id, else
 * its first), or the server at `baseUrl`; null when that runs on this computer.
 */
export function serviceFor(
  kind: ChatProviderKind,
  baseUrl: string | null,
  endpointId?: string | null,
): ExternalService | null {
  const { hosted } = kindInfo(kind, endpointId);
  if (hosted) return hosted;
  return baseUrl ? serviceForUrl(baseUrl) : null;
}
