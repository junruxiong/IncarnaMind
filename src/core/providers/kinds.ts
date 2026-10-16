/** What each kind of chat provider needs, and where its data goes. */
import { type ChatProviderKind, chatProviderKinds, type ExternalService } from "../api";
import { InvalidInputError } from "../errors";
import { catalogProviderOf, endpointOf } from "./catalog/providers";
import type { CatalogProvider, Endpoint } from "./catalog/types";
import { CHATGPT_SERVICE } from "./chatgpt/service";
import { serviceAt, serviceForUrl } from "./service";

export { isLoopbackHost, serviceForUrl } from "./service";

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

const endpointService = (provider: CatalogProvider, endpoint: Endpoint) =>
  serviceAt(endpoint.baseUrl, provider.name.en);

function kindInfo(
  kind: ChatProviderKind,
  endpointId?: string | null,
  catalogId?: string | null,
): KindInfo {
  if (kind === "chatgpt") return { hosted: CHATGPT_SERVICE, baseUrl: "none", apiKey: "none" };
  const provider = catalogProviderOf(kind, catalogId);
  if (!provider) throw new Error(`The catalog has no provider for ${kind}.`);
  const endpoint = endpointOf(provider, endpointId);
  // An address built from a value the User gives is a service of its own per value.
  const hosted = endpoint && !endpoint.field ? endpointService(provider, endpoint) : null;
  return {
    ...(hosted && { hosted }),
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
export function endpointUrl(
  kind: ChatProviderKind,
  endpointId?: string | null,
  catalogId?: string | null,
): string | null {
  const provider = catalogProviderOf(kind, catalogId);
  const endpoint = provider && endpointOf(provider, endpointId);
  return endpoint && !endpoint.field ? endpoint.baseUrl : null;
}

export const requiresApiKey = (kind: ChatProviderKind, catalogId?: string | null) =>
  kindInfo(kind, null, catalogId).apiKey === "required";
export const acceptsApiKey = (kind: ChatProviderKind, catalogId?: string | null) =>
  kindInfo(kind, null, catalogId).apiKey !== "none";

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
    throw new InvalidInputError({ key: "error.provider.urlInvalid", values: { url: raw } });
  }
  if (url.protocol !== "http:" && url.protocol !== "https:") {
    throw new InvalidInputError({ key: "error.provider.urlScheme" });
  }
  if (url.username || url.password || url.search || url.hash) {
    throw new InvalidInputError({ key: "error.provider.urlExtras" });
  }
  return `${url.origin}${url.pathname.replace(/\/+$/, "")}`;
}

/** The base URL to store for a kind, from what the User entered. */
export function baseUrlFor(
  kind: ChatProviderKind,
  raw: unknown,
  catalogId?: string | null,
): string | null {
  const { baseUrl: rule } = kindInfo(kind, null, catalogId);
  const given = typeof raw === "string" && raw.trim() !== "" ? raw : undefined;
  if (raw !== undefined && raw !== null && typeof raw !== "string") {
    throw new InvalidInputError("The server URL must be text.");
  }
  if (rule === "none") {
    if (given) {
      const name = catalogProviderOf(kind, catalogId)?.name.en ?? kind;
      throw new InvalidInputError(`${name} doesn't take a server URL.`);
    }
    return null;
  }
  if (!given) {
    if (rule === "required") throw new InvalidInputError({ key: "error.provider.urlMissing" });
    return defaultOllamaUrl();
  }
  return normalizeBaseUrl(given);
}

/** Ollama's address: the URL the User gave, else `OLLAMA_HOST`, else the default. */
export const ollamaBaseUrl = (raw: unknown): string =>
  baseUrlFor("ollama", raw) ?? defaultOllamaUrl();

/**
 * Where a provider's requests go: its endpoint (the one with this id, else
 * its first), or the server at `baseUrl`; null when that runs on this computer.
 */
export function serviceFor(
  kind: ChatProviderKind,
  baseUrl: string | null,
  endpointId?: string | null,
  catalogId?: string | null,
): ExternalService | null {
  const { hosted } = kindInfo(kind, endpointId, catalogId);
  if (hosted) return hosted;
  if (!baseUrl) return null;
  const provider = catalogProviderOf(kind, catalogId);
  const endpoint = provider && endpointOf(provider, endpointId);
  return provider && endpoint?.field
    ? serviceAt(baseUrl, provider.name.en)
    : serviceForUrl(baseUrl);
}
