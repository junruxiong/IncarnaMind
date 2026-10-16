/**
 * Every request carries a current access token, only to the API's origin, and
 * IncarnaMind's User-Agent; a refused token is refreshed once and retried.
 */
import { version } from "../../../package.json";
import { isRecord } from "../errors";
import type { OAuthCredentials } from "../oauthAccount";

export const USER_AGENT = `IncarnaMind/${version}`;

/** What a route adds to every request it sends, besides IncarnaMind's own. */
export interface RequestTransform {
  /** System blocks before IncarnaMind's own. */
  systemBlocks?: readonly string[];
  /** Headers to set; an `anthropic-beta` adds its flags to those the request has. */
  headers?: Readonly<Record<string, string>>;
}

export function createSignedFetch(options: {
  credentials: OAuthCredentials;
  /** The API's base URL: requests anywhere else are refused before the token is read. */
  apiBaseUrl: string;
  /** Sets the request's headers for the access token, after the User-Agent is set. */
  sign(headers: Headers, accessToken: string): void;
  body?(url: string, body: RequestInit["body"]): RequestInit["body"];
  /** A response the route can't use, as its error; null hands it to the AI SDK. */
  refusal(response: Response): Promise<Error | null>;
  fetch?: typeof globalThis.fetch;
}): typeof globalThis.fetch {
  const fetcher = options.fetch ?? globalThis.fetch;
  const origin = new URL(options.apiBaseUrl).origin;
  const signedFetch = async (input: Parameters<typeof globalThis.fetch>[0], init?: RequestInit) => {
    const url = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
    if (new URL(url).origin !== origin) {
      throw new Error(`The sign-in's token is only sent to ${origin}.`);
    }
    const body = options.body ? options.body(url, init?.body) : init?.body;

    const send = async (rejected?: string) => {
      const { accessToken } = await options.credentials.access(rejected ? { rejected } : {});
      const headers = new Headers(init?.headers);
      headers.set("user-agent", USER_AGENT);
      options.sign(headers, accessToken);
      headers.delete("content-length");
      // A redirect would carry the token to another address.
      const response = await fetcher(url, { ...init, headers, body, redirect: "error" });
      return { accessToken, response };
    };

    const first = await send();
    let { response } = first;
    if (response.status === 401) {
      // The token may have been revoked early: refresh once and retry. A refusal after that is final.
      await response.body?.cancel();
      ({ response } = await send(first.accessToken));
    }
    const refused = await options.refusal(response);
    if (refused) throw refused;
    return response;
  };
  return signedFetch;
}

/** An API error's message and code, from Anthropic's, OpenAI's or GitHub's shape; the body's first words otherwise. */
export async function errorOf(
  response: Response,
): Promise<{ message: string; code: string | null }> {
  const raw = await response.text();
  let body: unknown;
  try {
    body = JSON.parse(raw);
  } catch {
    body = undefined;
  }
  const error = isRecord(body) && isRecord(body.error) ? body.error : body;
  const message = isRecord(error) && typeof error.message === "string" ? error.message : null;
  const code = isRecord(error) && typeof error.code === "string" ? error.code : null;
  return { message: message ?? (raw.slice(0, 200) || response.statusText), code };
}

/** Whether a successful response is an answer an AI SDK provider can read: JSON or a stream of events. */
export const isAnswer = (response: Response) =>
  /application\/json|text\/event-stream/.test(response.headers.get("content-type") ?? "");
