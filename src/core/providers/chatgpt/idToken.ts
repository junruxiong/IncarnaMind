/**
 * Checks the ID token a ChatGPT sign-in returns before IncarnaMind trusts who
 * it says the User is: its RS256 signature against OpenAI's published keys
 * (JWKS), its issuer, its audience (the client OpenAI issued to IncarnaMind),
 * its expiry and the sign-in's nonce. OpenAI signs ID tokens with RS256 only
 * (`id_token_signing_alg_values_supported` in its OpenID configuration), so
 * Node's crypto is enough and no JWT library is needed.
 */
import { createPublicKey, type JsonWebKey, verify } from "node:crypto";
import { isRecord } from "../../errors";

/** How far this computer's clock may be off from OpenAI's. */
const CLOCK_SKEW_SECONDS = 5 * 60;

export class IdTokenError extends Error {
  override name = "IdTokenError";
}

export interface IdTokenClaims {
  /** The ChatGPT account's stable identifier. */
  subject: string;
  email: string | null;
}

const part = (value: string): Record<string, unknown> | null => {
  try {
    const parsed: unknown = JSON.parse(Buffer.from(value, "base64url").toString("utf8"));
    return isRecord(parsed) ? parsed : null;
  } catch {
    return null;
  }
};

/** An RSA public key from a JWKS, by its key ID. */
function rsaKey(value: unknown): { kid: string; jwk: JsonWebKey } | null {
  if (!isRecord(value) || value.kty !== "RSA") return null;
  const { kid, n, e } = value;
  if (typeof kid !== "string" || typeof n !== "string" || typeof e !== "string") return null;
  return { kid, jwk: { kty: "RSA", n, e } };
}

/** The keys OpenAI signs with, fetched again when a token names one not seen yet. */
export function createJwks(url: string, fetcher: typeof globalThis.fetch = globalThis.fetch) {
  let keys: { kid: string; jwk: JsonWebKey }[] = [];

  const load = async (signal?: AbortSignal) => {
    const response = await fetcher(url, {
      headers: { accept: "application/json" },
      ...(signal && { signal }),
    });
    if (!response.ok) {
      throw new IdTokenError(`OpenAI's signing keys couldn't be read (${response.status}).`);
    }
    const body: unknown = await response.json();
    const listed = isRecord(body) && Array.isArray(body.keys) ? body.keys : [];
    keys = listed.flatMap((key) => rsaKey(key) ?? []);
  };

  return {
    async key(kid: string, signal?: AbortSignal): Promise<JsonWebKey> {
      const find = () => keys.find((key) => key.kid === kid)?.jwk;
      if (!find()) await load(signal);
      const key = find();
      if (!key) throw new IdTokenError("The sign-in was signed with a key OpenAI doesn't publish.");
      return key;
    },
  };
}

export type Jwks = ReturnType<typeof createJwks>;

/**
 * The verified claims of `token`, or `IdTokenError` saying what didn't hold.
 * `now` is in milliseconds since the epoch.
 */
export async function verifyIdToken(options: {
  token: string;
  jwks: Jwks;
  issuer: string;
  audience: string;
  nonce: string;
  now: number;
  signal?: AbortSignal;
}): Promise<IdTokenClaims> {
  const segments = options.token.split(".");
  const [encodedHeader, encodedPayload, encodedSignature] = segments;
  if (segments.length !== 3 || !encodedHeader || !encodedPayload || !encodedSignature) {
    throw new IdTokenError("The sign-in's ID token isn't a JWT.");
  }
  const header = part(encodedHeader);
  const claims = part(encodedPayload);
  if (!header || !claims) throw new IdTokenError("The sign-in's ID token can't be read.");
  if (header.alg !== "RS256" || typeof header.kid !== "string") {
    throw new IdTokenError("The sign-in's ID token isn't signed the way OpenAI signs.");
  }

  const jwk = await options.jwks.key(header.kid, options.signal);
  const signed = verify(
    "RSA-SHA256",
    Buffer.from(`${encodedHeader}.${encodedPayload}`),
    createPublicKey({ key: jwk, format: "jwk" }),
    Buffer.from(encodedSignature, "base64url"),
  );
  if (!signed) throw new IdTokenError("The sign-in's ID token isn't signed by OpenAI.");

  if (claims.iss !== options.issuer) {
    throw new IdTokenError("The sign-in's ID token comes from another issuer.");
  }
  const audience = Array.isArray(claims.aud) ? claims.aud : [claims.aud];
  if (!audience.includes(options.audience)) {
    throw new IdTokenError("The sign-in's ID token was issued to another app.");
  }
  const seconds = options.now / 1000;
  if (typeof claims.exp !== "number" || claims.exp + CLOCK_SKEW_SECONDS < seconds) {
    throw new IdTokenError(
      "The sign-in's ID token has expired. Check that this computer's clock is right.",
    );
  }
  if (typeof claims.iat === "number" && claims.iat - CLOCK_SKEW_SECONDS > seconds) {
    throw new IdTokenError(
      "The sign-in's ID token was issued in the future. Check that this computer's clock is right.",
    );
  }
  if (claims.nonce !== options.nonce) {
    throw new IdTokenError("The sign-in's ID token belongs to another sign-in.");
  }
  if (typeof claims.sub !== "string" || claims.sub === "") {
    throw new IdTokenError("The sign-in's ID token doesn't say which account signed in.");
  }
  return {
    subject: claims.sub,
    email: typeof claims.email === "string" && claims.email !== "" ? claims.email : null,
  };
}
