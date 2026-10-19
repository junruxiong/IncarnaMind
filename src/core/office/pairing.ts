/**
 * How an Office task pane proves it is the User's own: a short code the app
 * shows (the User asked for it there), typed once into the pane, which swaps
 * it for a token only that pane holds. The code lives two minutes, allows three
 * wrong tries and works once; the token is random, and only its hash is kept,
 * in ../secrets, never in a log.
 */
import { createHash, randomBytes, randomInt, randomUUID, timingSafeEqual } from "node:crypto";
import type { Secrets } from "../secrets";

export const OFFICE_HOSTS = ["word", "excel"] as const;
export type OfficeHost = (typeof OFFICE_HOSTS)[number];

export interface OfficePairing {
  id: string;
  host: OfficeHost;
  createdAt: string;
}

interface StoredPairing extends OfficePairing {
  tokenHash: string;
}

export type PairingRefusal = "no-code" | "expired" | "wrong" | "locked";

export class PairingRefusedError extends Error {
  readonly reason: PairingRefusal;

  constructor(reason: PairingRefusal) {
    super(`Pairing refused: ${reason}.`);
    this.reason = reason;
  }
}

export interface PairingCode {
  code: string;
  expiresAt: number;
}

export const CODE_LIFETIME_MS = 120_000;
export const WRONG_TRIES_ALLOWED = 3;
const STORE = "office-bridge:pairings";

export interface PairingsOptions {
  secrets: Pick<Secrets, "set" | "tryGet" | "delete">;
  /** Milliseconds, any origin: only differences are used. */
  clock(): number;
  now(): string;
  removed(pairingId: string): void;
}

const hashOf = (token: string) => createHash("sha256").update(token).digest();

function sameText(a: string, b: string): boolean {
  const left = Buffer.from(a);
  const right = Buffer.from(b);
  return left.length === right.length && timingSafeEqual(left, right);
}

export function createPairings(options: PairingsOptions) {
  let active: (PairingCode & { wrong: number }) | null = null;
  let cache: StoredPairing[] | null = null;

  const load = async (): Promise<StoredPairing[]> => {
    if (cache) return cache;
    const text = await options.secrets.tryGet(STORE);
    cache = text ? (JSON.parse(text) as StoredPairing[]) : [];
    return cache;
  };
  const save = async (pairings: StoredPairing[]) => {
    cache = pairings;
    if (pairings.length === 0) await options.secrets.delete(STORE);
    else await options.secrets.set(STORE, JSON.stringify(pairings));
  };
  const visible = ({ id, host, createdAt }: StoredPairing): OfficePairing => ({
    id,
    host,
    createdAt,
  });

  return {
    /** A new code replaces the last one: only the one on screen can be typed. */
    start(): PairingCode {
      const code = String(randomInt(0, 1_000_000)).padStart(6, "0");
      active = { code, expiresAt: options.clock() + CODE_LIFETIME_MS, wrong: 0 };
      return { code: active.code, expiresAt: active.expiresAt };
    },

    async redeem(
      code: string,
      host: OfficeHost,
    ): Promise<{ token: string; pairing: OfficePairing }> {
      if (!active) throw new PairingRefusedError("no-code");
      if (options.clock() >= active.expiresAt) {
        active = null;
        throw new PairingRefusedError("expired");
      }
      if (active.wrong >= WRONG_TRIES_ALLOWED) throw new PairingRefusedError("locked");
      if (!sameText(code, active.code)) {
        active.wrong += 1;
        if (active.wrong >= WRONG_TRIES_ALLOWED) active = null;
        throw new PairingRefusedError("wrong");
      }
      active = null;
      const token = randomBytes(32).toString("base64url");
      const stored: StoredPairing = {
        id: randomUUID(),
        host,
        createdAt: options.now(),
        tokenHash: hashOf(token).toString("hex"),
      };
      await save([...(await load()), stored]);
      return { token, pairing: visible(stored) };
    },

    async authenticate(token: string): Promise<OfficePairing | null> {
      const presented = hashOf(token);
      let found: StoredPairing | null = null;
      for (const pairing of await load()) {
        if (timingSafeEqual(presented, Buffer.from(pairing.tokenHash, "hex"))) found = pairing;
      }
      return found && visible(found);
    },

    async list(): Promise<OfficePairing[]> {
      return (await load()).map(visible);
    },

    async remove(pairingId: string): Promise<void> {
      const pairings = await load();
      if (!pairings.some((pairing) => pairing.id === pairingId)) return;
      await save(pairings.filter((pairing) => pairing.id !== pairingId));
      options.removed(pairingId);
    },
  };
}

export type Pairings = ReturnType<typeof createPairings>;
