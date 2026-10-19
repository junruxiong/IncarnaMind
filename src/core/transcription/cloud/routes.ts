/**
 * Where meeting audio is transcribed (#141), as the core decides it: on this computer by default,
 * or by the cloud service the User chose in Settings › Meetings or on the first-meeting sheet,
 * with their own key. OpenAI's is the key of the OpenAI provider in Settings › Models; Soniox
 * asks for its own, kept in the keychain.
 *
 * Choosing a service is the consent to send it meeting audio (ADR-0022): the "transcription"
 * flow. Nothing goes to it in Local mode, after the User revokes it in Settings › Privacy, or
 * without a key; the audio is then transcribed on this computer. The audio is counted against
 * the monthly cap before it is sent (../../cost/meter).
 */
import type { CloudStream } from "../../adapters";
import type {
  CloudTranscriptionRoute,
  SetTranscriptionKeyInput,
  TranscriptionRoute,
  TranscriptionServiceStatus,
} from "../../api";
import type { Consent } from "../../consent";
import type { Meter } from "../../cost/meter";
import { InvalidInputError, isRecord } from "../../errors";
import { TRANSCRIPTION_SERVICES } from "../../providers/catalog/transcription";
import type { Secrets } from "../../secrets";
import type { SettingsStore } from "../../settings";

/** What the "transcription" flow sends. */
const SENDS = ["meeting-audio"] as const;

/** Where a service's own key is checked: a request that reads nothing of the User's. */
const KEY_CHECKS: Partial<Record<CloudTranscriptionRoute, string>> = {
  soniox: "https://api.soniox.com/v1/models",
};
const KEY_CHECK_TIMEOUT_MS = 15_000;

const keyName = (route: CloudTranscriptionRoute) => `transcription:${route}:api-key`;

export const isCloudRoute = (route: unknown): route is CloudTranscriptionRoute =>
  typeof route === "string" && Object.hasOwn(TRANSCRIPTION_SERVICES, route);

/** A stretch of recording's audio, counted as it goes to the service. */
export interface CloudAudioMeter {
  /** Counts audio about to be sent; false, counting nothing, when it would pass the monthly cap. */
  add(ms: number): boolean;
}

export type CloudStart =
  | { route: CloudTranscriptionRoute; stream: CloudStream; meter: CloudAudioMeter }
  | { route: CloudTranscriptionRoute; refused: "cap" };

export interface CloudTranscriptionOptions {
  settings: SettingsStore;
  secrets: Secrets;
  consent: Consent;
  meter: Meter;
  localOnly(): boolean;
  /** The key of the saved chat provider for a catalog entry, with its row. */
  providerKey(catalogId: string): Promise<{ providerId: string; apiKey: string } | null>;
  /** Silero VAD's file, downloaded first if needed: OpenAI's turns need it. */
  vadFile(): Promise<string>;
  fetch?: typeof fetch;
}

export type CloudTranscription = ReturnType<typeof createCloudTranscription>;

export function createCloudTranscription(options: CloudTranscriptionOptions) {
  const { settings, secrets, consent, meter } = options;
  const request = options.fetch ?? fetch;
  const chosen = () => settings.get().device.transcriptionRoute;

  async function keyOf(
    route: CloudTranscriptionRoute,
  ): Promise<{ providerId: string; apiKey: string } | null> {
    if (TRANSCRIPTION_SERVICES[route].key === "provider") return options.providerKey(route);
    const apiKey = await secrets.tryGet(keyName(route));
    return apiKey ? { providerId: route, apiKey } : null;
  }

  consent.registry.register({
    id: "transcription",
    sends: SENDS,
    async services() {
      const route = chosen();
      return isCloudRoute(route) ? [TRANSCRIPTION_SERVICES[route].service] : [];
    },
  });

  const list = async (): Promise<TranscriptionServiceStatus[]> =>
    Promise.all(
      Object.values(TRANSCRIPTION_SERVICES).map(async (service) => ({
        route: service.route,
        name: service.name,
        model: service.model,
        perHour: service.perHour,
        key: service.key,
        hasKey: (await keyOf(service.route)) !== null,
        liveSpeakers: service.liveSpeakers,
        keyUrl: service.keyUrl ?? null,
      })),
    );

  /** Whether meeting audio may go to the service now: not in Local mode, and still allowed. */
  const allowed = (route: CloudTranscriptionRoute) =>
    !options.localOnly() && consent.allows("transcription", TRANSCRIPTION_SERVICES[route].service);

  return {
    list,

    allowed,

    /** The User chose where meeting audio is transcribed: a cloud service, chosen, is allowed. */
    chose(route: TranscriptionRoute | null): void {
      if (isCloudRoute(route))
        consent.choose(["transcription"], TRANSCRIPTION_SERVICES[route].service);
    },

    /** A choice the User can make now: a cloud service needs its key. */
    async check(route: unknown): Promise<void> {
      if (!isCloudRoute(route)) return;
      if (options.localOnly())
        throw new InvalidInputError({ key: "error.transcription.localMode" });
      if (!(await keyOf(route))) {
        throw new InvalidInputError({
          key: "error.transcription.noKey",
          values: { service: TRANSCRIPTION_SERVICES[route].name },
        });
      }
    },

    async setKey(input: unknown): Promise<TranscriptionServiceStatus[]> {
      const given = isRecord(input) ? (input as Partial<SetTranscriptionKeyInput>) : {};
      const route = given.route;
      if (!isCloudRoute(route) || TRANSCRIPTION_SERVICES[route].key !== "own") {
        throw new InvalidInputError({ key: "error.transcription.noOwnKey" });
      }
      const apiKey = typeof given.apiKey === "string" ? given.apiKey.trim() : "";
      if (!apiKey) throw new InvalidInputError({ key: "error.transcription.keyMissing" });
      const check = KEY_CHECKS[route];
      if (check) {
        let status: number;
        try {
          status = (
            await request(check, {
              headers: { Authorization: `Bearer ${apiKey}` },
              signal: AbortSignal.timeout(KEY_CHECK_TIMEOUT_MS),
            })
          ).status;
        } catch {
          throw new InvalidInputError({
            key: "error.transcription.unreachable",
            values: { service: TRANSCRIPTION_SERVICES[route].name },
          });
        }
        if (status === 401 || status === 403) {
          throw new InvalidInputError({
            key: "error.transcription.keyRefused",
            values: { service: TRANSCRIPTION_SERVICES[route].name },
          });
        }
      }
      await secrets.set(keyName(route), apiKey);
      return list();
    },

    async removeKey(route: unknown): Promise<TranscriptionServiceStatus[]> {
      if (!isCloudRoute(route))
        throw new InvalidInputError({ key: "error.transcription.unknownService" });
      if (TRANSCRIPTION_SERVICES[route].key === "own") await secrets.delete(keyName(route));
      if (chosen() === route) settings.update({ device: { transcriptionRoute: "local" } });
      return list();
    },

    /**
     * Where a stretch of recording starting now is transcribed: the chosen cloud service, ready
     * to open streams, or null for this computer. Refused at the monthly cap.
     */
    async start(): Promise<CloudStart | null> {
      const route = chosen();
      if (!isCloudRoute(route) || !allowed(route)) return null;
      const service = TRANSCRIPTION_SERVICES[route];
      const key = await keyOf(route);
      if (!key) return null;
      const audio = meter.audio({
        providerId: key.providerId,
        modelId: service.model,
        perHour: service.perHour,
      });
      if (!audio.add(0)) return { route, refused: "cap" };
      const stream: CloudStream =
        route === "openai"
          ? {
              service: "openai",
              model: service.model,
              apiKey: key.apiKey,
              vad: await options.vadFile(),
            }
          : { service: "soniox", model: service.model, apiKey: key.apiKey };
      return { route, stream, meter: audio };
    },
  };
}
