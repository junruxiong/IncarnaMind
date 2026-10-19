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
  ExternalService,
  SetTranscriptionKeyInput,
  TranscriptionRoute,
  TranscriptionServiceStatus,
} from "../../api";
import type { Consent } from "../../consent";
import type { Meter } from "../../cost/meter";
import { InvalidInputError, isRecord } from "../../errors";
import { qwenRegion, TRANSCRIPTION_SERVICES } from "../../providers/catalog/transcription";
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
  | {
      route: CloudTranscriptionRoute;
      stream: CloudStream;
      meter: CloudAudioMeter;
      /** Where its audio goes, which the consent is for. */
      service: ExternalService;
    }
  | { route: CloudTranscriptionRoute; refused: "cap" };

interface ServiceKey {
  providerId: string;
  apiKey: string;
  endpoint: string | null;
}

export interface CloudTranscriptionOptions {
  settings: SettingsStore;
  secrets: Secrets;
  consent: Consent;
  meter: Meter;
  localOnly(): boolean;
  /** The key of the saved chat provider for a catalog entry, with its row. */
  providerKey(catalogId: string): Promise<ServiceKey | null>;
  /** Silero VAD's file, downloaded first if needed: OpenAI's turns need it. */
  vadFile(): Promise<string>;
  fetch?: typeof fetch;
}

export type CloudTranscription = ReturnType<typeof createCloudTranscription>;

export function createCloudTranscription(options: CloudTranscriptionOptions) {
  const { settings, secrets, consent, meter } = options;
  const request = options.fetch ?? fetch;
  const chosen = () => settings.get().device.transcriptionRoute;

  async function keyOf(route: CloudTranscriptionRoute): Promise<ServiceKey | null> {
    if (TRANSCRIPTION_SERVICES[route].key === "provider") return options.providerKey(route);
    const apiKey = await secrets.tryGet(keyName(route));
    return apiKey ? { providerId: route, apiKey, endpoint: null } : null;
  }

  const serviceOf = (route: CloudTranscriptionRoute, key: ServiceKey | null): ExternalService =>
    route === "qwen"
      ? qwenRegion(key?.endpoint ?? null).service
      : TRANSCRIPTION_SERVICES[route].service;

  consent.registry.register({
    id: "transcription",
    sends: SENDS,
    async services() {
      const route = chosen();
      return isCloudRoute(route) ? [serviceOf(route, await keyOf(route))] : [];
    },
  });

  const list = async (): Promise<TranscriptionServiceStatus[]> =>
    Promise.all(
      Object.values(TRANSCRIPTION_SERVICES).map(async (service) => {
        const key = await keyOf(service.route);
        const region = service.route === "qwen" && key ? qwenRegion(key.endpoint) : null;
        return {
          route: service.route,
          name: service.name,
          recipient: service.recipient,
          model: service.model,
          perHour: region?.perHour ?? service.perHour,
          region: region?.label ?? null,
          dataUse: service.dataUse ?? null,
          key: service.key,
          hasKey: key !== null,
          liveSpeakers: service.liveSpeakers,
          keyUrl: service.keyUrl ?? null,
        };
      }),
    );

  /** Whether meeting audio may go to the service now: not in Local mode, and still allowed. */
  const allowed = (service: ExternalService) =>
    !options.localOnly() && consent.allows("transcription", service);

  return {
    list,

    allowed,

    /** The User chose where meeting audio is transcribed: a cloud service, chosen, is allowed. */
    async chose(route: TranscriptionRoute | null): Promise<void> {
      if (isCloudRoute(route)) {
        consent.choose(["transcription"], serviceOf(route, await keyOf(route)));
      }
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
      if (!isCloudRoute(route)) return null;
      const key = await keyOf(route);
      const where = serviceOf(route, key);
      if (!key || !allowed(where)) return null;
      const service = TRANSCRIPTION_SERVICES[route];
      const region = route === "qwen" ? qwenRegion(key.endpoint) : null;
      const audio = meter.audio({
        providerId: key.providerId,
        modelId: service.model,
        perHour: region?.perHour ?? service.perHour,
      });
      if (!audio.add(0)) return { route, refused: "cap" };
      const { model } = service;
      const { apiKey } = key;
      const stream: CloudStream = region
        ? { service: "qwen", model, apiKey, url: region.url }
        : route === "openai"
          ? { service: "openai", model, apiKey, vad: await options.vadFile() }
          : { service: "soniox", model, apiKey };
      return { route, stream, meter: audio, service: where };
    },
  };
}
