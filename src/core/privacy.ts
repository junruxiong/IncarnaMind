/**
 * Privacy: the User's choices about crash reports, update checks and usage
 * data on this device, and the registry of network traffic that carries no
 * User content.
 *
 * Data that carries User content goes through a data flow and needs consent
 * (see ./consent). Everything else that reaches the network, such as update
 * checks and model downloads, is registered here by the feature that makes
 * it, so the Privacy page can list it. Crash reports are off until the User
 * opts in; the host's `CrashReporter` is turned on only then. Usage data is
 * sent only while the User agrees (see ./usageData).
 */
import type { CrashReporter } from "./adapters";
import {
  type ExternalService,
  type NetworkTraffic,
  type NetworkTrafficId,
  networkTrafficIds,
  type PrivacySettings,
} from "./api";
import { InvalidInputError, isRecord } from "./errors";
import type { SettingsStore } from "./settings";
import type { UsageData } from "./usageData";

export interface NetworkTrafficDefinition {
  id: NetworkTrafficId;
  /** Where the traffic goes. */
  service?: ExternalService;
  /**
   * Where it goes when that depends on what the User set up, e.g. each remote
   * Connector's servers: one entry per service. Used instead of `service`.
   */
  services?(): ExternalService[] | Promise<ExternalService[]>;
  /**
   * Whether the traffic can happen in this app at all, e.g. the ChatGPT
   * sign-in only while the experimental provider is on. Unlisted otherwise.
   * Defaults to always.
   */
  listed?(): boolean | Promise<boolean>;
  /** False while the User has turned it off, e.g. automatic update checks. Defaults to on. */
  enabled?(): boolean | Promise<boolean>;
}

/** The registry of network traffic without User content. Registering an id again replaces it. */
export interface NetworkTrafficRegistry {
  register(definition: NetworkTrafficDefinition): void;
  get(id: NetworkTrafficId): NetworkTrafficDefinition | undefined;
  list(): NetworkTrafficDefinition[];
}

/** Per-device values, outside `DeviceSettings` so only this module changes them. */
const CRASH_REPORTS = "privacy.crashReports";
const AUTOMATIC_UPDATE_CHECKS = "privacy.automaticUpdateChecks";

const PATCH_KEYS = new Set(["crashReports", "automaticUpdateChecks", "usageData"]);

export function createPrivacy(options: {
  settings: SettingsStore;
  /** Absent when this copy can't send crash reports. */
  crashReporter: CrashReporter | undefined;
  usageData: UsageData;
}) {
  const { settings, crashReporter, usageData } = options;
  const definitions = new Map<NetworkTrafficId, NetworkTrafficDefinition>();

  const registry: NetworkTrafficRegistry = {
    register(definition) {
      if (!networkTrafficIds.includes(definition.id)) {
        throw new Error(`Unknown network traffic "${definition.id}".`);
      }
      definitions.set(definition.id, { ...definition });
    },
    get: (id) => definitions.get(id),
    list: () => [...definitions.values()],
  };

  // Off unless the User opted in on this device, and only if this copy can send them.
  const crashReportsOn = () =>
    crashReporter !== undefined && settings.readDeviceValue(CRASH_REPORTS) === true;
  // On unless the User turned them off.
  const updateChecksOn = () => settings.readDeviceValue(AUTOMATIC_UPDATE_CHECKS) !== false;

  const status = (): PrivacySettings => ({
    crashReports: { available: crashReporter !== undefined, enabled: crashReportsOn() },
    automaticUpdateChecks: updateChecksOn(),
    usageData: usageData.status(),
  });

  return {
    traffic: registry,
    status,

    /** Starts crash reports if the User opted in before. Nothing is sent before they do. */
    start(): void {
      if (crashReportsOn()) crashReporter?.setEnabled(true);
    },

    /** Returns whether anything changed, and the settings now in effect. */
    update(patch: unknown): { changed: boolean; settings: PrivacySettings } {
      if (!isRecord(patch)) throw new InvalidInputError("Privacy settings must be an object.");
      for (const [key, value] of Object.entries(patch)) {
        if (!PATCH_KEYS.has(key)) throw new InvalidInputError(`Unknown privacy setting "${key}".`);
        if (typeof value !== "boolean") {
          throw new InvalidInputError(`The privacy setting "${key}" must be true or false.`);
        }
      }
      const {
        crashReports,
        automaticUpdateChecks,
        usageData: shareUsage,
      } = patch as {
        crashReports?: boolean;
        automaticUpdateChecks?: boolean;
        usageData?: boolean;
      };
      if (crashReports === true && !crashReporter) {
        throw new InvalidInputError("This copy of IncarnaMind can't send crash reports.");
      }
      const before = status();
      // Checked before anything changes, so a refused patch changes nothing.
      if (shareUsage !== undefined && !before.usageData.available) {
        throw new InvalidInputError("This copy of IncarnaMind can't send usage data.");
      }
      if (shareUsage === true && before.usageData.localMode) {
        throw new InvalidInputError(
          'Usage data stays off while "Keep everything on this computer" is on.',
        );
      }

      if (shareUsage !== undefined) usageData.setEnabled(shareUsage);
      if (crashReports !== undefined && crashReports !== before.crashReports.enabled) {
        if (crashReports) {
          settings.writeDeviceValue(CRASH_REPORTS, true);
          crashReporter?.setEnabled(true);
        } else {
          // Stop first: opting out stops reporting at once, before anything else.
          crashReporter?.setEnabled(false);
          settings.writeDeviceValue(CRASH_REPORTS, false);
        }
      }
      if (
        automaticUpdateChecks !== undefined &&
        automaticUpdateChecks !== before.automaticUpdateChecks
      ) {
        settings.writeDeviceValue(AUTOMATIC_UPDATE_CHECKS, automaticUpdateChecks);
      }
      const after = status();
      const changed =
        after.crashReports.enabled !== before.crashReports.enabled ||
        after.automaticUpdateChecks !== before.automaticUpdateChecks ||
        after.usageData.enabled !== before.usageData.enabled ||
        after.usageData.asked !== before.usageData.asked;
      return { changed, settings: after };
    },

    /** Every registered traffic that can happen now, in the order registered. */
    async listTraffic(): Promise<NetworkTraffic[]> {
      const result: NetworkTraffic[] = [];
      for (const definition of definitions.values()) {
        if (definition.listed && !(await definition.listed())) continue;
        const enabled = definition.enabled ? await definition.enabled() : true;
        const services = definition.services
          ? await definition.services()
          : definition.service
            ? [definition.service]
            : [];
        for (const service of services) {
          result.push({ id: definition.id, service: { ...service }, enabled });
        }
      }
      return result;
    },
  };
}
