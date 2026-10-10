/**
 * Usage data (#187): whether this device sends it, the install ID, and the
 * gate every event goes through.
 *
 * Only a copy built with an analytics project has a sender (see
 * `UsageDataSender`); without one, nothing here sends or asks anything. With
 * one, usage data is sent only while the User agrees:
 * - a release build sends nothing until the User says yes, asked once on
 *   the first run; the default is no;
 * - a test build (the alpha) sends until the User says no, as testers agree
 *   to when they join; the first run says so, with the switch;
 * - local mode ("Keep everything on this computer") turns it off, and keeps
 *   it off while it is on;
 * - turning it off stops sending at once, and the sender drops what is queued.
 *
 * Every event is checked against the catalog (./usageEvents) before it goes,
 * and carries the install ID: a random UUID, made the first time an event is
 * sent, which the User can reset. Nothing else identifies the install.
 */
import { randomUUID } from "node:crypto";
import type { UsageDataSender } from "./adapters";
import type { UsageDataSettings } from "./api";
import { InvalidInputError } from "./errors";
import type { SettingsStore } from "./settings";
import {
  InvalidUsageEventError,
  parseCommonFields,
  parseUsageEvent,
  UI_USAGE_EVENTS,
  type UsageEvent,
} from "./usageEvents";

/** Per-device values, outside `DeviceSettings` so only this module changes them. */
const CHOICE = "usageData.enabled";
const ASKED = "usageData.asked";
const INSTALL_ID = "usageData.installId";

const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/;

const osOf = (platform: string) =>
  platform === "darwin" || platform === "win32" || platform === "linux" ? platform : "other";
const archOf = (arch: string) => (arch === "arm64" || arch === "x64" ? arch : "other");

export type UsageData = ReturnType<typeof createUsageData>;

export function createUsageData(options: {
  settings: SettingsStore;
  /** Absent when this copy can't send usage data. */
  sender: UsageDataSender | undefined;
  /** Whether local mode is on. */
  localOnly: () => boolean;
  /** Defaults to the console. */
  reportError?: (error: unknown) => void;
}) {
  const { settings, sender, localOnly } = options;
  const reportError = options.reportError ?? ((error: unknown) => console.error(error));

  /** The User's choice on this device, or null before they made one. */
  const choice = (): boolean | null => {
    const value = settings.readDeviceValue(CHOICE);
    return typeof value === "boolean" ? value : null;
  };
  const enabled = () => sender !== undefined && !localOnly() && (choice() ?? sender.testerBuild);

  /** What the sender was last told. */
  let sending = false;
  /** Tells the sender when sending starts or stops. */
  const apply = () => {
    const next = enabled();
    if (next === sending) return;
    sending = next;
    sender?.setEnabled(next);
  };

  const installId = (): string => {
    const stored = settings.readDeviceValue(INSTALL_ID);
    if (typeof stored === "string" && UUID.test(stored)) return stored;
    const id = randomUUID();
    settings.writeDeviceValue(INSTALL_ID, id);
    return id;
  };

  // Set by the core, never by a caller, and checked like any field.
  const common = sender
    ? parseCommonFields(sender.appVersion, {
        app_version: sender.appVersion,
        os: osOf(process.platform),
        arch: archOf(process.arch),
        build: sender.testerBuild ? "tester" : "release",
      })
    : null;

  /** Sends an event that has been checked, if sending is on. */
  const send = (event: UsageEvent) => {
    if (!sending || !sender || !common) return;
    sender.capture({
      installId: installId(),
      event: event.event,
      properties: { ...event.fields, ...common },
    });
  };

  return {
    /** Starts sending if the User agreed before (or, in a test build, didn't say no). */
    start(): void {
      apply();
    },

    status(): UsageDataSettings {
      return {
        available: sender !== undefined,
        enabled: enabled(),
        testerBuild: sender?.testerBuild ?? false,
        asked: settings.readDeviceValue(ASKED) === true,
        localMode: localOnly(),
      };
    },

    /** The User's answer, from the first run or the Privacy page. */
    setEnabled(on: boolean): void {
      if (!sender) throw new InvalidInputError("This copy of IncarnaMind can't send usage data.");
      if (on && localOnly()) {
        throw new InvalidInputError(
          'Usage data stays off while "Keep everything on this computer" is on.',
        );
      }
      if (!on) {
        // Stop first: nothing more goes out, and what is queued is dropped.
        sending = false;
        sender.setEnabled(false);
      }
      settings.writeDeviceValue(CHOICE, on);
      settings.writeDeviceValue(ASKED, true);
      apply();
    },

    /** Local mode was turned on or off: on, it turns usage data off, and the choice stays off. */
    localModeChanged(): void {
      if (sender && localOnly() && choice() !== false) {
        sending = false;
        sender.setEnabled(false);
        settings.writeDeviceValue(CHOICE, false);
      }
      apply();
    },

    resetInstallId(): void {
      settings.writeDeviceValue(INSTALL_ID, randomUUID());
    },

    /**
     * An event from the core itself. A malformed one is a bug: it is reported
     * and dropped, and never gets in the way of what the User is doing.
     */
    record(event: UsageEvent): void {
      if (!sending) return;
      try {
        send(parseUsageEvent(event.event, event.fields));
      } catch (error) {
        reportError(error);
      }
    },

    /** An event from the interface: checked whether or not it is sent, so a bad one fails loudly. */
    recordFromUi(input: unknown): void {
      const { event, fields } = (input ?? {}) as { event?: unknown; fields?: unknown };
      if (!UI_USAGE_EVENTS.some((name) => name === event)) {
        throw new InvalidInputError(`The interface doesn't send the usage event "${event}".`);
      }
      let parsed: UsageEvent;
      try {
        parsed = parseUsageEvent(event, fields);
      } catch (error) {
        if (error instanceof InvalidUsageEventError) throw new InvalidInputError(error.message);
        throw error;
      }
      send(parsed);
    },
  };
}
