/**
 * The Connectors directory: the remote Connectors IncarnaMind has reviewed,
 * which Settings › Connectors lists under Browse. The list is data
 * (./directory.json), checked here when the core starts, so it changes
 * without touching the code that shows it.
 */
import {
  type ConnectorDirectoryEntry,
  type ConnectorDirectorySignIn,
  connectorDirectoryGroups,
  type LocalizedText,
} from "../api";
import { isRecord } from "../errors";
import type { Language } from "../language";
import { parseRemoteUrl } from "./config";
import shipped from "./directory.json";

const LANGUAGES: readonly Language[] = ["en", "zh-CN"];
const isLanguage = (value: unknown): value is Language => LANGUAGES.some((each) => each === value);
const SIGN_INS: readonly ConnectorDirectorySignIn[] = ["browser", "own-app", "none"];

const text = (value: unknown, where: string): string => {
  if (typeof value !== "string" || !value.trim()) throw new Error(`${where} must be text.`);
  return value;
};

function localized(value: unknown, where: string): LocalizedText {
  if (!isRecord(value)) throw new Error(`${where} must have its words in each language.`);
  return { en: text(value.en, `${where}.en`), "zh-CN": text(value["zh-CN"], `${where}.zh-CN`) };
}

function languagesOf(value: unknown, where: string): Language[] {
  if (!Array.isArray(value) || !value.every(isLanguage)) {
    throw new Error(`${where} lists an unknown language.`);
  }
  return [...value];
}

function entryOf(value: unknown, index: number): ConnectorDirectoryEntry {
  const where = `Connector directory entry ${index}`;
  if (!isRecord(value)) throw new Error(`${where} must be an object.`);
  const group = connectorDirectoryGroups.find((each) => each === value.group);
  if (!group) throw new Error(`${where} has an unknown group: ${String(value.group)}.`);
  const signIn = SIGN_INS.find((each) => each === value.signIn);
  if (!signIn) throw new Error(`${where} has an unknown sign-in: ${String(value.signIn)}.`);
  const subscription = value.subscription;
  if (
    subscription !== null &&
    subscription !== "free" &&
    !(isRecord(subscription) && typeof subscription.product === "string")
  ) {
    throw new Error(`${where} has an unknown subscription.`);
  }
  const languages = value.languages === undefined ? undefined : languagesOf(value.languages, where);
  return {
    id: text(value.id, `${where}.id`),
    name: text(value.name, `${where}.name`),
    publisher: text(value.publisher, `${where}.publisher`),
    group,
    summary: localized(value.summary, `${where}.summary`),
    subscription: isRecord(subscription)
      ? { product: text(subscription.product, `${where}.subscription.product`) }
      : subscription,
    url: parseRemoteUrl(value.url),
    signIn,
    ...(languages && { languages }),
  };
}

/**
 * The directory's entries, checked: each with a known group and sign-in, its
 * words in every language, an https URL, and an id and URL no other entry has.
 * Throws on the first that isn't, naming it: a broken list is a bug to fix
 * before release, not something to show half of.
 */
export function connectorDirectory(list: unknown = shipped): ConnectorDirectoryEntry[] {
  if (!Array.isArray(list)) throw new Error("The Connector directory must be a list.");
  const entries = list.map(entryOf);
  for (const [index, entry] of entries.entries()) {
    const earlier = entries.slice(0, index);
    if (earlier.some((each) => each.id === entry.id)) {
      throw new Error(`Two Connector directory entries have the id "${entry.id}".`);
    }
    if (earlier.some((each) => each.url === entry.url)) {
      throw new Error(`Two Connector directory entries have the URL ${entry.url}.`);
    }
  }
  return entries;
}
