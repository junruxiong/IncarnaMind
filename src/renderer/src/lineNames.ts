import { nameInLine } from "../../shared/naming";
import type { Translate } from "./objects";

/** The placeholders of the dictionary that hold something the User named. */
const NAME_PARAMS: ReadonlySet<string> = new Set([
  "name",
  "title",
  "document",
  "mind",
  "folder",
  "tag",
  "into",
  "item",
]);

const cutting =
  (t: Translate): Translate =>
  (key, params) =>
    t(
      key,
      params &&
        Object.fromEntries(
          Object.entries(params).map(([each, value]) => [
            each,
            typeof value === "string" && NAME_PARAMS.has(each) ? nameInLine(value) : value,
          ]),
        ),
    );

/**
 * What `build` says with every name in it cut in the middle, for a line (DESIGN.md, Words on
 * screen); `fullMessage`, for the tooltip and the accessible name, only when a name was cut.
 */
export function saidInLine(
  t: Translate,
  build: (t: Translate) => string,
): { message: string; fullMessage?: string } {
  const message = build(cutting(t));
  const fullMessage = build(t);
  return message === fullMessage ? { message } : { message, fullMessage };
}
