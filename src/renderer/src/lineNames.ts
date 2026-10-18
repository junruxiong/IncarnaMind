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

export interface Said {
  message: string;
  /** `message` with every name whole, for the tooltip and the accessible name; absent when none was cut. */
  fullMessage?: string;
}

/**
 * What `build` says with every name in it cut in the middle, for a line (DESIGN.md, Words on
 * screen). A name given to `t` as a placeholder is cut by it; one made part of a bigger value
 * first, such as a quoted list, goes through `cut`.
 */
export function saidInLine(
  t: Translate,
  build: (t: Translate, cut: (name: string) => string) => string,
): Said {
  const message = build(cutting(t), nameInLine);
  const fullMessage = build(t, (name) => name);
  return message === fullMessage ? { message } : { message, fullMessage };
}
