import { NAME_IN_LINE_COLUMNS, shortenMiddle } from "../../shared/naming";
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
  (t: Translate, columns: number): Translate =>
  (key, params) =>
    t(
      key,
      params &&
        Object.fromEntries(
          Object.entries(params).map(([each, value]) => [
            each,
            typeof value === "string" && NAME_PARAMS.has(each)
              ? shortenMiddle(value, columns)
              : value,
          ]),
        ),
    );

export interface Said {
  message: string;
  /** `message` with every name whole, for the tooltip and the accessible name; absent when none was cut. */
  fullMessage?: string;
  /** The message with each name cut to at most `columns`, for a line with less room than `message` needs. */
  within(columns: number): string;
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
  const within = (columns: number) =>
    build(cutting(t, columns), (name) => shortenMiddle(name, columns));
  const message = within(NAME_IN_LINE_COLUMNS);
  const fullMessage = build(t, (name) => name);
  return message === fullMessage ? { message, within } : { message, fullMessage, within };
}
