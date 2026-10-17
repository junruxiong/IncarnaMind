import * as Y from "yjs";
import { isRecord } from "../core/errors";

const textOfRun = (run: unknown): string =>
  isRecord(run) && typeof run.insert === "string" ? run.insert : "";

const textOf = (element: Y.XmlElement): string =>
  element
    .toArray()
    .map((child) => (child instanceof Y.XmlText ? child.toDelta().map(textOfRun).join("") : ""))
    .join("");

/**
 * Whether a Mind the assistant made holds no more than it was made with:
 * empty lines, and the heading of its title its text starts with (#908).
 * Rejecting its changes then takes the Mind away too.
 */
export function holdsOnlyItsTitle(blocks: Y.XmlFragment, title: string): boolean {
  return blocks
    .toArray()
    .filter((block): block is Y.XmlElement => block instanceof Y.XmlElement)
    .every((block, index) =>
      block.nodeName === "paragraph"
        ? block.length === 0
        : index === 0 && block.nodeName === "heading" && textOf(block) === title,
    );
}
