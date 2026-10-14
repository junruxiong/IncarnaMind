/** The reference list of a Mind's Blocks, as ../../shared/referenceList reads it. */
import type * as Y from "yjs";
import { type ListedReference, referenceEntries } from "../../shared/referenceList";
import { INLINE_NODE } from "../../shared/sentences";
import { textOf } from "../answers/checks/sentences";
import { BLOCK_ID_ATTRIBUTE } from "../api";
import { textBlocks } from "../proposals/write";

export function referenceList(blocks: Y.XmlFragment): ListedReference[] {
  return referenceEntries(
    textBlocks(blocks).flatMap(({ element }) => {
      const id = element.getAttribute(BLOCK_ID_ATTRIBUTE);
      return typeof id === "string"
        ? [
            {
              id,
              heading: element.nodeName === "heading",
              text: textOf(element).replaceAll(INLINE_NODE, " "),
            },
          ]
        : [];
    }),
  );
}
