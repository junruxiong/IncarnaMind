/** The references a check looks up in a Mind's Blocks, as ../../shared/referenceList reads them. */
import type * as Y from "yjs";
import { checkableReferences, type ListedReference } from "../../shared/referenceList";
import { INLINE_NODE } from "../../shared/sentences";
import { textOf } from "../answers/checks/sentences";
import { BLOCK_ID_ATTRIBUTE } from "../api";
import { textBlocks } from "../proposals/write";

export function referenceList(blocks: Y.XmlFragment): ListedReference[] {
  return checkableReferences(
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
