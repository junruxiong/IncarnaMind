/**
 * An Answer of the assistant's conversations as Blocks (ADR-0017): the Markdown
 * the model wrote made into the node types of Notes, each Citation marker made
 * into its Citation, exactly as an Answer written into a Mind is made. The
 * assistant shows them, Copy copies them and Add to note puts them into a note.
 */

import { markdownToBlocks } from "../../core/answers/markdown";
import type { ConversationAnswer } from "../../core/api";
import { citationNode, type NodeJSON, withCitations } from "../../shared/blockJson";

/**
 * While it is written, what would still change how it looks is held back, and
 * a marker whose Citation hasn't come yet shows as being checked; once it has
 * ended, a marker with no Citation is left out.
 */
export function answerBlocks(
  answer: Pick<ConversationAnswer, "text" | "citations" | "status">,
): NodeJSON[] {
  const streaming = answer.status === "streaming";
  const cited = new Map(answer.citations.map(({ marker, ...citation }) => [marker, citation]));
  return withCitations(markdownToBlocks(answer.text, { streaming }), (marker) => {
    const citation = cited.get(marker);
    if (citation) return citationNode(citation);
    return streaming ? citationNode(null) : null;
  });
}

/** Whether Blocks hold any words: an Answer that failed before writing any has nothing to keep. */
export const hasText = (blocks: readonly NodeJSON[]): boolean =>
  blocks.some(
    (block) =>
      (block.type === "text" && (block.text ?? "").trim() !== "") || hasText(block.content ?? []),
  );
