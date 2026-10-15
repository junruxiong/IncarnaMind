/**
 * The scripted model's Edit with AI replies that the guards on an edit
 * applying at once are for (#880), as real models wrote them.
 */

/** The second paragraph, made formal as Claude Haiku 5.5 made it, word for word kept where it could. */
const formal = (paragraph: string) =>
  paragraph
    .replace(/\bIt['’]s\b/g, "It is")
    .replace(/\bdon['’]t\b/g, "do not")
    .replace(/\ba lot of\b/g, "a great many")
    .replace(/\bbig\b/g, "considerable");

/**
 * The new text for `passage`, or null for the scripted model's other replies:
 * - "make the second paragraph more formal; leave the others as they are"
 *   gives only that paragraph back, as Haiku did with the whole note selected;
 * - "delete the second paragraph" gives the others back as they were;
 * - "replace it all with a haiku" gives three lines like none of the passage's.
 */
export function appliedEditReply(passage: string, asked: string): string | null {
  const paragraphs = passage.split(/\n\n+/);
  if (/second paragraph more formal|第二段.*正式/iu.test(asked)) return formal(paragraphs[1] ?? "");
  if (/^delete the second paragraph|^删除第二段/iu.test(asked)) {
    return [paragraphs[0], ...paragraphs.slice(2)].join("\n\n");
  }
  if (/haiku|俳句/iu.test(asked)) {
    return "Gulls wheel, crying\n\nsalt wind on stone\n\nnight falls soft";
  }
  return null;
}
