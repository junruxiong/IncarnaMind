const formal = (paragraph: string) =>
  paragraph
    .replace(/\bIt['’]s\b/g, "It is")
    .replace(/\bdon['’]t\b/g, "do not")
    .replace(/\ba lot of\b/g, "a great many")
    .replace(/\bbig\b/g, "considerable");

/**
 * Replies the guards on an applied edit are for (#880), or null: "make the
 * second paragraph more formal" gives only that paragraph back, as Claude
 * Haiku 5.5 did with the whole note selected; "add a closing paragraph" adds
 * a Block after the passage.
 */
export function appliedEditReply(passage: string, asked: string): string | null {
  const paragraphs = passage.split(/\n\n+/);
  if (/second paragraph more formal|第二段.*正式/iu.test(asked)) return formal(paragraphs[1] ?? "");
  if (/^delete the second paragraph|^删除第二段/iu.test(asked)) {
    return [paragraphs[0], ...paragraphs.slice(2)].join("\n\n");
  }
  if (/add a closing paragraph|加一段结尾/iu.test(asked))
    return `${passage}\n\nThat is all for now.`;
  if (/haiku|俳句/iu.test(asked)) {
    return "Gulls wheel, crying\n\nsalt wind on stone\n\nnight falls soft";
  }
  return null;
}

/** Asked of the assistant, "replace X with Y" is its own change to the Mind, which waits for the User. */
export const ASSISTANT_REPLACES = /^replace\s+.+\s+with\s+/i;
