import { type GenerateOptions, lastUserText, systemOf } from "./prompt";

/** Finding a verified document's claims (#97, src/core/answers/checks/claims.ts). */
const FIND_CLAIMS = /^You sort the sentences of a document someone else wrote/m;

/** The numbers of the listed sentences ("[3] …") that claim something: all but questions and thanks. */
export function fakeClaims(options: GenerateOptions): string | null {
  if (!FIND_CLAIMS.test(systemOf(options.prompt))) return null;
  const claims = lastUserText(options.prompt)
    .split("\n")
    .flatMap((line) => {
      const listed = /^\[(\d+)\] (.*)$/.exec(line);
      return listed?.[2] === undefined || /[?？]$|thank|谢/i.test(listed[2])
        ? []
        : [Number(listed[1])];
    });
  return JSON.stringify({ claims });
}
