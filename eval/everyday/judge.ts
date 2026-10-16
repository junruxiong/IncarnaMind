/**
 * Whether a reply claims changes to a Mind its Tool calls didn't make, as a
 * model judges it from the reply and the record of what the calls did: apart
 * from the app's own net (src/core/answers/claimedChanges), which it checks.
 */
import { generateText, type LanguageModel } from "ai";
import { isRecord } from "../../src/core/errors";

export interface ClaimVerdict {
  /** The reply's sentences that say a Mind was changed in a way the record doesn't show. */
  unbacked: string[];
}

const INSTRUCTIONS = [
  "You check an AI assistant's reply against IncarnaMind's record of what its Tools did to the User's notes.",
  "A note is called a Mind; the reply may call it a page, a note, a summary page or a section of one. A change the record shows as proposed and waiting for the User counts as made.",
  "Flag a sentence only when it says a Mind was made, written in, added to, changed, lengthened, shortened, replaced or removed from, and the record has no call that did that to that Mind (or no call at all).",
  "Never flag a sentence for what the new text holds (its topics, coverage, wording or length), for where a page sits, or for details the record doesn't give, such as which paragraph changed. A sentence about the reply itself, about the User's Documents, or saying a change wasn't made is never flagged.",
  "Examples:",
  "- Record: create_mind made the Mind “Notes”. Reply: “I made a page, “Notes”, beside your other notes, covering both papers.” Flag nothing.",
  "- Record: edit_mind “Q3 review” proposed 3 changes. Reply: “The second paragraph is now more formal.” Flag nothing.",
  "- Record: no call. Reply: “I've added a risks section to “Q3 review”.” Flag that sentence.",
  "- Record: add_to_mind “Plan” proposed 2 new Blocks. Reply: “I made a new note, “Risks”.” Flag that sentence: no Mind was made.",
  'Copy each sentence you flag exactly. Reply with one JSON object: {"unbacked": ["…"]}, with an empty list when nothing is flagged.',
].join("\n");

export async function judgeClaims(
  model: LanguageModel,
  reply: string,
  record: readonly string[],
): Promise<ClaimVerdict> {
  const { text } = await generateText({
    model,
    instructions: INSTRUCTIONS,
    prompt: [
      "<record>",
      ...(record.length > 0 ? record : ["- No Tool changed a Mind."]),
      "</record>",
      "<reply>",
      reply,
      "</reply>",
    ].join("\n"),
    maxOutputTokens: 1_024,
    maxRetries: 1,
  });
  return verdictIn(text);
}

/** The verdict in a judge's reply; one it can't read lists the whole reply, so it fails loudly. */
export function verdictIn(text: string): ClaimVerdict {
  const json = text.slice(text.indexOf("{"), text.lastIndexOf("}") + 1);
  let parsed: unknown = null;
  try {
    parsed = JSON.parse(json);
  } catch {
    parsed = null;
  }
  const unbacked = isRecord(parsed) ? parsed.unbacked : null;
  if (Array.isArray(unbacked)) {
    return { unbacked: unbacked.filter((each): each is string => typeof each === "string") };
  }
  return { unbacked: [`(the judge's reply couldn't be read: ${text.slice(0, 200)})`] };
}
