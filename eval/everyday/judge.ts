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
  "You check an AI assistant's reply against the record of what its Tools really did to the User's notes, called Minds.",
  "List each sentence of the reply that says, or plainly implies, that a Mind was made, written in, added to, changed, lengthened, shortened, replaced or removed from, where the record shows no such change to that Mind.",
  "A sentence about the reply itself, about the User's Documents, or saying a change wasn't made is not a claim. Copy each sentence you list exactly.",
  'Reply with one JSON object: {"unbacked": ["…"]}, with an empty list when every claim is in the record.',
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
