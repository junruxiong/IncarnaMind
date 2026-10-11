/**
 * Find sources as a Tool an Answer can call (ADR-0013): "find papers since
 * 2021 on …" asked in the composer searches the same providers the Library
 * does, with the same consent, and the Answer lists what it found. What it
 * finds isn't the User's Documents, so it is never cited with a marker; the
 * User adds works from the call's results ("Show results"), and an added
 * open-access one becomes a Document the next Answer can cite. Its results
 * are public records from the providers, not the User's data, so the chat
 * flow doesn't count them as "tool-results" (../core: `sendsToolResults`).
 */
import { formatName } from "../../shared/references";
import type { FindSourcesInput, FindSourcesResult, FoundSource } from "../api";
import { InvalidInputError } from "../errors";
import { SOURCE_TOOLS, type Tool, type ToolProviderInfo } from "../tools";
import type { SourceProvider } from "./types";

const PROVIDER: ToolProviderInfo = { kind: "sources", id: "sources", name: "Find sources" };

/** How many works the model is shown: enough to choose from, few enough to fit a small window. */
const SHOWN = 15;

const yearOf = (value: unknown, what: string): number | null => {
  if (value === undefined || value === null) return null;
  if (typeof value !== "number" || !Number.isInteger(value)) {
    throw new InvalidInputError(`${what} must be a year, such as 2021.`);
  }
  return value;
};

const list = (names: readonly string[]) =>
  names.length <= 1
    ? (names[0] ?? "")
    : `${names.slice(0, -1).join(", ")} and ${names[names.length - 1]}`;

function describe(work: FoundSource, index: number): string {
  const { fields } = work;
  const authors = fields.authors.slice(0, 3).map(formatName);
  const parts = [
    `${index + 1}. "${fields.title}"`,
    authors.length > 0 ? `${authors.join(", ")}${fields.authors.length > 3 ? " et al." : ""}` : "",
    fields.issued ? `(${fields.issued.slice(0, 4)})` : "",
    fields.containerTitle ? `${fields.containerTitle}.` : "",
    fields.doi ? `DOI ${fields.doi}.` : "",
    work.access.kind === "open"
      ? `Open access (${
          work.access.format === "pdf"
            ? "PDF"
            : work.access.format === "web-page"
              ? "a web page"
              : "full text"
        }${
          work.access.licence ? `, ${work.access.licence.label}` : ""
        }${work.access.licence?.commercial === false ? ", non-commercial use only" : ""}).`
      : "No open full text: it can be added as a reference only.",
    work.citedBy !== null ? `Cited by ${work.citedBy}.` : "",
    work.inLibrary
      ? work.inLibrary.documentId
        ? "Already in the User's Library as a Document."
        : "Already in the User's Library as a reference."
      : "",
    `Found by ${list(work.foundBy.map((each) => each.name))}.`,
  ];
  return parts.filter(Boolean).join(" ");
}

/** What the model reads of a search. */
export function resultText(result: FindSourcesResult): string {
  if (result.outcome === "local-mode") {
    return "Find sources is off: the User keeps everything on this computer (local mode), so nothing was searched. Say so.";
  }
  const searched = result.providers.filter((each) => each.failure === null);
  const lines = [
    result.works.length === 0
      ? `Nothing was found${searched.length > 0 ? ` on ${list(searched.map((each) => each.name))}` : ""}.`
      : `Found ${result.works.length} works on ${list(searched.map((each) => each.name))}. They aren't the User's Documents: never put citation markers on them; name each by its title, first author and year. The User can add them to their Library from this search's results ("Show results", under this Answer); an open-access one then becomes a Document that can be cited.`,
    "",
    ...result.works.slice(0, SHOWN).map(describe),
  ];
  for (const each of result.providers) {
    if (!each.failure) continue;
    lines.push(
      "",
      each.failure.reason === "declined"
        ? `${each.name} wasn't searched: the User didn't allow it.`
        : `${each.name} couldn't be searched just now (${each.failure.reason}).`,
    );
  }
  return lines.join("\n").trim();
}

export function sourcesTool(
  search: (input: FindSourcesInput, signal: AbortSignal) => Promise<FindSourcesResult>,
  searching: readonly SourceProvider[],
): Tool {
  return {
    name: SOURCE_TOOLS.find,
    description: [
      `Searches ${list(searching.map((each) => each.name))} for published papers and preprints, outside the User's Documents.`,
      'Call it only when the User asks to find papers, articles, studies or other published sources (for example "find papers since 2021 on …"), not to answer from the User\'s Documents.',
      "Write the query as the key words of the topic (a title, a DOI or an arXiv ID also works), in the language the field publishes in, usually English.",
    ].join(" "),
    inputSchema: {
      type: "object",
      properties: {
        query: {
          type: "string",
          description: "The key words of the topic, a title, a DOI or an arXiv ID.",
        },
        from_year: { type: "integer", description: "Only works published in or after this year." },
        to_year: { type: "integer", description: "Only works published in or before this year." },
        open_access_only: {
          type: "boolean",
          description: "Only works whose full text is openly available.",
        },
      },
      required: ["query"],
      additionalProperties: false,
    },
    provider: PROVIDER,
    providerTool: SOURCE_TOOLS.find,
    title: "Find sources",
    // The query goes to each provider, as the User allowed (asked first, once each).
    effects: () =>
      searching.map((each) => ({
        action: "network" as const,
        scope: { kind: "service" as const, serviceId: each.service.id, name: each.service.name },
      })),
    untrustedResult: true,
    shownInput: (input) => ({
      query: input.query,
      ...(input.from_year !== undefined && { from_year: input.from_year }),
      ...(input.to_year !== undefined && { to_year: input.to_year }),
      ...(input.open_access_only !== undefined && { open_access_only: input.open_access_only }),
    }),
    async call(input, context) {
      if (typeof input.query !== "string") throw new InvalidInputError("query must be text.");
      const result = await search(
        {
          query: input.query,
          fromYear: yearOf(input.from_year, "from_year"),
          toYear: yearOf(input.to_year, "to_year"),
          openAccessOnly: input.open_access_only === true,
        },
        context.signal,
      );
      context.counted?.(result.outcome === "found" ? result.works.length : 0);
      return resultText(result);
    },
  };
}
