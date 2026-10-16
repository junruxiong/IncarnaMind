import { EXCERPT_END, EXCERPT_START } from "../../core/tags/classify";
import type { GenerateOptions, Prompt } from "./prompt";
import { fakeSuggestion } from "./suggestion";

const escapeRegExp = (text: string) => text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

function excerptOf(prompt: Prompt): string {
  const text = prompt
    .map((message) =>
      typeof message.content === "string"
        ? message.content
        : message.content.map((part) => (part.type === "text" ? part.text : "")).join(""),
    )
    .join("\n");
  // The instructions name the markers too: the excerpt is between the last ones.
  return text.slice(text.lastIndexOf(EXCERPT_START), text.lastIndexOf(EXCERPT_END));
}

const unreachedOnce = new Set<string>();

/** A Document whose text says "unreachable" finds the model unreachable the first time it is tagged, as offline. */
export function tagsUnreachable(options: GenerateOptions): boolean {
  const format = options.responseFormat;
  const schema = format?.type === "json" ? format.schema : undefined;
  if (!schema?.properties?.tags || schema.properties.groupId) return false;
  const excerpt = excerptOf(options.prompt);
  if (!/\bunreachable\b/i.test(excerpt) || unreachedOnce.has(excerpt)) return false;
  unreachedOnce.add(excerpt);
  return true;
}

/**
 * The scripted tagger: of the Tag names offered, those that appear as a word
 * (ignoring case) in the Document's excerpt, i.e. its name or text.
 */
function fakeTags(prompt: Prompt, offered: readonly string[]): string[] {
  const excerpt = excerptOf(prompt);
  return offered.filter((name) =>
    new RegExp(`(?<![\\p{L}\\p{N}])${escapeRegExp(name)}(?![\\p{L}\\p{N}])`, "iu").test(excerpt),
  );
}

/** The Tag names a tagging request's structured output may choose from. */
function offeredTags(options: GenerateOptions): string[] {
  const format = options.responseFormat;
  const schema = (format?.type === "json" ? format.schema : undefined) as
    | { properties?: { tags?: { items?: { enum?: unknown[] } } } }
    | undefined;
  const names: unknown[] = schema?.properties?.tags?.items?.enum ?? [];
  return names.filter((name): name is string => typeof name === "string");
}

/** Test-only classifier: chooses a group whose name appears in the excerpt. */
export function fakeStructuredOutput(options: GenerateOptions): string {
  const format = options.responseFormat;
  const schema = format?.type === "json" ? format.schema : undefined;
  if (schema?.properties?.quote && schema.properties.passage) return fakeSuggestion(options);
  if (schema?.properties?.groupId) {
    const message = options.prompt.find((part) => part.role === "user");
    const text =
      message && typeof message.content !== "string"
        ? message.content
            .filter((part) => part.type === "text")
            .map((part) => part.text)
            .join("")
        : "";
    const input = JSON.parse(text) as {
      groups: { id: string; name: string }[];
      tags?: { id: string; name: string }[];
      document: { name: string; text: string };
    };
    const document = `${input.document.name} ${input.document.text}`.toLowerCase();
    const selected = input.groups.find((group) => document.includes(group.name.toLowerCase()));
    return JSON.stringify({
      groupId: selected?.id ?? "__unsorted__",
      tags: (input.tags ?? [])
        .filter((tag) => document.includes(tag.name.toLowerCase()))
        .map((tag) => tag.id),
    });
  }
  return JSON.stringify({ tags: fakeTags(options.prompt, offeredTags(options)) });
}
