/**
 * SKILL.md: YAML frontmatter, then the Skill's instructions in Markdown
 * (https://agentskills.io/specification). The frontmatter is read with a small
 * subset of YAML, enough for the format's fields and how Skills found online
 * write them: `key: value` pairs, quoted values, block scalars (`|` and `>`),
 * plain values that run on over indented lines, one level of nested mapping
 * (`metadata`), and lists of plain values. Every value is kept as text, so
 * `version: 1.0` stays "1.0". Anything else (anchors, flow collections, tags)
 * is kept as its raw text, which is fine for fields the format doesn't define.
 */

/** The fields IncarnaMind uses, validated. Unknown fields are allowed and ignored. */
export interface SkillFrontmatter {
  name: string;
  description: string;
  license: string | null;
  compatibility: string | null;
  metadata: Record<string, string>;
}

export type ParsedSkillMd =
  | { ok: true; frontmatter: SkillFrontmatter; body: string }
  | { ok: false; field: string | null; message: string };

/** The format's limits. */
const MAX_NAME_LENGTH = 64;
const MAX_DESCRIPTION_LENGTH = 1024;
const MAX_COMPATIBILITY_LENGTH = 500;
/** Lowercase letters and digits, in runs joined by single hyphens. */
const NAME = /^[a-z0-9]+(?:-[a-z0-9]+)*$/;

type YamlValue = string | null | YamlValue[] | { [key: string]: YamlValue };

class YamlError extends Error {}

interface Line {
  indent: number;
  /** Without its indentation. */
  text: string;
  /** 1-based, within the frontmatter. */
  number: number;
}

const isBlank = (line: Line) => line.text === "" || line.text.startsWith("#");

/** A plain value's comment (" #…") removed. */
const withoutComment = (text: string) => text.replace(/\s+#.*$/, "");

const KEY = /^("(?:[^"\\]|\\.)*"|'(?:[^']|'')*'|[^\s:#'"][^:]*?)\s*:(?:\s+|$)/;

function unquoteKey(key: string): string {
  if (key.startsWith('"')) return parseDoubleQuoted(key);
  if (key.startsWith("'")) return key.slice(1, -1).replace(/''/g, "'");
  return key;
}

const ESCAPES: Readonly<Record<string, string>> = {
  "0": "\0",
  a: "\x07",
  b: "\b",
  t: "\t",
  "\t": "\t",
  n: "\n",
  v: "\v",
  f: "\f",
  r: "\r",
  e: "\x1b",
  " ": " ",
  '"': '"',
  "/": "/",
  "\\": "\\",
  N: "\u0085",
  _: " ",
};

/** A double-quoted scalar, quotes included, with its escapes resolved. */
function parseDoubleQuoted(quoted: string): string {
  const inner = quoted.slice(1, -1);
  return inner.replace(/\\(x[0-9a-fA-F]{2}|u[0-9a-fA-F]{4}|U[0-9a-fA-F]{8}|.)/g, (_, sequence) => {
    const code = /^[xuU]/.test(sequence) ? Number.parseInt(sequence.slice(1), 16) : null;
    if (code !== null) return String.fromCodePoint(code);
    const replaced = ESCAPES[sequence as string];
    if (replaced === undefined) throw new YamlError(`Unknown escape "\\${sequence}".`);
    return replaced;
  });
}

/** Lines of a quoted scalar that spans lines, folded the way YAML folds them. */
function foldQuoted(parts: readonly string[]): string {
  let text = parts[0] ?? "";
  for (let index = 1; index < parts.length; index++) {
    const part = (parts[index] ?? "").trim();
    if (part === "") text += "\n";
    else text += text.endsWith("\n") || text === "" ? part : ` ${part}`;
  }
  return text;
}

/** Parses the frontmatter's lines as a mapping. */
function parseYaml(source: string): Record<string, YamlValue> {
  const lines: Line[] = source.split(/\r?\n/).map((raw, index) => {
    const indentation = /^[ \t]*/.exec(raw)?.[0] ?? "";
    if (indentation.includes("\t") && raw.trim() !== "") {
      throw new YamlError(`Line ${index + 1} is indented with a tab; YAML needs spaces.`);
    }
    return {
      indent: indentation.length,
      text: raw.slice(indentation.length).trimEnd(),
      number: index + 1,
    };
  });
  let at = 0;

  const skipBlank = () => {
    while (at < lines.length && isBlank(lines[at] as Line)) at++;
  };

  /** A block scalar (`|` or `>`) whose header is at `parentIndent`. */
  function blockScalar(header: string, parentIndent: number): string {
    const match = /^([|>])([+-]?)(\d?)([+-]?)$/.exec(header);
    if (!match) throw new YamlError(`"${header}" isn't a block scalar header.`);
    const folded = match[1] === ">";
    const chomp = match[2] || match[4];
    const explicit = match[3] ? Number(match[3]) : null;
    const body: Line[] = [];
    while (at < lines.length) {
      const line = lines[at] as Line;
      if (line.text !== "" && line.indent <= parentIndent) break;
      body.push(line);
      at++;
    }
    const first = body.find((line) => line.text !== "");
    const indent = explicit !== null ? parentIndent + explicit : (first?.indent ?? 0);
    const texts = body.map((line) =>
      line.text === "" ? "" : " ".repeat(Math.max(0, line.indent - indent)) + line.text,
    );
    let text: string;
    if (folded) {
      text = "";
      texts.forEach((part, index) => {
        if (index === 0) text = part;
        else if (part === "" || part.startsWith(" ")) text += `\n${part}`;
        else text += text.endsWith("\n") || text === "" ? part : ` ${part}`;
      });
    } else {
      text = texts.join("\n");
    }
    const content = text.replace(/\n+$/, "");
    if (chomp === "-") return content;
    if (chomp === "+") return `${text}\n`;
    return content === "" ? "" : `${content}\n`;
  }

  /** A quoted scalar starting at `start` (quote included), maybe running on over the next lines. */
  function quoted(start: string, keyIndent: number): string {
    const quote = start[0] as string;
    const closes = (text: string) =>
      quote === '"' ? /(?:^|[^\\])(?:\\\\)*"\s*(?:#.*)?$/.test(text) : /'\s*(?:#.*)?$/.test(text);
    const parts = [start.slice(1)];
    let closed = start.length > 1 && closes(start.slice(1));
    while (!closed) {
      const line = lines[at];
      if (!line || (line.text !== "" && line.indent <= keyIndent)) {
        throw new YamlError(`A ${quote}-quoted value isn't closed.`);
      }
      parts.push(line.text);
      at++;
      closed = closes(line.text);
    }
    const joined = foldQuoted(parts);
    const end = joined.lastIndexOf(quote);
    const inner = joined.slice(0, end);
    return quote === '"' ? parseDoubleQuoted(`"${inner}"`) : inner.replace(/''/g, "'");
  }

  /** A plain scalar, continued by more indented lines. */
  function plain(first: string, keyIndent: number): string | null {
    const parts = [withoutComment(first).trim()];
    while (at < lines.length) {
      const line = lines[at] as Line;
      if (line.text === "") {
        // A blank line inside a plain value is a line break, if the value goes on after it.
        const next = lines.slice(at).find((each) => each.text !== "");
        if (!next || next.indent <= keyIndent) break;
        parts.push("");
        at++;
        continue;
      }
      if (line.indent <= keyIndent || line.text.startsWith("#")) break;
      parts.push(withoutComment(line.text).trim());
      at++;
    }
    const text = foldQuoted(parts).trim();
    return text === "" || text === "~" || text === "null" ? null : text;
  }

  /** The value after `key:` on a line at `indent`; `rest` is what follows the colon. */
  function value(rest: string, indent: number): YamlValue {
    if (rest === "" || rest.startsWith("#")) {
      skipBlank();
      const next = lines[at];
      if (!next || next.indent <= indent) {
        // A list may sit at the same indent as its key.
        if (next && next.indent === indent && /^-(\s|$)/.test(next.text)) return sequence(indent);
        return null;
      }
      return /^-(\s|$)/.test(next.text) ? sequence(next.indent) : mapping(next.indent);
    }
    if (rest.startsWith("|") || rest.startsWith(">"))
      return blockScalar(withoutComment(rest), indent);
    if (rest.startsWith('"') || rest.startsWith("'")) return quoted(rest, indent);
    return plain(rest, indent);
  }

  /** "- item" lines at `indent`. */
  function sequence(indent: number): YamlValue[] {
    const items: YamlValue[] = [];
    for (skipBlank(); at < lines.length; skipBlank()) {
      const line = lines[at] as Line;
      if (line.indent !== indent || !/^-(\s|$)/.test(line.text)) break;
      at++;
      items.push(value(line.text.slice(1).trim(), indent));
    }
    return items;
  }

  /** "key: value" lines at `indent`. */
  function mapping(indent: number): Record<string, YamlValue> {
    const map: Record<string, YamlValue> = {};
    for (skipBlank(); at < lines.length; skipBlank()) {
      const line = lines[at] as Line;
      if (line.indent < indent) break;
      if (line.indent > indent) throw new YamlError(`Line ${line.number} is indented too far.`);
      const match = KEY.exec(line.text);
      if (!match) throw new YamlError(`Line ${line.number} isn't "key: value".`);
      const key = unquoteKey(match[1] as string);
      if (Object.hasOwn(map, key)) throw new YamlError(`"${key}" is given twice.`);
      at++;
      map[key] = value(line.text.slice(match[0].length).trim(), indent);
    }
    return map;
  }

  skipBlank();
  const root = mapping(lines[at]?.indent ?? 0);
  skipBlank();
  if (at < lines.length) throw new YamlError(`Line ${(lines[at] as Line).number} is out of place.`);
  return root;
}

const FENCE = /^---[ \t]*$/;
const END = /^(?:---|\.\.\.)[ \t]*$/;

/** The frontmatter's text and the body after it, or null if SKILL.md doesn't start with one. */
function splitFrontmatter(text: string): { yaml: string; body: string } | null {
  const lines = text.replace(/^﻿/, "").split(/\r?\n/);
  if (!FENCE.test(lines[0] ?? "")) return null;
  const end = lines.findIndex((line, index) => index > 0 && END.test(line));
  if (end < 0) return null;
  return {
    yaml: lines.slice(1, end).join("\n"),
    body: lines
      .slice(end + 1)
      .join("\n")
      .replace(/^\s*\n/, "")
      .trimEnd(),
  };
}

const failure = (field: string | null, message: string): ParsedSkillMd => ({
  ok: false,
  field,
  message,
});

/** Reads SKILL.md's text: its frontmatter, checked against the format, and its instructions. */
export function parseSkillMd(text: string): ParsedSkillMd {
  const split = splitFrontmatter(text);
  if (!split) {
    return failure(null, "SKILL.md must start with frontmatter between two lines of ---.");
  }
  let fields: Record<string, YamlValue>;
  try {
    fields = parseYaml(split.yaml);
  } catch (error) {
    if (error instanceof YamlError)
      return failure(null, `The frontmatter isn't valid YAML: ${error.message}`);
    throw error;
  }

  const name = fields.name;
  if (typeof name !== "string" || name === "") return failure("name", "name is missing.");
  if (name.length > MAX_NAME_LENGTH) {
    return failure("name", `name must be at most ${MAX_NAME_LENGTH} characters.`);
  }
  if (!NAME.test(name)) {
    return failure(
      "name",
      "name may only have lowercase letters, digits and single hyphens, and can't start or end with a hyphen.",
    );
  }

  const description = typeof fields.description === "string" ? fields.description.trim() : "";
  if (!description) return failure("description", "description is missing.");
  if (description.length > MAX_DESCRIPTION_LENGTH) {
    return failure(
      "description",
      `description must be at most ${MAX_DESCRIPTION_LENGTH} characters.`,
    );
  }

  const license = fields.license ?? null;
  if (license !== null && typeof license !== "string") {
    return failure("license", "license must be text.");
  }

  const compatibility = fields.compatibility ?? null;
  if (compatibility !== null) {
    if (typeof compatibility !== "string") {
      return failure("compatibility", "compatibility must be text.");
    }
    if (compatibility.trim().length > MAX_COMPATIBILITY_LENGTH) {
      return failure(
        "compatibility",
        `compatibility must be at most ${MAX_COMPATIBILITY_LENGTH} characters.`,
      );
    }
  }

  const metadata: Record<string, string> = {};
  const rawMetadata = fields.metadata ?? null;
  if (rawMetadata !== null) {
    if (typeof rawMetadata !== "object" || Array.isArray(rawMetadata)) {
      return failure("metadata", "metadata must map names to text.");
    }
    for (const [key, item] of Object.entries(rawMetadata)) {
      if (typeof item !== "string") return failure("metadata", `metadata.${key} must be text.`);
      metadata[key] = item;
    }
  }

  const allowedTools = fields["allowed-tools"] ?? null;
  if (allowedTools !== null && typeof allowedTools !== "string") {
    return failure("allowed-tools", "allowed-tools must be text: tool names separated by spaces.");
  }

  return {
    ok: true,
    frontmatter: {
      name,
      description,
      license: license?.trim() || null,
      compatibility: compatibility?.trim() || null,
      metadata,
    },
    body: split.body,
  };
}
