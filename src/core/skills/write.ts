/**
 * Writing a Skill's SKILL.md from what the User types or a model drafts: a
 * name, when to use it and its steps. The steps are the instructions, kept as
 * they were written; the frontmatter holds the rest. Also the zip a Skill is
 * exported as, which the import reads back (./source) with the same preview.
 */
import { deflateRawSync } from "node:zlib";
import type { SkillDraft } from "../api";
import { writeZip } from "../documents/formats/zip";
import type { SkillFrontmatter } from "./frontmatter";

/** The longest "when to use", as the format allows it. */
export const MAX_DESCRIPTION_LENGTH = 1024;

/** "When to use" on one line: a line break in a frontmatter value would be folded anyway. */
export const oneLine = (text: string): string => text.replace(/\s+/g, " ").trim();

/** Fields kept from a Skill's frontmatter when its SKILL.md is written again. */
type Kept = Pick<SkillFrontmatter, "license" | "compatibility" | "metadata">;

/** A SKILL.md: frontmatter (values as quoted YAML strings), a blank line, then the steps. */
export function composeSkillMd(draft: SkillDraft, kept?: Kept): string {
  const lines = [
    "---",
    `name: ${draft.name}`,
    `description: ${JSON.stringify(draft.description)}`,
    ...(kept?.license ? [`license: ${JSON.stringify(kept.license)}`] : []),
    ...(kept?.compatibility ? [`compatibility: ${JSON.stringify(kept.compatibility)}`] : []),
  ];
  const metadata = Object.entries(kept?.metadata ?? {});
  if (metadata.length > 0) {
    lines.push("metadata:");
    for (const [key, value] of metadata) {
      lines.push(`  ${JSON.stringify(key)}: ${JSON.stringify(value)}`);
    }
  }
  lines.push("---", "", draft.steps.trim(), "");
  return lines.join("\n");
}

/** A Skill as a zip, in a folder of its name, so unzipping it gives a Skill folder. */
export function skillZip(
  name: string,
  files: readonly { path: string; data: Uint8Array }[],
): Uint8Array {
  return writeZip(
    files.map((file) => ({ name: `${name}/${file.path}`, data: file.data })),
    deflateRawSync,
  );
}
