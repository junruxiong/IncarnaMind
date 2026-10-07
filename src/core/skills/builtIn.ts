/**
 * Built-in Skills (#42): Skills that ship with the app. They are written
 * in-house, under the app's licence (Apache-2.0), with nothing borrowed: check
 * the licence of anything borrowed before adding it. They follow the standard
 * SKILL.md format, one folder per Skill named like the Skill, in
 * `resources/skills/` (`BUILT_IN_SKILLS_SOURCE`). electron-builder
 * copies that folder into the packaged app's resources (`extraResources` in
 * electron-builder.yml), and the host passes its path to the core as
 * `Paths.builtInSkills`.
 *
 * This module reads them. ./index installs them into the data folder at
 * startup, like imported Skills but marked built-in, and updates them when
 * the app's copy changes (a newer version of the app).
 */
import { createHash } from "node:crypto";
import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { parseSkillMd, type SkillFrontmatter } from "./frontmatter";
import { INSTRUCTIONS_FILE, inSkillOrder, isJunk, type SkillSourceFile } from "./source";

/** Where the built-in Skills are in the repository, from its root. */
export const BUILT_IN_SKILLS_SOURCE = "resources/skills";

/** Where electron-builder puts them in the packaged app, from its resources folder. */
export const BUILT_IN_SKILLS_PACKAGED = "skills";

/** A built-in Skill as the app ships it. */
export interface BuiltInSkill {
  name: string;
  frontmatter: SkillFrontmatter;
  /** SKILL.md first, then the others in path order. */
  files: SkillSourceFile[];
  /** SHA-256 of its files' paths and bytes, in hex: different files, a different digest. */
  digest: string;
}

function digestOf(files: readonly SkillSourceFile[]): string {
  const hash = createHash("sha256");
  for (const file of files) {
    hash.update(`${file.path}\0${file.data.length}\0`);
    hash.update(file.data);
  }
  return hash.digest("hex");
}

/** One built-in Skill's folder: its files, read now. Throws if it isn't a valid Skill. */
function readBuiltInSkill(root: string, folderName: string): BuiltInSkill {
  const files: SkillSourceFile[] = [];
  const walk = (segments: string[]) => {
    const entries = readdirSync(join(root, ...segments), { withFileTypes: true });
    for (const entry of entries.sort((a, b) => (a.name < b.name ? -1 : 1))) {
      const path = [...segments, entry.name];
      if (isJunk(path)) continue;
      if (entry.isDirectory()) walk(path);
      else if (entry.isFile()) {
        files.push({ path: path.join("/"), data: readFileSync(join(root, ...path)) });
      }
    }
  };
  walk([]);

  const ordered = inSkillOrder(files);
  const instructions = ordered[0];
  if (instructions?.path !== INSTRUCTIONS_FILE) {
    throw new Error(`The built-in Skill ${folderName} has no ${INSTRUCTIONS_FILE}.`);
  }
  const parsed = parseSkillMd(instructions.data.toString("utf8"));
  if (!parsed.ok) {
    throw new Error(`The built-in Skill ${folderName}'s SKILL.md is invalid: ${parsed.message}`);
  }
  if (parsed.frontmatter.name !== folderName) {
    throw new Error(
      `The built-in Skill in ${folderName} is named ${parsed.frontmatter.name}: a built-in Skill's folder has its name.`,
    );
  }
  return {
    name: parsed.frontmatter.name,
    frontmatter: parsed.frontmatter,
    files: ordered,
    digest: digestOf(ordered),
  };
}

/**
 * The built-in Skills in `folder`, in name order. A Skill that can't be read
 * is reported and left out, so the app still starts; none if the folder is
 * missing.
 */
export function readBuiltInSkills(
  folder: string,
  reportError: (error: unknown) => void,
): BuiltInSkill[] {
  let names: string[];
  try {
    names = readdirSync(folder, { withFileTypes: true })
      .filter((entry) => entry.isDirectory() && !isJunk([entry.name]))
      .map((entry) => entry.name)
      .sort();
  } catch (error) {
    reportError(error);
    return [];
  }
  const skills: BuiltInSkill[] = [];
  for (const name of names) {
    try {
      skills.push(readBuiltInSkill(join(folder, name), name));
    } catch (error) {
      reportError(error);
    }
  }
  return skills;
}
