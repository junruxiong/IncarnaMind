/**
 * The Skills as a Tool provider (see ../tools): `use_skill` loads a Skill's
 * instructions, `read_skill_file` one of its files, and `run_skill_script`
 * runs one of its scripts. What each call does is given by the Answer that
 * offers them, which opened the Skills it may use (see `SkillSession`) and
 * asks the User before a script runs.
 */
import { SKILL_TOOLS, type Tool, type ToolCallContext, type ToolProviderInfo } from "../tools";

/** Who provides the Skill Tools, for their Tool-call cards. */
const SKILLS_PROVIDER: ToolProviderInfo = { kind: "skills", id: "skills", name: "Skills" };

/** What the Skill Tools do, for one Answer. */
export interface SkillToolsOptions {
  /** Offer `use_skill`: there are Skills the model may load, listed in the instructions. */
  loadable: boolean;
  /** A Skill's full instructions and its list of files, for the model. Throws for an unknown Skill. */
  useSkill(name: string): Promise<string>;
  /** One of a Skill's files, as text. Throws for a path outside the Skill. */
  readSkillFile(skill: string, path: string): Promise<string>;
  /**
   * Runs one of a Skill's scripts (`run_skill_script`, with `{ skill, script,
   * args }`), asking the User first unless the Skill's scripts always run.
   * Resolves with what to tell the model (how it ended and what it wrote, or
   * that the User denied it); rejects with why it couldn't run. Absent when no
   * script can run: none of the Skills has one, or the User turned scripts off.
   */
  runScript?(input: Record<string, unknown>, context: ToolCallContext): Promise<string>;
}

/** An argument as text: what the Tool-call card shows, and what the call gets. */
const text = (value: unknown) => String(value ?? "");

/**
 * A `run_skill_script` call's arguments as the Tool-call card shows them:
 * the Skill, the script and the arguments, as text.
 */
export function scriptCallInput(input: unknown): { skill: string; script: string; args: string[] } {
  const fields = isPlainObject(input) ? input : {};
  return {
    skill: text(fields.skill),
    script: text(fields.script),
    args: Array.isArray(fields.args) ? fields.args.map((each) => text(each)) : [],
  };
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/** The Skill Tools one Answer offers: `use_skill` when Skills are listed, and `run_skill_script` when scripts can run. */
export function skillTools(options: SkillToolsOptions): Tool[] {
  const own = (name: string) => ({ name, provider: SKILLS_PROVIDER, providerTool: name });
  const tools: Tool[] = [];
  if (options.loadable) {
    tools.push({
      ...own(SKILL_TOOLS.use),
      description:
        "Load a Skill's full instructions by its name, before answering a Question the Skill is for. Returns the instructions and the Skill's files.",
      inputSchema: {
        type: "object",
        properties: { name: { type: "string", description: "The Skill's name, as listed." } },
        required: ["name"],
      },
      shownInput: ({ name }) => ({ name: text(name) }),
      call: async ({ name }) => options.useSkill(text(name)),
    });
  }
  tools.push({
    ...own(SKILL_TOOLS.readFile),
    description:
      "Read one of a Skill's files, such as a reference its instructions point to. Only files inside the Skill can be read.",
    inputSchema: {
      type: "object",
      properties: {
        skill: { type: "string", description: "The Skill's name." },
        path: {
          type: "string",
          description: "The file's path inside the Skill, e.g. references/guide.md.",
        },
      },
      required: ["skill", "path"],
    },
    shownInput: ({ skill, path }) => ({ skill: text(skill), path: text(path) }),
    call: async ({ skill, path }) => options.readSkillFile(text(skill), text(path)),
  });
  const { runScript } = options;
  if (runScript) {
    tools.push({
      ...own(SKILL_TOOLS.runScript),
      description:
        "Run one of a Skill's scripts when its instructions say to, with arguments. It runs on the User's computer in a new, empty working folder; the Skill's own folder is in the SKILL_DIR environment variable. Python (.py), JavaScript (.js, .mjs) and shell (.sh) scripts can run. The User approves each run first and may deny it. Returns the exit code and what the script printed.",
      inputSchema: {
        type: "object",
        properties: {
          skill: { type: "string", description: "The Skill's name." },
          script: {
            type: "string",
            description: "The script's path inside the Skill, e.g. scripts/convert.py.",
          },
          args: {
            type: "array",
            items: { type: "string" },
            description: "The script's command-line arguments, in order.",
          },
        },
        required: ["skill", "script"],
      },
      shownInput: scriptCallInput,
      call: async (input, context) => runScript(input, context),
    });
  }
  return tools;
}
