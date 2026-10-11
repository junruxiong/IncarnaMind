import type { ComponentType } from "react";
import type { CommandId, ShortcutKeys } from "../../../shared/commands";
import type { MessageKey } from "../../../shared/i18n";
import type { ObjectKind, ObjectRef, Translate, Undoable } from "../objects";
import { checkKeys } from "./keys";

/** Where the User is acting, as `data-surface` marks it on the page. */
export const SURFACES = ["sidebar", "library", "mind", "viewer", "palette"] as const;
export type Surface = (typeof SURFACES)[number];

/** The menus' order of commands (docs/agents/interaction.md, rule 5). */
export const COMMAND_GROUPS = ["open", "main", "move", "tags", "edit", "file", "danger"] as const;
export type CommandGroup = (typeof COMMAND_GROUPS)[number];

export interface Shortcut {
  keys: ShortcutKeys;
  /**
   * A list's own keys (⌘A, ⌘⌫, Enter, F2, Space): they act only while the
   * focus is in a list, and never while the User types. Other shortcuts act
   * anywhere in the window.
   */
  list?: true;
}

export interface CommandContext {
  selection: readonly ObjectRef[];
  focus: ObjectRef | null;
  surface: Surface;
}

export interface Command {
  id: CommandId;
  label: MessageKey | ((ctx: CommandContext, t: Translate) => string);
  keywords?: readonly MessageKey[];
  icon?: ComponentType<{ className?: string }>;
  shortcut?: Shortcut;
  /** False keeps it out of ⌘K; it then needs a shortcut. */
  palette?: false;
  group: CommandGroup;
  /** The kinds it acts on, as the selection or the focus. Without them, it acts on the window. */
  appliesTo?: readonly ObjectKind[];
  /** Whether it is offered at all. */
  when?(ctx: CommandContext): boolean;
  /** Offered, but not now, and why. */
  enabled?(ctx: CommandContext): true | { reason: MessageKey };
  /** Undone with ⌘Z, or asked about first, naming what will be lost (rule 4). */
  reversible: true | { confirm: MessageKey };
  run(ctx: CommandContext): void | Undoable | Promise<Undoable | void>;
}

const NO_CONTEXT: CommandContext = { selection: [], focus: null, surface: "mind" };

const targetsOf = (ctx: CommandContext): readonly ObjectRef[] =>
  ctx.selection.length > 0 ? ctx.selection : ctx.focus ? [ctx.focus] : [];

/** Whether a command is offered here: it fits what is selected or focused, and `when` agrees. */
export function offered(command: Command, ctx: CommandContext): boolean {
  const kinds = command.appliesTo;
  if (kinds) {
    const targets = targetsOf(ctx);
    if (targets.length === 0 || !targets.every((target) => kinds.includes(target.kind))) {
      return false;
    }
  }
  return command.when?.(ctx) ?? true;
}

export const isEnabled = (command: Command, ctx: CommandContext): boolean =>
  (command.enabled?.(ctx) ?? true) === true;

export const labelOf = (command: Command, ctx: CommandContext, t: Translate): string =>
  typeof command.label === "string" ? t(command.label) : command.label(ctx, t);

const registered = new Map<CommandId, Command>();

export const commands = {
  register(command: Command): () => void {
    if (registered.has(command.id)) throw new Error(`Two commands are called ${command.id}.`);
    if (command.shortcut) checkKeys(command.shortcut.keys);
    registered.set(command.id, command);
    return () => {
      if (registered.get(command.id) === command) registered.delete(command.id);
    };
  },

  get: (id: CommandId): Command | undefined => registered.get(id),

  all: (): Command[] => [...registered.values()],

  /** The commands offered here, enabled or not. */
  list: (ctx: CommandContext): Command[] =>
    [...registered.values()].filter((command) => offered(command, ctx)),

  /**
   * Runs a command, if it is offered and enabled here. Resolves to its undo
   * when it changed something.
   */
  async run(id: CommandId, given: Partial<CommandContext> = {}): Promise<Undoable | undefined> {
    const command = registered.get(id);
    if (!command) throw new Error(`There is no command ${id}.`);
    const ctx = { ...NO_CONTEXT, ...given };
    if (!offered(command, ctx) || !isEnabled(command, ctx)) return undefined;
    return (await command.run(ctx)) ?? undefined;
  },
};
