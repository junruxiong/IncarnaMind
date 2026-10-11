import type { CommandId } from "../../../shared/commands";
import type { MessageKey } from "../../../shared/i18n";
import type { ObjectKind } from "../objects";
import { COMMAND_GROUPS, type Command, type CommandContext, commands, offered } from "./registry";

const registered = new Map<ObjectKind, CommandId[]>();

/**
 * Adds commands to a kind's menu. They show in the order of their groups
 * (`COMMAND_GROUPS`), and in the order given within a group, so a feature
 * registers its own items and needs no one else's edit. Calling it again for
 * a kind adds to its menu.
 */
export function registerMenu(kind: ObjectKind, ids: readonly CommandId[]): void {
  const held = registered.get(kind) ?? [];
  registered.set(kind, [...held, ...ids.filter((id) => !held.includes(id))]);
}

/** The kinds that have a menu. */
export const menuKinds = (): ObjectKind[] => [...registered.keys()];

/** The commands registered for a kind, in the order given. */
export const menuIds = (kind: ObjectKind): readonly CommandId[] => registered.get(kind) ?? [];

/** An item of a menu: a command that applies here, and whether it can run now (else, why not). */
export interface MenuItem {
  command: Command;
  enabled: true | { reason: MessageKey };
}

/** What a menu draws: items, with `"rule"` between groups. */
export type MenuEntry = MenuItem | "rule";

/**
 * A kind's menu for this context. A command that doesn't apply is left out; one
 * that applies but can't run now stays, disabled with its reason. Groups
 * follow `COMMAND_GROUPS`, with a rule between those that have items.
 */
export function buildMenu(kind: ObjectKind, ctx: CommandContext): MenuEntry[] {
  const items = menuIds(kind).flatMap((id): { group: number; item: MenuItem }[] => {
    const command = commands.get(id);
    if (!command || !offered(command, ctx)) return [];
    return [
      {
        group: COMMAND_GROUPS.indexOf(command.group),
        item: { command, enabled: command.enabled?.(ctx) ?? true },
      },
    ];
  });
  const entries: MenuEntry[] = [];
  COMMAND_GROUPS.forEach((_, group) => {
    const inGroup = items.filter((each) => each.group === group);
    if (inGroup.length === 0) return;
    if (entries.length > 0) entries.push("rule");
    for (const each of inGroup) entries.push(each.item);
  });
  return entries;
}
