import type { ObjectKind } from "../../src/renderer/src/objects";

/**
 * The kinds of object no row shows a menu for yet, and why. A kind leaves this
 * list when its menu is registered. The conformance checks read it:
 * tests/renderer/conformance.test.ts and e2e/conformance.spec.ts.
 */
export const NO_MENU_YET = {
  conversation: "the assistant's History rows, where it would show, have no menus yet (#258)",
  block: "a Block's menu is the note handle's own, not yet on the shared menu (#608)",
} as const satisfies Partial<Record<ObjectKind, string>>;
