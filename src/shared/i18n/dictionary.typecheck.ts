/**
 * Compile-time checks, run by `npm run typecheck` (never bundled): a dictionary
 * with a missing or an unknown key must not compile. If a `@ts-expect-error`
 * below stops being needed, tsc reports it as unused and the check fails.
 */
import { en } from "./en";
import type { Dictionary } from "./types";

const { "sidebar.newMind": _dropped, ...missingOneKey } = en;

// @ts-expect-error: "sidebar.newMind" is missing.
missingOneKey satisfies Dictionary;

// @ts-expect-error: "sidebar.unknown" is not a key.
({ ...en, "sidebar.unknown": "Unknown" }) satisfies Dictionary;
