import type * as Y from "yjs";
import { heldByMind } from "./heldByMind";

const pages = heldByMind<Y.Doc>();

/** The Minds shown in the window, by id, with the copy each page edits: for ⌘K's page commands. */
export const useShownPages = pages.use;

export const holdShownPage = pages.hold;
