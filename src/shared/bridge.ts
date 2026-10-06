/** Names shared by the main process and the preload script for the typed core bridge. */
import type { CoreApiMethod } from "../core/api";

/** The renderer reaches the core as `window.incarnamind`. */
export const BRIDGE_KEY = "incarnamind";

export const channelFor = (method: CoreApiMethod): string => `core:${method}`;
