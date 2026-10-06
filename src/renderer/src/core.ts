import type { CoreBridge } from "../../core/api";
import { BRIDGE_KEY, FILES_BRIDGE_KEY, type FilesBridge } from "../../shared/bridge";

declare global {
  interface Window {
    /** The core's public interface (methods and events), exposed by the preload script. */
    readonly [BRIDGE_KEY]: CoreBridge;
    /** File helpers that need Electron, exposed by the preload script. */
    readonly [FILES_BRIDGE_KEY]: FilesBridge;
  }
}

/** The only way the UI reaches the core. */
export const core: CoreBridge = window[BRIDGE_KEY];

/** Turns dropped or picked files into the paths `core.addDocuments` takes. */
export const files: FilesBridge = window[FILES_BRIDGE_KEY];
