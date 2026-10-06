import type { CoreBridge } from "../../core/api";
import { BRIDGE_KEY } from "../../shared/bridge";

declare global {
  interface Window {
    /** The core's public interface (methods and events), exposed by the preload script. */
    readonly [BRIDGE_KEY]: CoreBridge;
  }
}

/** The only way the UI reaches the core. */
export const core: CoreBridge = window[BRIDGE_KEY];
