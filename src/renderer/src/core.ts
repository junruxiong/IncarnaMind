import type { CoreApi } from "../../core/api";
import { BRIDGE_KEY } from "../../shared/bridge";

declare global {
  interface Window {
    /** The core's public interface, exposed by the preload script. */
    readonly [BRIDGE_KEY]: CoreApi;
  }
}

/** The only way the UI reaches the core. */
export const core: CoreApi = window[BRIDGE_KEY];
