/**
 * Exposes the core's public interface to the renderer as `window.incarnamind`.
 * Runs sandboxed with context isolation: the renderer gets these functions and
 * nothing else from Node or Electron.
 */
import { contextBridge, ipcRenderer } from "electron";
import {
  type CoreApi,
  type CoreBridge,
  type CoreEventListener,
  type CoreEventName,
  coreApiMethods,
} from "../core/api";
import { BRIDGE_KEY, channelFor, EVENT_CHANNEL } from "../shared/bridge";

const methods = Object.fromEntries(
  coreApiMethods.map((method) => [
    method,
    (...args: unknown[]) => ipcRenderer.invoke(channelFor(method), ...args),
  ]),
) as unknown as CoreApi;

// One IPC listener for all core events, fanned out to the renderer's listeners by event name.
const listeners = new Map<string, Set<(payload: unknown) => void>>();
ipcRenderer.on(EVENT_CHANNEL, (_event, name: string, payload: unknown) => {
  for (const listener of listeners.get(name) ?? []) {
    try {
      listener(payload);
    } catch (error) {
      console.error(error);
    }
  }
});

const bridge: CoreBridge = {
  ...methods,
  on<E extends CoreEventName>(event: E, listener: CoreEventListener<E>) {
    let set = listeners.get(event);
    if (!set) {
      set = new Set();
      listeners.set(event, set);
    }
    const entry = listener as (payload: unknown) => void;
    set.add(entry);
    return () => {
      set.delete(entry);
    };
  },
};

contextBridge.exposeInMainWorld(BRIDGE_KEY, bridge);
