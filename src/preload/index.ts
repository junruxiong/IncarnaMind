/**
 * Exposes the core's public interface to the renderer as `window.incarnamind`.
 * Runs sandboxed with context isolation: the renderer gets these functions and
 * nothing else from Node or Electron.
 */
import { contextBridge, ipcRenderer } from "electron";
import { type CoreApi, coreApiMethods } from "../core/api";
import { BRIDGE_KEY, channelFor } from "../shared/bridge";

const bridge = Object.fromEntries(
  coreApiMethods.map((method) => [
    method,
    (...args: unknown[]) => ipcRenderer.invoke(channelFor(method), ...args),
  ]),
) as unknown as CoreApi;

contextBridge.exposeInMainWorld(BRIDGE_KEY, bridge);
