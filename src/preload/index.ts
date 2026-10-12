/**
 * Exposes the core's public interface to the renderer as `window.incarnamind`.
 * Runs sandboxed with context isolation: the renderer gets these functions and
 * nothing else from Node or Electron.
 */
import { contextBridge, ipcRenderer, webUtils } from "electron";
import {
  type CoreApi,
  type CoreBridge,
  type CoreEventListener,
  type CoreEventName,
  coreApiMethods,
} from "../core/api";
import {
  BRIDGE_KEY,
  channelFor,
  EVENT_CHANNEL,
  FILES_BRIDGE_KEY,
  FILES_CHANNELS,
  type FilesBridge,
  type MenuCommand,
} from "../shared/bridge";
import {
  CITED_PAGE_ARGUMENT,
  CITED_PAGE_BRIDGE_KEY,
  CITED_PAGE_CHANNELS,
  type CitedPageBridge,
} from "../shared/citedPage";
import { titleBarOf } from "../shared/titleBar";

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

// A sandboxed renderer can't see where a dropped or picked file lives; Electron's preload can.
const files: FilesBridge = {
  pathForFile: (file) => webUtils.getPathForFile(file),
  showSavedFile: (path) => ipcRenderer.invoke(FILES_CHANNELS.showSavedFile, path),
  pickOutputLocation: () => ipcRenderer.invoke(FILES_CHANNELS.pickOutputLocation),
  openDataFolder: () => ipcRenderer.invoke(FILES_CHANNELS.openDataFolder),
  openLogsFolder: () => ipcRenderer.invoke(FILES_CHANNELS.openLogsFolder),
  pickSkill: (kind) => ipcRenderer.invoke(FILES_CHANNELS.pickSkill, kind),
  openDocumentExternally: (documentId) =>
    ipcRenderer.invoke(FILES_CHANNELS.openDocumentExternally, documentId),
  showDocumentInFolder: (documentId) =>
    ipcRenderer.invoke(FILES_CHANNELS.showDocumentInFolder, documentId),
  pickLinkedFolder: () => ipcRenderer.invoke(FILES_CHANNELS.pickLinkedFolder),
  pickDocuments: () => ipcRenderer.invoke(FILES_CHANNELS.pickDocuments),
  pickReferenceFile: () => ipcRenderer.invoke(FILES_CHANNELS.pickReferenceFile),
  showLinkedFolder: (linkedFolderId) =>
    ipcRenderer.invoke(FILES_CHANNELS.showLinkedFolder, linkedFolderId),
  logError: (report) => ipcRenderer.send(FILES_CHANNELS.logError, report),
  onMenuCommand: (listener) => {
    const receive = (_event: unknown, command: MenuCommand) => listener(command);
    ipcRenderer.on(FILES_CHANNELS.menuCommand, receive);
    return () => {
      ipcRenderer.removeListener(FILES_CHANNELS.menuCommand, receive);
    };
  },
  setSidebarHidden: (hidden) => ipcRenderer.send(FILES_CHANNELS.sidebarHidden, hidden),
  setEditMenu: (state) => ipcRenderer.send(FILES_CHANNELS.editMenu, state),
  titleBar: titleBarOf(process.platform),
  onFullScreenChange: (listener) => {
    const receive = (_event: unknown, fullScreen: boolean) => listener(fullScreen);
    ipcRenderer.on(FILES_CHANNELS.fullScreen, receive);
    return () => {
      ipcRenderer.removeListener(FILES_CHANNELS.fullScreen, receive);
    };
  },
};

// The hidden window that draws a cited page (#133) shows the viewer's views, which need the core,
// and says to the main process where it drew the page; it needs none of the file helpers.
if (process.argv.includes(CITED_PAGE_ARGUMENT)) {
  const citedPage: CitedPageBridge = {
    request: () => ipcRenderer.invoke(CITED_PAGE_CHANNELS.request),
    resize: (height) => ipcRenderer.invoke(CITED_PAGE_CHANNELS.resize, height),
    drawn: (layout) => ipcRenderer.send(CITED_PAGE_CHANNELS.drawn, layout),
    failed: (reason) => ipcRenderer.send(CITED_PAGE_CHANNELS.failed, reason),
  };
  contextBridge.exposeInMainWorld(CITED_PAGE_BRIDGE_KEY, citedPage);
} else {
  contextBridge.exposeInMainWorld(FILES_BRIDGE_KEY, files);
}
