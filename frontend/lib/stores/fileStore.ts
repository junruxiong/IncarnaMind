import { create } from "zustand";
import { FileMetadata } from "@/lib/interfaces/interface";
import {
  persist,
  devtools,
  createJSONStorage,
  StateStorage,
} from "zustand/middleware";
import { get, set, del } from "idb-keyval"; // can use anything: IndexedDB, Ionic Storage, etc.

// Custom IndexedDB storage object
const storage: StateStorage = {
  getItem: async (name: string): Promise<any | null> => {
    console.log(name, "has been retrieved");
    return (await get(name)) || null;
  },
  setItem: async (name: string, value: any): Promise<void> => {
    console.log(name, "with value", value, "has been saved");
    await set(name, value);
  },
  removeItem: async (name: string): Promise<void> => {
    console.log(name, "has been deleted");
    await del(name);
  },
};

type FileActions = {
  //   activeFile: any;
  files: any[];
  activeMetadata: FileMetadata | null;
  fileMetadata: FileMetadata[];
  hasMore: boolean;
  addMetadata: (newMetadata: FileMetadata | FileMetadata[]) => void;
  renameMetadata: (fileId: string, newName: string) => void;
  deleteMetadatas: (fileIds: string | string[]) => void;
  setActiveMetadata: (fileId: string | null) => void;

  addFile2IdxB: (file: any[]) => void;
};

export const useFileStore = create<FileActions>()(
  devtools(
    persist(
      (set) => ({
        // activeFile: null,
        files: [],
        activeMetadata: null,
        fileMetadata: [],
        hasMore: true,

        addMetadata: (newMetadata) =>
          set((state) => ({
            fileMetadata: Array.isArray(newMetadata)
              ? [...state.fileMetadata, ...newMetadata]
              : [...state.fileMetadata, newMetadata],
          })),

        renameMetadata: (fileId, newName) =>
          set((state) => ({
            fileMetadata: state.fileMetadata.map((file) =>
              file.id === fileId
                ? { ...file, original_filename: newName }
                : file
            ),
          })),

        deleteMetadatas: (fileIds) =>
          set((state) => ({
            fileMetadata: state.fileMetadata.filter((file) =>
              !Array.isArray(fileIds)
                ? file.id !== fileIds
                : !fileIds.includes(file.id)
            ),
          })),

        setActiveMetadata: (fileId) =>
          set((state) => ({
            activeMetadata:
              state.fileMetadata.find((file) => file.id === fileId) || null,
          })),

        addFile2IdxB: (file) =>
          set((state) => {
            const newFiles = [...state.files, file];
            if (newFiles.length > 3) {
              newFiles.shift();
            }
            return {
              files: newFiles,
            };
          }),
      }),

      {
        name: "file-store",
        getStorage: () => sessionStorage,
        // storage: createJSONStorage(() => storage), //idxedDB storgae
        partialize: (state) => ({
          //   activeFile: state.activeFile,
          activeMetadata: state.activeMetadata,
        }),
        skipHydration: true,
      }
    )
  )
);
