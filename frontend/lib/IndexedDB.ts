import Dexie from "dexie";
import { CacheFiles } from "@/lib/interfaces/interface";

// Extend Dexie to define your database
export class FileDatabase extends Dexie {
  public files: Dexie.Table<CacheFiles, string>;

  constructor() {
    super("CacheFilesDB");
    this.version(1).stores({
      files: "id",
    });
    this.files = this.table("files");
  }

  // Add a new file, and ensure only 10 files are stored
  async addFile(file: CacheFiles) {
    const allFiles = await this.files.toArray();
    if (allFiles.length >= 10) {
      // Delete last items in allFiles
      await this.files.delete(allFiles[allFiles.length - 1].id);
    }
    await this.files.put(file); // This will update if the file exists, otherwise add a new one
  }

  // Query file by ID
  async getFile(id: string) {
    return await this.files.get(id);
  }

  // delete file by ID
  async deleteFile(id: string) {
    return await this.files.delete(id);
  }
}

// Initialize the database
export const CacheFilesDB = new FileDatabase();
