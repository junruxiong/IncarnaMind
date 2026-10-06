import { getFile, uploadFiles } from "@/lib/service/file";
import { useCallback, useState } from "react";
import { useDropzone } from "react-dropzone";
import { AiOutlineFileAdd } from "react-icons/ai";
import CryptoJS from "crypto-js";
import { useFileStore } from "@/lib/stores/fileStore";
import { CacheFilesDB } from "@/lib/IndexedDB";

function arrayBufferToWordArray(
  arrayBuffer: ArrayBuffer
): CryptoJS.lib.WordArray {
  const uint8Array = new Uint8Array(arrayBuffer);
  const words: number[] = []; // Explicitly define the array as number[]
  for (let i = 0; i < uint8Array.length; i++) {
    if (words[i >>> 2] === undefined) {
      words[i >>> 2] = 0;
    }
    words[i >>> 2] |= uint8Array[i] << (24 - (i % 4) * 8);
  }
  return CryptoJS.lib.WordArray.create(words, uint8Array.length);
}

export const FileUploader = () => {
  const [uploadProgress, setUploadProgress] = useState({
    uploaded: 0,
    total: 0,
  });
  const [isUploading, setIsUploading] = useState(false); // State to track upload status

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    setUploadProgress({ uploaded: 0, total: acceptedFiles.length });
    setIsUploading(true);

    // Function to split files into chunks
    const chunkFiles = (files: any, size: number) => {
      let chunks = [];
      for (let i = 0; i < files.length; i += size) {
        chunks.push(files.slice(i, i + size));
      }
      return chunks;
    };

    // Split files into chunks of 3
    const fileChunks = chunkFiles(acceptedFiles, 3);

    // Process each chunk
    for (const chunk of fileChunks) {
      const formData = new FormData();
      const readFile = (file: File) => {
        return new Promise<void>((resolve, reject) => {
          const reader = new FileReader();

          reader.onabort = () => reject("File reading was aborted");
          reader.onerror = () => reject("File reading has failed");
          reader.onload = () => {
            if (reader.result && typeof reader.result !== "string") {
              const wordArray = arrayBufferToWordArray(reader.result);
              const md5Hash = CryptoJS.MD5(wordArray).toString(); // MD5 hash is generated here from file content

              const originalFilename = file.name;
              const fileType = file.type;
              formData.append("files", file);
              formData.append("md5s", md5Hash); // Append the MD5 hash of the file content
              formData.append("original_filenames", originalFilename); // Append the original filename for server-side reference
              formData.append("types", fileType); // Append the file type for server-side reference
            }
            resolve();
          };
          reader.readAsArrayBuffer(file);
        });
      };

      // Process each file in the chunk
      const filePromises = chunk.map(readFile);
      await Promise.all(filePromises);

      // Upload the chunk
      try {
        const response = await uploadFiles(formData);
        if (response.hasOwnProperty("uploaded")) {
          useFileStore.getState().addMetadata(response.uploaded);

          const allFiles = formData.getAll("files");
          for (let i = 0; i < response.uploaded.length; i++) {
            const id = response.uploaded[i].id;
            const file = allFiles[i];
            if (file instanceof File) {
              const data = new Blob([file], { type: file.type });
              await CacheFilesDB.addFile({ id: id, data: data });
            }
          }
        }
      } catch (error) {
        console.error("Error uploading files:", error);
      }

      setUploadProgress((prev) => ({
        uploaded: prev.uploaded + chunk.length,
        total: prev.total,
      }));
      setIsUploading(false);
    }
  }, []);

  const { getRootProps, getInputProps } = useDropzone({ onDrop });

  // Calculate the percentage of files uploaded
  const uploadPercentage =
    uploadProgress.total > 0
      ? (uploadProgress.uploaded / uploadProgress.total) * 100
      : 0;

  return (
    <>
      {isUploading ? (
        // Show only the progress bar when uploading
        <div className="flex h-[120px] flex-col items-center justify-center px-3 py-4 text-sm text-gray-600">
          <div className="w-full mb-2 bg-gray-200 rounded-[9px] dark:bg-gray-700">
            <div
              className={`text-xs font-medium text-gray-300 text-center p-0.5 leading-none rounded-full ${
                uploadPercentage >= 6 ? "bg-black" : "bg-transparent"
              }`}
              style={{ width: `${uploadPercentage}%` }}
            >
              {Math.round(uploadPercentage)}%
            </div>
          </div>
          <p className="text-gray-600">Processing files</p>
        </div>
      ) : (
        // Show the rest of the components when not uploading
        <div
          className="flex h-[120px] flex-col items-center justify-center px-3 py-4 text-sm text-gray-600"
          {...getRootProps()}
        >
          <input {...getInputProps()} />
          <AiOutlineFileAdd className="cursor-pointer mb-2 text-[25px]" />
          <div className="cursor-pointer text-center mb-1">
            Drop or click to upload files
          </div>
          <div className="cursor-pointer text-center mb-1 text-xs font-semibold">
            Please name them correctly
          </div>
          <p className="cursor-pointer text-center text-xs">(support pdfs)</p>
        </div>
      )}
    </>
  );
};
