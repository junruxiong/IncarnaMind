import { useCallback, useEffect, useState } from "react";
import { getFileMetadata, getFile, uploadFiles } from "../service/file";
import { useFileStore } from "../stores/fileStore";
import { CacheFilesDB } from "../IndexedDB";

export const useFiles = () => {
  const [page, setPage] = useState(1);
  const fileMetadata = useFileStore((state) => state.fileMetadata);
  //   console.log("sessions+++++", sessions);

  const setPageCallback = useCallback(
    (newPage: any) => {
      setPage(newPage);
    },
    [page]
  );

  useEffect(() => {
    async function fetchMetadata() {
      try {
        const { results, next } = await getFileMetadata(page);
        fileMetadata.length === 0 &&
          useFileStore.setState({ fileMetadata: results });
        page > 1 && useFileStore.getState().addMetadata(results);
        // if next is null, hasMore is false
        next ?? useFileStore.setState({ hasMore: false });
      } catch (error) {
        console.log(error);
      }
    }
    fetchMetadata();
  }, [page]);
  return { setPageCallback };
};

export const renderFile = () => {
  const activeMetadata = useFileStore((state) => state.activeMetadata);
  const [pdfFile, setPdfFile] = useState<string | null>(null);

  useEffect(() => {
    async function fetchFile(fileId: string) {
      if (!fileId) return;
      //   first check if the file is in the cache (getFileById), if not, fetch it
      const cachedFile = await CacheFilesDB.getFile(fileId);
      if (cachedFile) {
        setPdfFile(URL.createObjectURL(cachedFile.data));
        return;
      }
      try {
        const data = await getFile(fileId);
        await CacheFilesDB.addFile({ id: fileId, data: data });
        setPdfFile(URL.createObjectURL(data));
      } catch (error) {
        console.error("Error fetching PDF:", error);
      }
    }
    if (activeMetadata) {
      fetchFile(activeMetadata.id);
    }
  }, [activeMetadata]);

  return pdfFile;
};
