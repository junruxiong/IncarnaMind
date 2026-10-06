import { useCallback, useRef, useState } from "react";
import { useFiles } from "@/lib/hooks/useFiles";

import { FileMetadata } from "@/lib/interfaces/interface";
import { useFileStore } from "@/lib/stores/fileStore";
import { useSegmentStore } from "@/lib/stores/segmentStore";

import { FcDocument } from "react-icons/fc";
import { AiOutlineMore, AiOutlineEdit, AiOutlineDelete } from "react-icons/ai";

import { Modal, Popover } from "@mui/material";
import { changeFileName, deleteFile } from "@/lib/service/file";
import { CacheFilesDB } from "@/lib/IndexedDB";

export const LeftSidebarFileList = () => {
  const { setPageCallback: setPage } = useFiles();
  const {
    fileMetadata,
    hasMore,
    activeMetadata,
    setActiveMetadata,
    renameMetadata,
    deleteMetadatas,
  } = useFileStore();

  const [currentFile, setCurrentFile] = useState<FileMetadata | null>(null);
  // State for Popover
  const [anchorEl, setAnchorEl] = useState<HTMLButtonElement | null>(null);
  const [isEditing, setIsEditing] = useState(false);
  const [editedFile, seteditedFile] = useState("");

  const openPopover = Boolean(anchorEl);

  const handleRenameClick = (fileMetadata: FileMetadata) => {
    seteditedFile(fileMetadata.original_filename);
    setIsEditing(true);
    setCurrentFile(fileMetadata);

    handlePopoverClose();
  };

  const handleDoubleClick = (fileMetadata: FileMetadata) => {
    seteditedFile(fileMetadata.original_filename);
    setIsEditing(true);
    setCurrentFile(fileMetadata);
  };

  const handleSaveFileName = async () => {
    const isUniqueFilename = !fileMetadata.some(
      (file) => file.original_filename === editedFile
    );
    if (
      currentFile &&
      editedFile !== currentFile.original_filename &&
      editedFile !== "" &&
      isUniqueFilename
    ) {
      const newFilename = new FormData();
      newFilename.append("original_filename", editedFile);
      try {
        const response = await changeFileName(currentFile.id, newFilename);
        renameMetadata(currentFile.id, editedFile);
        setIsEditing(false);
      } catch (error) {
        console.error("Error updating file name:", error);
      }
    } else {
      setIsEditing(false); // No change made, just exit editing mode
      // show a modal
    }
  };

  const handleDeleteFile = async (fileId: string) => {
    try {
      const response = await deleteFile(fileId);
      deleteMetadatas(fileId);
      await CacheFilesDB.deleteFile(fileId);
    } catch (error) {
      console.error("Error deleting file:", error);
    }
  };

  // State for Modal
  const [openModal, setOpenModal] = useState(false);

  // Handle Popover
  const handlePopoverOpen = (
    event: React.MouseEvent<HTMLButtonElement>,
    fileMetadata: FileMetadata
  ) => {
    event.stopPropagation();
    setCurrentFile(fileMetadata);
    setAnchorEl(event.currentTarget);
  };

  const handlePopoverClose = () => {
    setAnchorEl(null);
  };

  // Handle Modal
  const handleModalOpen = () => {
    setOpenModal(true);
    handlePopoverClose(); // Close popover when opening modal
  };

  const handleModalClose = () => {
    setOpenModal(false);
  };

  const {
    lefBarWidth,
    rodWidth,
    isViewerVisible,
    prevviewerWidth,
    minChatWidth,
  } = useSegmentStore();

  const scrollContainer = useRef<HTMLDivElement>(null);
  const observerRef = useRef<IntersectionObserver>();

  const handleObserver = (entries: IntersectionObserverEntry[]) => {
    if (entries[0].isIntersecting && hasMore) {
      setPage((prevPage: any) => prevPage + 1);
    }
  };

  const lastFileRef = useCallback(
    (node: HTMLDivElement | null) => {
      if (observerRef.current) observerRef.current.disconnect();
      observerRef.current = new IntersectionObserver(handleObserver);
      if (node) observerRef.current.observe(node);
    },
    [hasMore] // Dependencies for the callback
  );

  const handleOpen = (id: string) => {
    if (isViewerVisible == false) {
      useSegmentStore.setState({ isViewerVisible: true });
      useSegmentStore.setState({
        viewerWidth: Math.min(
          prevviewerWidth,
          window.innerWidth - lefBarWidth - 2 * rodWidth - minChatWidth
        ),
      });
    }
    if (isEditing == false) {
      setActiveMetadata(id);
    }
  };

  return (
    <div
      ref={scrollContainer}
      className="flex-grow w-full overflow-y-auto hide-scrollbar"
    >
      {fileMetadata.map((metadata: FileMetadata, index) => (
        <div
          key={metadata.id}
          ref={index === fileMetadata.length - 1 ? lastFileRef : null}
          className="flex-grow w-full overflow-y-auto hide-scrollbar"
        >
          <div className="flex-grow flex-col mx-3">
            <div
              title={metadata.original_filename}
              onClick={() => handleOpen(metadata.id)}
              className={`w-full px-1 my-[1px] py-[5px] rounded-[9px] text-sm ${
                activeMetadata && activeMetadata.id === metadata.id
                  ? "bg-gray-200"
                  : "hover:bg-gray-100"
              }`}
            >
              <div onDoubleClick={() => handleDoubleClick(metadata)}>
                {isEditing && currentFile?.id === metadata.id ? (
                  <div className="group relative inline-flex items-center w-full">
                    <FcDocument className="absolute left-0 text-base text-gray-500 cursor-pointer"></FcDocument>
                    <input
                      type="text"
                      value={editedFile}
                      onChange={(e) => seteditedFile(e.target.value)}
                      onBlur={handleSaveFileName}
                      onKeyDown={(e) => {
                        if (e.key === "Enter") {
                          handleSaveFileName();
                        }
                      }}
                      className="ml-5 w-full whitespace-nowrap overflow-hidden outline-gray-400 outline-1 text-gray-700"
                      autoFocus
                    />
                  </div>
                ) : (
                  <div className="group relative inline-flex items-center w-full">
                    <FcDocument className="absolute left-0 text-base text-gray-500 cursor-pointer"></FcDocument>
                    <div className="pl-5 pr-4 whitespace-nowrap text-ellipsis overflow-hidden text-gray-700 cursor-pointer">
                      {metadata.original_filename}
                    </div>
                    <AiOutlineMore
                      title="More"
                      onClick={(event: React.MouseEvent<HTMLButtonElement>) =>
                        handlePopoverOpen(event, metadata)
                      }
                      className="absolute right-0 text-base text-gray-500 hover:text-black cursor-pointer"
                    ></AiOutlineMore>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      ))}

      {anchorEl && (
        <Popover
          open={openPopover}
          anchorEl={anchorEl}
          onClose={handlePopoverClose}
          anchorOrigin={{ vertical: "bottom", horizontal: "left" }}
          sx={{
            transform: "translate(6px,-3px)",
            "& .MuiPaper-root": {
              // height: `88px`,
              boxShadow: "0 0 10px rgba(0, 0, 0, 0.15)", // Custom shadow
              borderRadius: "9px", // Rounded corners
              backgroundColor: "rgba(255, 255, 255, 0.9)", // Semi-transparent white background
              backdropFilter: "blur(9px)",
            },
          }}
        >
          <div className="flex flex-col p-1 text-sm">
            <button
              onClick={() => currentFile && handleRenameClick(currentFile)}
              className="group relative inline-flex items-center w-full px-2 py-1 my-[2px] rounded-[5px] hover:bg-gray-100"
            >
              <AiOutlineEdit className="absolute left-1 text-base text-gray-600 cursor-pointer"></AiOutlineEdit>
              <p className="pl-5 whitespace-nowrap text-ellipsis overflow-hidden text-gray-700 cursor-pointer">
                Rename
              </p>
            </button>

            <button
              onClick={handleModalOpen}
              className="group relative inline-flex items-center w-full px-2 py-1 my-[2px] rounded-[5px] hover:bg-gray-100"
            >
              <AiOutlineDelete className="absolute left-1 text-base text-red-600	 cursor-pointer"></AiOutlineDelete>
              <p className="pl-5 whitespace-nowrap text-ellipsis overflow-hidden text-red-600 cursor-pointer">
                Delete
              </p>
            </button>
          </div>
        </Popover>
      )}

      <Modal open={openModal} onClose={handleModalClose}>
        <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 max-w-4xl bg-white shadow-custom_focus p-4 rounded-[9px]">
          <div>
            <h2 className="text-lg font-semibold">Delete File</h2>
            {currentFile && (
              <div className="mt-2">
                This action will delete{" "}
                <span className="font-medium">
                  {currentFile.original_filename}
                </span>
                .
              </div>
            )}
          </div>

          <div className="flex justify-end mt-4">
            <button
              onClick={handleModalClose}
              className="border border-solid border-gray-300 rounded-[9px] px-4 py-2 mr-2 hover:bg-gray-100"
            >
              Cancel
            </button>
            <button
              onClick={() => {
                if (currentFile) {
                  handleDeleteFile(currentFile.id);
                }
                handleModalClose();
              }}
              className=" text-white bg-red-700 rounded-[9px] px-4 py-2 hover:bg-red-800"
            >
              Delete
            </button>
          </div>
        </div>
      </Modal>
    </div>
  );
};
