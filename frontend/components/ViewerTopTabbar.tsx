import { useSegmentStore } from "@/lib/stores/segmentStore";
import { AiOutlineClose } from "react-icons/ai";
import { FcDocument } from "react-icons/fc";
import { useFileStore } from "@/lib/stores/fileStore";

export const VierwerTopTabbar = () => {
  const viewerWidth = useSegmentStore((state) => state.viewerWidth);

  const handleClose = () => {
    useFileStore.getState().setActiveMetadata(null);
    useSegmentStore.setState({ isViewerVisible: false });
    useSegmentStore.setState({ prevviewerWidth: viewerWidth });
    useSegmentStore.setState({ viewerWidth: 0 });
  };

  const activeMetadata = useFileStore((state) => state.activeMetadata);

  // map sessions to tabbar
  return (
    <div className="flex flex-row h-10">
      <div className="group relative inline-flex items-center mt-2">
        <FcDocument className="absolute left-[10px] text-base text-gray-500 font-normal"></FcDocument>

        <button
          title={activeMetadata?.original_filename ?? ""}
          className="whitespace-nowrap bg-white text-gray-700 text-sm h-8 pl-8 pr-7 rounded-t-[9px]"
        >
          <p className=" max-w-[140px] overflow-hidden text-ellipsis whitespace-nowrap">
            {activeMetadata?.original_filename ?? ""}
          </p>
        </button>

        <button
          onClick={handleClose}
          className="absolute right-[8px] text-sm text-gray-500 p-[2px] hover:text-gray-600 hover:bg-gray-200 rounded-[9px]"
        >
          <AiOutlineClose />
        </button>
      </div>
    </div>
  );
};
