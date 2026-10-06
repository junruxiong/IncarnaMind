"use client";
import { useSegmentStore } from "@/lib/stores/segmentStore";
import { FcStumbleupon } from "react-icons/fc";
import { AiOutlineSearch } from "react-icons/ai";
import { UtilityBar } from "./LeftbarUtilities";

import { LeftSidebarFileList } from "./LeftbarFileList";

export const LeftSideBar = () => {
  const {
    lefBarWidth,
    viewerWidth,
    rodWidth,
    minviewerWidth,
    minChatWidth,
    minLefBarWidth,
  } = useSegmentStore();

  const max_width =
    window.innerWidth -
    Math.max(viewerWidth, minviewerWidth) -
    2 * rodWidth -
    minChatWidth;

  const handleMouseDown = (e: React.MouseEvent<HTMLDivElement>) => {
    e.preventDefault();
    document.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("mouseup", handleMouseUp);
  };

  const handleMouseMove = (e: MouseEvent) => {
    const newWidth = Math.min(Math.max(minLefBarWidth, e.clientX), max_width);
    useSegmentStore.setState({ lefBarWidth: newWidth });
  };

  const handleMouseUp = () => {
    document.removeEventListener("mousemove", handleMouseMove);
    document.removeEventListener("mouseup", handleMouseUp);
  };

  return (
    <div className="flex relative">
      <div
        className="bg-gray-50 flex flex-col"
        style={{ width: `${lefBarWidth}px` }}
      >
        <div className="group relative inline-flex items-center ml-2 mr-3 mt-2 mb-2">
          <FcStumbleupon className="absolute cursor-pointer text-[35px]"></FcStumbleupon>
          <input
            type="text"
            placeholder="IncarnaMind Search"
            className="text-sm font-normal ml-[39px] p-2 pl-3 pr-[36px] w-full rounded-[15px] h-8 text-gray-700 border-[1px] border-gray-300 outline-none hover:shadow-custom_unfocus focus:shadow-custom_unfocus"
          />
          <AiOutlineSearch className="absolute cursor-pointe right-[7px] text-[26px] text-gray-400 hover:text-gray-700"></AiOutlineSearch>
        </div>

        <LeftSidebarFileList />

        <UtilityBar lefBarWidth={lefBarWidth} />
      </div>
      <div
        className="cursor-col-resize bg-gray-200 flex items-center justify-center three-dots"
        onMouseDown={handleMouseDown}
        style={{ width: `${rodWidth}px` }}
      />
    </div>
  );
};

{
  /* <div className="flex grow flex-col mx-3">
            <div className="w-ful px-1 py-[6px] rounded-[9px] text-sm hover:bg-gray-200">
              <div className="group relative inline-flex items-center w-full">
                <FcDocument className="absolute left-0 text-base text-gray-500"></FcDocument>
                <span
                  onClick={handleOpen}
                  className="pl-5 pr-4 whitespace-nowrap text-ellipsis overflow-hidden text-gray-700 cursor-pointer"
                >
                  A Neural Corpus Indexer
                </span>
                <AiOutlineMore className="absolute right-0 text-base text-gray-500 hover:text-black"></AiOutlineMore>
              </div>
            </div>
        
          </div> */
}

{
  /* <div className="flex-grow w-full overflow-y-auto hide-scrollbar">
          <div className="flex grow flex-col mx-[15px] py-1">
            <div className="w-ful mx-1 mb-[20px] text-sm">
              <div className="group relative inline-flex items-center w-full">
                <FcDocument className="absolute left-0 text-base text-gray-500"></FcDocument>
                <p
                  onClick={handleOpen}
                  className="pl-5 pr-3 whitespace-nowrap text-ellipsis overflow-hidden text-gray-700 cursor-pointer hover:underline w-full text-custom"
                >
                  I was a marine biologist Few-Shot Learners
                </p>
              </div>
              <p className="pl-5 pr-2 line-clamp-2 text-gray-500 custom-sm font-lora">
                So I started to walk into the water. I won't lie to you boys, I
                was terrified. But I pressed on, and as I made my way past the
                breakers a strange calm came over me. I don't know if it was
                divine intervention or the kinship of all living things but I
                tell you Jerry at that moment, I was a marine biologist.
              </p>
            </div>

            <div className="w-ful mx-1 mb-[20px] text-sm">
              <div className="group relative inline-flex items-center w-full">
                <FcFaq className="absolute left-0 text-base text-gray-500"></FcFaq>
                <p
                  onClick={handleOpen}
                  className="pl-5 pr-2 whitespace-nowrap text-ellipsis overflow-hidden text-gray-700 cursor-pointer hover:underline w-full text-custom"
                >
                  Models are Few-Shot Learners
                </p>
              </div>
              <p className="pl-5 pr-2 line-clamp-2 text-gray-500 custom-sm font-lora">
                But I pressed on, s I made my way past the breakers a strange
                calm't know if it was divine intervention or the kinship of all
                living things but I tell you Jerry at that moment, I was a
                marine biologist.
              </p>
            </div>
          </div>
        </div> */
}
