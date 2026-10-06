"use client";
import {
  Worker,
  Viewer,
  Tooltip,
  Position,
  Popover,
  MinimalButton,
  Button,
  SpecialZoomLevel,
  Icon,
} from "@react-pdf-viewer/core";
import { toolbarPlugin, ToolbarSlot } from "@react-pdf-viewer/toolbar";
import {
  highlightPlugin,
  Trigger,
  HighlightArea,
  RenderHighlightsProps,
  SelectionData,
  MessageIcon,
  RenderHighlightTargetProps,
} from "@react-pdf-viewer/highlight";
import { bookmarkPlugin } from "@react-pdf-viewer/bookmark";

import { AiOutlineEdit } from "react-icons/ai";

// Import the styles provided by the react-pdf-viewer packages
import "@react-pdf-viewer/core/lib/styles/index.css";
import "@react-pdf-viewer/bookmark/lib/styles/index.css";
import "@react-pdf-viewer/default-layout/lib/styles/index.css";
import "@react-pdf-viewer/highlight/lib/styles/index.css";
import { useEffect, useState } from "react";
import { useFileStore } from "@/lib/stores/fileStore";
import { getFile } from "@/lib/service/file";
import { renderFile } from "@/lib/hooks/useFiles";

// const fileUrl = "/Gradient Descent The Ultimate Optimizer.pdf";

export const ViewerWindow = ({ viewerWidth }: { viewerWidth: number }) => {
  const [sidebarOpened, setSidebarOpened] = useState(false);
  const [isToolbarVisible, setIsToolbarVisible] = useState(false);

  const toolbarPluginInstance = toolbarPlugin();
  const { Toolbar } = toolbarPluginInstance;
  const bookmarkPluginInstance = bookmarkPlugin();
  const { Bookmarks } = bookmarkPluginInstance;

  // const [pdfFile, setPdfFile] = useState<string | null>(null);

  const fileUrl = renderFile();
  // console.log("pdfFile", pdfFile);

  // const renderHighlightTarget = (props: RenderHighlightTargetProps) => (
  //   <div
  //     className={`bg-gray-100 flex absolute translate-y-2 z-10`}
  //     style={{
  //       left: `${props.selectionRegion.left}%`,
  //       top: `${props.selectionRegion.top + props.selectionRegion.height}%`,
  //     }}
  //   >
  //     <Tooltip
  //       position={Position.TopCenter}
  //       target={
  //         <Button onClick={props.toggle}>
  //           <AiOutlineEdit />
  //         </Button>
  //       }
  //       content={() => <div style={{ width: "100px" }}>Add a note</div>}
  //       offset={{ left: 0, top: -8 }}
  //     />
  //   </div>
  // );

  // const highlightPluginInstance = highlightPlugin({
  //   renderHighlightTarget,
  // });

  const handleMouseEnter = () => setIsToolbarVisible(true);
  const handleMouseLeave = () => setIsToolbarVisible(false);

  return (
    <div className="bg-white h-full rounded-tr-[6px] overflow-auto viewerWindow pt-[0px] px-[0px]">
      <div className="flex relative h-full">
        <div
          onMouseEnter={handleMouseEnter}
          onMouseLeave={handleMouseLeave}
          className={`items-center bg-gray-100 bg-opacity-80 rounded-[9px] border top-2 flex left-1/2 p-1 absolute transform -translate-x-1/2 z-10 shadow-custom_unfocus transition-opacity duration-200 ${
            isToolbarVisible ? "opacity-100" : "opacity-0"
          }`}
        >
          <Toolbar>
            {(props: ToolbarSlot) => {
              const { ZoomIn, ZoomOut, Zoom } = props;
              return (
                <>
                  <div className="px-[2px] text-sm">
                    <Tooltip
                      position={Position.BottomCenter}
                      target={
                        <MinimalButton
                          ariaLabel="Bookmarks"
                          isSelected={sidebarOpened}
                          onClick={() => setSidebarOpened((opened) => !opened)}
                        >
                          <Icon size={16}>
                            <rect
                              x="0.5"
                              y="0.497"
                              width="22"
                              height="22"
                              rx="1"
                              ry="1"
                            />
                            <line x1="7.5" y1="0.497" x2="7.5" y2="22.497" />
                          </Icon>
                        </MinimalButton>
                      }
                      content={() => <div className="w-18">Bookmarks</div>}
                      offset={{ left: 0, top: 8 }}
                    />
                  </div>
                  {/* <div className="p-0 text-sm">
                    <ZoomOut8
                      {(props: RenderZoomOutProps) => (
                        <Tooltip
                          position={Position.BottomCenter}
                          target={
                            <button
                              className={`flex items-center justify-center h-8 w-8 rounded hover:bg-gray-200`}
                              onClick={props.onClick}
                            >
                              <AiOutlineZoomOut className="text-xl" />
                            </button>
                          }
                          content={() => <div className="w-16">Zoom out</div>}
                          offset={{ left: 0, top: 0 }}
                        ></Tooltip>
                      )}
                    </ZoomOut>
                  </div> */}

                  <div className="px-[2px] text-sm">
                    <ZoomOut></ZoomOut>
                  </div>
                  <div className="px-[2px] text-sm">
                    <Zoom></Zoom>
                  </div>
                  <div className="px-[2px] text-sm">
                    <ZoomIn></ZoomIn>
                  </div>
                </>
              );
            }}
          </Toolbar>
        </div>

        <div
          className={`text-sm text-gray-700 overflow-auto transition-width duration-400 ease-in-out ${
            sidebarOpened ? "w-[204px] ml-2" : "w-0 ml-0"
          }`}
        >
          <Bookmarks></Bookmarks>
        </div>

        <div className="flex-1 pt-1 overflow-hidden">
          <Worker workerUrl="https://unpkg.com/pdfjs-dist@3.4.120/build/pdf.worker.min.js">
            {fileUrl && (
              <Viewer
                fileUrl={fileUrl}
                plugins={[
                  bookmarkPluginInstance,
                  toolbarPluginInstance,
                  // highlightPluginInstance,
                ]}
                defaultScale={SpecialZoomLevel.PageWidth}
              />
            )}
          </Worker>
        </div>
      </div>
    </div>
  );
};
