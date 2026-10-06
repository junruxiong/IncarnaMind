"use client";
import { useAuthStore } from "@/lib/stores/authStore";
import { useEffect, useState } from "react";
import { TopTabbar } from "./ChatTopTabbar";
import { useSegmentStore } from "@/lib/stores/segmentStore";
import { ViewerWindow } from "./ViewerWindow";
import { VierwerTopTabbar } from "./ViewerTopTabbar";
import { useFileStore } from "@/lib/stores/fileStore";

export const Viewer = () => {
  const {
    lefBarWidth,
    viewerWidth,
    rodWidth,
    isViewerVisible,
    minChatWidth,
    minviewerWidth,
  } = useSegmentStore();
  const max_width =
    window.innerWidth - lefBarWidth - 2 * rodWidth - minChatWidth;

  const handleMouseDown = (e: React.MouseEvent<HTMLDivElement>) => {
    e.preventDefault();
    document.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("mouseup", handleMouseUp);
  };

  const handleMouseMove = (e: MouseEvent) => {
    const newWidth = Math.min(
      Math.max(minviewerWidth, e.clientX - lefBarWidth - rodWidth),
      max_width
    );
    useSegmentStore.setState({ viewerWidth: newWidth });
  };

  const handleMouseUp = () => {
    document.removeEventListener("mousemove", handleMouseMove);
    document.removeEventListener("mouseup", handleMouseUp);
  };

  if (!isViewerVisible) {
    return null;
  }

  return (
    <div className="flex relative bg-gray-200">
      <div style={{ width: `${viewerWidth}px` }}>
        <div className="h-screen flex flex-col">
          <VierwerTopTabbar />
          <ViewerWindow viewerWidth={viewerWidth} />
        </div>
      </div>
      <div
        className="cursor-col-resize bg-gray-200 flex items-center justify-center three-dots"
        onMouseDown={handleMouseDown}
        style={{ width: `${rodWidth}px` }}
      />
    </div>
  );
};
