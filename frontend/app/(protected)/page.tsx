"use client";
import { ChatWindow } from "@/components/ChatWindow";
import { TopTabbar } from "@/components/ChatTopTabbar";
import { ProtectedRoutes } from "@/lib/ProtectedRoutes";
import { useAuthStore } from "@/lib/stores/authStore";
import { useState } from "react";
// import { useSessionStore } from "@/lib/stores/sessionStore";

export default function Home({ params }: { params: { uid: string | null } }) {
  // useSessionStore.getState().resetSession();

  return (
    <div
      className={`flex-grow flex flex-col items-center rounded-tl-[6px] overflow-hidden bg-white border-gray-300 text-gray-800`}
    >
      <ChatWindow uid={params.uid} />
    </div>
  );
}
