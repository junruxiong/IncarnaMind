"use client";
import { ChatWindow } from "@/components/ChatWindow";

export default function ChatPage({ params }: { params: { uid: string } }) {
  // params.uid;
  return (
    <div
      className={`flex-grow flex flex-col items-center rounded-tl-[6px] overflow-hidden bg-white border-gray-300 text-gray-800`}
    >
      <ChatWindow uid={params.uid} />
    </div>
  );
}
