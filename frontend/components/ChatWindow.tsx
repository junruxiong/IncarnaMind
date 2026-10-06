"use client";
import "./editor/styles.scss";
import React, { useState, useEffect, useRef } from "react";
import {
  useContent,
  useInfiniteScroll,
} from "@/lib/hooks/useMessage";
import { BubbleMenu, EditorContent, useEditor } from "@tiptap/react";
import { Node } from "@tiptap/pm/model";

import { useSessionStore } from "@/lib/stores/sessionStore";
import { Extensions } from "./editor/extensions";
import { MyBubbleMenu } from "./editor/BubbleMenu";
import Props from "./editor/props";
// import EditArea from "./editor/edtiArea";

const onNodeAdded = (addedNode: Node) => {
  console.log("Node added:", addedNode);
};

const onNodeDeleted = (id: string) => {
  console.log("Node deleted:", id);
};

const onOrderChanged = (oldOrder: string[], newOrder: string[]) => {
  // Logic when order changes
  console.log("Order changed:", oldOrder, newOrder);
};

interface ChatWindowProps {
  uid: string | null;
}

export const ChatWindow: React.FC<ChatWindowProps> = ({ uid }) => {
  const loadMoreRef = useRef(null);

  const { content, fetchNextPage, hasNextPage, isFetchingNextPage } =
    useContent(uid);

  useInfiniteScroll(
    loadMoreRef,
    hasNextPage,
    isFetchingNextPage,
    fetchNextPage
  );

  const activeSession = useSessionStore((state) => ({
    activeSession: state.activeSession,
  }));

  // console.log("messages", content);

  const editor = useEditor({
    extensions: Extensions,
    editorProps: Props,
    content: ``,
    onUpdate: ({ editor, transaction }) => {
      // Track order and node deletions
      const oldOrder: string[] = [];
      const oldIds = new Set<string>();
      transaction.before.descendants((node) => {
        if (node.attrs.id) {
          oldOrder.push(node.attrs.id);
          oldIds.add(node.attrs.id);
        }
      });

      const newOrder: string[] = [];
      editor.state.doc.descendants((node) => {
        if (node.attrs.id) {
          newOrder.push(node.attrs.id);
          // Check for node additions
          if (!oldIds.has(node.attrs.id)) {
            onNodeAdded(node);
          }
        }
      });

      // Check for node deletions
      oldIds.forEach((id) => {
        if (!newOrder.includes(id)) {
          onNodeDeleted(id);
        }
      });

      // Check for order changes
      if (JSON.stringify(oldOrder) !== JSON.stringify(newOrder)) {
        onOrderChanged(oldOrder, newOrder);
      }

      // print the html

      console.log("html", editor.getHTML());
    },
  });

  useEffect(() => {
    if (editor && content) {
      // Use a microtask to defer the update
      Promise.resolve().then(() => {
        editor.commands.setContent(content);
      });
    }
  }, [content, editor]);

  return (
    <div className="flex-grow px-10 mt-2 w-full flex flex-col text-gray-800 overflow-auto">
      <p className={`p-3 mt-4 text-3xl font-medium break-words`}>
        {activeSession.activeSession?.session_name ?? "New Mind"}
      </p>
      <MyBubbleMenu editor={editor} />
      <EditorContent editor={editor!} />
    </div>
  );
};
