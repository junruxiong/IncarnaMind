import "./styles.scss";
import StarterKit from "@tiptap/starter-kit";
import {
  TextContainerNode,
  QAContainerNode,
  QueryNode,
  OutputNode,
  TitleNode,
  TextNode,
} from "./nodes";
import { mergeAttributes, Node } from "@tiptap/core";
import {
  ReactNodeViewRenderer,
  BubbleMenu,
  EditorContent,
  useEditor,
  FloatingMenu,
} from "@tiptap/react";
import { TextContainer, QAContainer, CodeBlockComponent } from "./Components";
import { Extension } from "@tiptap/core";
import Typography from "@tiptap/extension-typography";
import Highlight from "@tiptap/extension-highlight";
import UniqueID from "@tiptap-pro/extension-unique-id";
import HardBreak from "@tiptap/extension-hard-break";
import Focus from "@tiptap/extension-focus";
import BulletList from "@tiptap/extension-bullet-list";

import CodeBlockLowlight from "@tiptap/extension-code-block-lowlight";
import Paragraph from "@tiptap/extension-paragraph";
import Text from "@tiptap/extension-text";
import css from "highlight.js/lib/languages/css";
import js from "highlight.js/lib/languages/javascript";
import ts from "highlight.js/lib/languages/typescript";
import html from "highlight.js/lib/languages/xml";
// load all highlight.js languages
import { all, createLowlight } from "lowlight";
import Commands from "../slashMenu/commands";
import getSuggestionItems from "../slashMenu/items";
import renderItems from "../slashMenu/renderItems";
import { Mathematics } from "@tiptap-pro/extension-mathematics";
import "katex/dist/katex.min.css";

const lowlight = createLowlight(all);
// you can also register languages
lowlight.register("html", html);
lowlight.register("css", css);
lowlight.register("js", js);
lowlight.register("ts", ts);

const Document = Node.create({
  name: "doc",
  topNode: true,
  content: "(textContainer?|qaContainer?)+",
  // content: "textContainer+",
});

const Keymap = Extension.create({
  name: "enterToLineBreak",

  addKeyboardShortcuts() {
    return {
      "Shift-Enter": ({ editor }) => editor.commands.splitBlock(),
      Enter: ({ editor }) => editor.commands.setHardBreak(),

      // Backspace: ({ editor }) => {
      //   const { empty, $anchor } = editor.state.selection;

      //   // Check if selection is empty and the current node is empty
      //   if (empty && $anchor.parent.content.size === 0) {
      //     console.log("Empty selection");
      //     return true; // Prevent default delete behavior
      //   }
      //   return false;
      // },
    };
  },
});

export const Extensions = [
  Document,
  QAContainerNode,
  QueryNode,
  OutputNode,
  TextContainerNode,
  TextNode,
  // Keymap,
  StarterKit.configure({
    document: false,
    codeBlock: false,
  }),
  Typography,

  Commands.configure({
    suggestion: {
      items: getSuggestionItems,
      render: renderItems,
    },
  }),

  Mathematics,
  Highlight,
  CodeBlockLowlight.extend({
    addNodeView() {
      return ReactNodeViewRenderer(CodeBlockComponent);
    },
  }).configure({ lowlight }),

  UniqueID.configure({
    types: ["textComponent", "queryComponent", "outputComponent"],
  }),
  Focus.configure({
    className: "has-focus",
    mode: "shallowest",
  }),
];
