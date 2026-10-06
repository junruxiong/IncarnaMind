import { mergeAttributes, Node } from "@tiptap/core";
import { ReactNodeViewRenderer } from "@tiptap/react";
import {
  TextContainer,
  QAContainer,
  OutputComponent,
  QueryComponent,
  TitleContainer,
  CodeBlockComponent,
  RootBlockComponent,
  TextComponent,
} from "./Components";

// Extend the Commands interface to include ComponentCommands
declare module "@tiptap/core" {
  interface Commands<ReturnType> {
    TextContainerNodeCommands: {
      setTextContainerNode: (position?: number) => ReturnType;
    };
    QAContainerNodeCommands: {
      setQAContainerNode: (position?: number) => ReturnType;
      setDeleteQAContainerNode: (commands: any, state: any) => ReturnType;
    };
    QueryNodeCommands: {
      setQueryNode: (position?: number) => ReturnType;
    };
    OutputNodeCommands: {
      setOutputNode: (position?: number) => ReturnType;
    };
  }
}

export const TitleNode = Node.create({
  name: "titleContainer",
  group: "titleContainer",
  content: "block*",
  // draggable: true,
  // selectable: false, // Node isn't selectable
  inline: false, // Node is a block-level element
  priority: 1000, // Priority for node resolution

  // Default options for the node
  addOptions() {
    return {
      HTMLAttributes: {},
    };
  },

  parseHTML() {
    return [
      {
        tag: "title-component",
      },
    ];
  },

  renderHTML({ HTMLAttributes }) {
    return ["title-component", mergeAttributes(HTMLAttributes), 0];
  },

  addNodeView() {
    return ReactNodeViewRenderer(TitleContainer);
  },
});

export const TextContainerNode = Node.create({
  name: "textContainer",
  group: "textContainer",
  // codeBlock or block
  content: "textComponent*",
  draggable: true,
  // selectable: false, // Node isn't selectable
  inline: false, // Node is a block-level element
  priority: 2000, // Priority for node resolution

  // Default options for the node
  addOptions() {
    return {
      HTMLAttributes: {},
    };
  },

  parseHTML() {
    return [
      {
        tag: "text-container",
      },
    ];
  },

  renderHTML({ HTMLAttributes }) {
    return ["text-container", mergeAttributes(HTMLAttributes), 0];
  },

  addNodeView() {
    return ReactNodeViewRenderer(TextContainer);
  },
});

export const TextNode = Node.create({
  name: "textComponent",
  group: "textContainer", // Update group to match the parent's content rule
  // group: "block",
  atom: true,
  content: "block*",
  // draggable: true,
  // selectable: false, // Node isn't selectable
  inline: false, // Node is a block-level element
  priority: 2000, // Priority for node resolution

  // Default options for the node
  addOptions() {
    return {
      HTMLAttributes: {},
    };
  },

  parseHTML() {
    return [
      {
        tag: "text-component",
      },
    ];
  },

  renderHTML({ HTMLAttributes }) {
    return ["text-component", mergeAttributes(HTMLAttributes), 0];
  },

  addNodeView() {
    return ReactNodeViewRenderer(TextComponent);
  },
});

export const QAContainerNode = Node.create({
  name: "qaContainer",
  group: "qaContainer",
  // content: "block* outputComponent*", // Allow OutputComponent as a child
  content: "(queryComponent?|outputComponent)+", // Allow OutputComponent as a child
  draggable: true,
  // selectable: false, // Node isn't selectable
  inline: false, // Node is a block-level element
  priority: 1000, // Priority for node resolution

  // Default options for the node
  addOptions() {
    return {
      HTMLAttributes: {},
    };
  },

  parseHTML() {
    return [
      {
        tag: "qa-container",
      },
    ];
  },

  renderHTML({ HTMLAttributes }) {
    return ["qa-container", mergeAttributes(HTMLAttributes), 0];
  },

  addNodeView() {
    return ReactNodeViewRenderer(QAContainer);
  },
});

export const QueryNode = Node.create({
  name: "queryComponent",
  group: "qaContainer", // Update group to match the parent's content rule
  // group: "block",
  content: "block*",
  // draggable: true,
  // selectable: false, // Node isn't selectable
  inline: false, // Node is a block-level element
  priority: 2000, // Priority for node resolution

  addCommands() {
    return {
      setQueryNode:
        () =>
        ({ state, chain }) => {
          // get the current node

          const { selection } = state;
          const { from, to } = selection;
          let preventDeletion = false;
          let currentNode;

          const { empty, $anchor } = state.selection;
          console.log("empty", empty);

          state.doc.nodesBetween(from, to, (node) => {
            if (node.type.name === "queryComponent") {
              currentNode = node;
              preventDeletion = true;
            }
          });

          console.log("node", preventDeletion);
          console.log("nodeddd", $anchor.parent.content);

          if (empty && $anchor.parent.content.size === 0 && preventDeletion) {
            return false; // Prevent deletion
          } else {
            return false;
          }
        },
    };
  },

  // addKeyboardShortcuts() {
  //   return {
  //     Backspace: () => this.editor.commands.setQueryNode(),
  //   };
  // },

  // Default options for the node
  addOptions() {
    return {
      HTMLAttributes: {},
    };
  },

  parseHTML() {
    return [
      {
        tag: "query-component",
      },
    ];
  },

  renderHTML({ HTMLAttributes }) {
    return ["query-component", mergeAttributes(HTMLAttributes), 0];
  },

  addNodeView() {
    return ReactNodeViewRenderer(QueryComponent);
  },
});

export const OutputNode = Node.create({
  name: "outputComponent",
  group: "qaContainer", // Update group to match the parent's content rule
  // group: "block",
  atom: true,
  content: "block*",
  // draggable: true,
  // selectable: false, // Node isn't selectable
  inline: false, // Node is a block-level element
  priority: 2000, // Priority for node resolution

  // Default options for the node
  addOptions() {
    return {
      HTMLAttributes: {},
    };
  },

  parseHTML() {
    return [
      {
        tag: "output-component",
      },
    ];
  },

  renderHTML({ HTMLAttributes }) {
    return ["output-component", mergeAttributes(HTMLAttributes), 0];
  },

  addNodeView() {
    return ReactNodeViewRenderer(OutputComponent);
  },
});

// CodeBlockComponent

// Create and export the RootBlock node
export const RootBlock = Node.create({
  name: "rootblock",
  group: "rootblock",
  content: "block*", // Ensure only one block element inside the rootblock
  draggable: true, // Make the node draggable
  // selectable: false, // Node isn't selectable
  inline: false, // Node is a block-level element
  priority: 1000, // Priority for node resolution

  // Default options for the node
  addOptions() {
    return {
      HTMLAttributes: {},
    };
  },

  // Rules to parse the node from HTML
  parseHTML() {
    return [
      {
        tag: 'div[data-type="rootblock"]',
      },
    ];
  },

  // Rules to render the node to HTML
  renderHTML({ HTMLAttributes }) {
    return [
      "div",
      mergeAttributes(HTMLAttributes, { "data-type": "rootblock" }),
      0,
    ];
  },

  // Use ReactNodeViewRenderer to render the node view with the RootBlockComponent
  addNodeView() {
    return ReactNodeViewRenderer(RootBlockComponent);
  },
});
