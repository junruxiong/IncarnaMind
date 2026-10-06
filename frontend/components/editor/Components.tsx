import "./CodeBlockComponent.scss";
import FormControlLabel from "@mui/material/FormControlLabel";

import {
  AiOutlineHolder,
  AiOutlineClose,
  AiOutlineEdit,
  AiFillCaretRight,
  AiOutlineDoubleRight,
} from "react-icons/ai";
import { FcDownRight, FcStumbleupon } from "react-icons/fc";

import { Node } from "@tiptap/pm/model";
import { NodeViewContent, NodeViewProps, NodeViewWrapper } from "@tiptap/react";
import { useEffect, useState } from "react";
import { IOSSwitch } from "../ui/buttons";
import Tooltip from "@mui/material/Tooltip";

export const TitleContainer: React.FC<NodeViewProps> = ({
  node,
  getPos,
  editor,
}) => {
  const handleCreateNodeAfter = (type: string) => {
    createNodeAfter(node, getPos, editor, type);
  };

  const handleChangeNodeType = () => {
    changeNode(node, getPos, editor, "qaContainer");
  };

  const handleDeleteNode = () => {
    deleteNode(node, getPos, editor);
  };

  const id = node.attrs.id;
  // console.log("node id", id);
  // console.log("node", node.toString());

  return (
    <NodeViewWrapper className="title-container-with-content text-custom bg-gray-0 mb-3 rounded-[9px]">
      <div>
        <button draggable data-drag-handle className="cursor-grab2">
          <AiOutlineHolder className="h-3 w-5" />
        </button>

        <button
          type="button"
          onClick={() => handleCreateNodeAfter("textComponent")}
          className=""
        >
          +1
        </button>
        <button
          type="button"
          onClick={() => handleCreateNodeAfter("queryComponent")}
          className=""
        >
          +2
        </button>

        <button type="button" onClick={handleDeleteNode} className="">
          -
        </button>

        <NodeViewContent className="content" />
      </div>
    </NodeViewWrapper>
  );
};

export const TextContainer: React.FC<NodeViewProps> = ({
  node,
  getPos,
  editor,
}) => {
  const handleCreateNodeAfter = (type: string) => {
    createNodeAfter(node, getPos, editor, type);
  };

  const handleDeleteNode = () => {
    deleteNode(node, getPos, editor);
  };

  const [switchState, setSwitchState] = useState(true);
  // Handler function to update the state
  const handleSwitchChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setSwitchState(event.target.checked);
  };

  const id = node.attrs.id;
  // console.log("node id", id);
  // console.log("node", node.toString());

  return (
    <NodeViewWrapper className="text-container-with-content relative text-custom my- bg-gray-0 rounded-[9px] group">
      <NodeViewContent className="content" />

      {/* Holder Button */}
      <div className="absolute top-3 -left-6 opacity-0 hover:opacity-100 transition-opacity duration-400">
        <button draggable data-drag-handle className="">
          <AiOutlineHolder className="h-5 w-5" />
        </button>
      </div>

      <div
        title="Remove/Add this text to prompt"
        className="absolute top-[6px] left-2"
      >
        <IOSSwitch checked={switchState} onChange={handleSwitchChange} />
      </div>

      {/* Close Button */}
      <div className="absolute top-[6px] right-2 opacity-0 group-hover:opacity-100 transition-opacity duration-400">
        <button
          type="button"
          onClick={handleDeleteNode}
          className="rounded-full hover:bg-gray-100 p-1 m-1 text-gray-500"
        >
          <AiOutlineClose className="h-4 w-4" />
        </button>
      </div>

      <div className="absolute -bottom-[22px] w-full flex-col items-center opacity-0 hover:opacity-100 transition-opacity duration-400">
        <div className="flex flex-col justify-center items-center relative text-sm z-10">
          <div className="relative z-20">
            <button
              type="button"
              onClick={() => handleCreateNodeAfter("textComponent")}
              className="m-1 px-3 py-[2px] bg-white hover:bg-gray-100 rounded-[9px] shadow-small"
            >
              Add Text
            </button>
            <button
              type="button"
              onClick={() => handleCreateNodeAfter("queryComponent")}
              className="m-1 px-3 py-[2px] bg-white hover:bg-gray-100 rounded-[9px] shadow-small"
            >
              Add Query
            </button>
          </div>

          {/* Line */}
          <div className="absolute top-1/2 left-0 w-full h-[1px] bg-gray-300 z-10"></div>
        </div>
      </div>
    </NodeViewWrapper>
  );
};

export const TextComponent: React.FC<NodeViewProps> = ({
  node,
  getPos,
  editor,
}) => {
  const handleDeleteNode = () => {
    deleteNode(node, getPos, editor);
  };

  const handleCreateNodeAfter = (type: string) => {
    createNodeAfter(node, getPos, editor, type);
  };

  const id = node.attrs.id;
  // console.log("node id", id);
  // console.log("node", node.toString());

  return (
    <NodeViewWrapper className="text-component-with-content relative  text-custom rounded-[9px]">
      <NodeViewContent className="content mx-9 py-2" />
      <div>
        {/* <div className="absolute top-3 left-[9px]">
          <button
            type="button"
            onClick={() => handleCreateNodeAfter("outputComponent")}
            className="bg-gray-800 rounded-full hover:bg-blac"
          >
            <AiFillCaretRight className="pl-[2px] h-5 w-5 text-white" />
          </button>
        </div> */}
      </div>
    </NodeViewWrapper>
  );
};

export const QAContainer: React.FC<NodeViewProps> = ({
  node,
  getPos,
  editor,
}) => {
  const handleCreateNodeAfter = (type: string) => {
    createNodeAfter(node, getPos, editor, type);
  };

  const handleChangeNodeType = () => {
    changeNode(node, getPos, editor, "textComponent");
  };

  const handleDeleteNode = () => {
    deleteNode(node, getPos, editor);
  };

  const id = node.attrs.id;
  // console.log("node id", id);
  // console.log("node", node.toString());

  return (
    <NodeViewWrapper className="text-container-with-content relative text-custom my-1 rounded-[9px] group">
      <NodeViewContent className="content" />

      {/* Holder Button */}
      <div className="absolute top-3 -left-6 opacity-0 hover:opacity-100 transition-opacity duration-400">
        <button draggable data-drag-handle className="">
          <AiOutlineHolder className="h-5 w-5" />
        </button>
      </div>

      {/* Close Button */}
      <div className="absolute top-[6px] right-2 opacity-0 group-hover:opacity-100 transition-opacity duration-400">
        <button
          type="button"
          onClick={handleDeleteNode}
          className="rounded-full hover:bg-gray-100 p-1 m-1 text-gray-500"
        >
          <AiOutlineClose className="h-4 w-4" />
        </button>
      </div>

      <div className="absolute -bottom-[22px] w-full flex-col items-center opacity-0 hover:opacity-100 transition-opacity duration-400">
        <div className="flex flex-col justify-center items-center relative text-sm z-10">
          <div className="relative z-20">
            <button
              type="button"
              onClick={() => handleCreateNodeAfter("textComponent")}
              className="m-1 px-3 py-[2px] bg-white hover:bg-gray-100 rounded-[9px] shadow-small"
            >
              Add Text
            </button>
            <button
              type="button"
              onClick={() => handleCreateNodeAfter("queryComponent")}
              className="m-1 px-3 py-[2px] bg-white hover:bg-gray-100 rounded-[9px] shadow-small"
            >
              Add Query
            </button>
          </div>

          {/* Line */}
          <div className="absolute top-1/2 left-0 w-full h-[1px] bg-gray-300 z-10"></div>
        </div>
      </div>
    </NodeViewWrapper>
  );
};

export const QueryComponent: React.FC<NodeViewProps> = ({
  node,
  getPos,
  editor,
}) => {
  const handleChangeNodeType = () => {
    changeNode(node, getPos, editor, "textComponent");
  };

  const handleDeleteNode = () => {
    deleteNode(node, getPos, editor);
  };

  const handleCreateNodeAfter = (type: string) => {
    createNodeAfter(node, getPos, editor, type);
  };

  const id = node.attrs.id;
  // console.log("node id", id);
  // console.log("node", node.toString());

  // if curent node is empty, prevent user from deleting it
  useEffect(() => {
    if (node.content.size === 0) {
      setTimeout(() => {
        handleDeleteNode();
      }, 0);
    }
  }, [node.content, handleDeleteNode]);

  return (
    <NodeViewWrapper className="query-component-with-content relative  text-custom bg-slate-50 rounded-[9px]">
      <NodeViewContent className="content mx-9 py-2" />
      <div>
        <div className="absolute top-3 left-[9px]">
          <button
            type="button"
            onClick={() => handleCreateNodeAfter("outputComponent")}
            className="bg-gray-800 rounded-full hover:bg-blac"
          >
            <AiFillCaretRight className="pl-[2px] h-5 w-5 text-white" />
          </button>
        </div>
      </div>
    </NodeViewWrapper>
  );
};

export const OutputComponent: React.FC<NodeViewProps> = ({
  node,
  getPos,
  editor,
}) => {
  const handleChangeNodeType = () => {
    changeNode(node, getPos, editor, "textComponent");
  };

  const handleDeleteNode = () => {
    deleteNode(node, getPos, editor);
  };

  const handleCreateNodeAfter = (type: string) => {
    createNodeAfter(node, getPos, editor, type);
  };

  const id = node.attrs.id;
  // console.log("node id", id);
  // console.log("node", node.toString());

  return (
    <NodeViewWrapper className="output-component-with-content relative mt-3 text-custom bg-gray-0 rounded-[9px] group">
      <NodeViewContent className="content mx-9 py-2" contentEditable={true} />

      {/* <div className="absolute top-3 left-[9px]">
        <button
          type="button"
          onClick={handleChangeNodeType}
          className="text-gray-500"
        >
          <AiOutlineEdit className="h-5 w-5" />
        </button>
      </div> */}

      <div className="absolute top-3 left-2 opacity-0 group-hover:opacity-100 transition-opacity duration-400">
        <button
          type="button"
          onClick={handleChangeNodeType}
          className="text-gray-500"
        >
          <AiOutlineEdit className="h-5 w-5" />
        </button>
      </div>

      {/* Close Button */}
      <div className="absolute top-[6px] right-2 opacity-0 group-hover:opacity-100 transition-opacity duration-400">
        <button
          type="button"
          onClick={handleDeleteNode}
          className="rounded-full hover:bg-gray-100 p-1 m-1 text-gray-500"
        >
          <AiOutlineClose className="h-4 w-4" />
        </button>
      </div>
    </NodeViewWrapper>
  );
};

export const CodeBlockComponent: React.FC<NodeViewProps> = ({
  node: {
    attrs: { language: defaultLanguage },
  },
  updateAttributes,
  extension,
}) => (
  <NodeViewWrapper className="code-block">
    <select
      contentEditable={false}
      defaultValue={defaultLanguage}
      onChange={(event) => updateAttributes({ language: event.target.value })}
    >
      <option value="null">auto</option>
      <option disabled>—</option>
      {extension.options.lowlight
        .listLanguages()
        .map((lang: any, index: number) => (
          <option key={index} value={lang}>
            {lang}
          </option>
        ))}
    </select>
    <pre>
      <NodeViewContent as="code" />
    </pre>
  </NodeViewWrapper>
);

export const createNodeAfter = (
  node: Node,
  getPos: () => number,
  editor: any,
  type: string
) => {
  const pos = getPos() + node.nodeSize;
  editor
    .chain()
    .insertContentAt(pos, {
      type: type,
      content: [
        {
          type: "paragraph",
        },
      ],
    })
    .focus(pos + 3)
    .run();

  // const newState = editor.state;
  // const newNode = newState.doc.nodeAt(pos);
  // if (newNode) {
  //   console.log("New node attributes:", newNode.attrs);
  //   // You can perform operations with newNode.attrs here
  // } else {
  //   console.error("New node was not found");
  // }
};

export const sendMessage = (
  node: Node,
  getPos: () => number,
  editor: any,
  type: string
) => {
  const pos = getPos() + node.nodeSize;
  const nextNodePos = pos + 1; // Position of the next node
  const nextNode = editor.state.doc.nodeAt(nextNodePos);

  // Check if the next node is 'outputComponent'
  if (nextNode && nextNode.type.name === "outputComponent") {
    // Modify the content of the next 'outputComponent' node
    // This can be done based on your specific requirements
    // For example, you might want to update its attributes or content
    editor
      .chain()
      .command(({ tr, state }) => {
        let transaction = tr.setNodeMarkup(nextNodePos, undefined, {
          ...nextNode.attrs,
          // modify attributes here
        });
        editor.view.dispatch(transaction);
        return true;
      })
      .focus(nextNodePos)
      .run();
  } else {
    // If the next node is not 'outputComponent', insert a new node
    editor
      .chain()
      .insertContentAt(pos, {
        type: type,
        content: [{ type: "paragraph" }],
      })
      .focus(pos + 0)
      .run();
  }
};

const changeNode = (
  node: Node,
  getPos: () => number,
  editor: any,
  newType: string
) => {
  const pos = getPos();

  const newNode = {
    type: newType,
    attrs: {
      ...node.attrs,
      id: node.attrs.id,
    },
    content: node.content.toJSON(),
  };

  editor
    .chain()
    .deleteRange({ from: pos, to: pos + node.nodeSize })
    .insertContentAt(pos, newNode)
    .run();
};

const deleteNode = (node: Node, getPos: () => number, editor: any) => {
  const pos = getPos();

  editor
    .chain()
    .deleteRange({ from: pos, to: pos + node.nodeSize })
    .run();

  // console.log("delete node", node.attrs.id);
};

export const RootBlockComponent: React.FC<NodeViewProps> = ({
  node,
  getPos,
  editor,
}) => {
  // Function to create a new node immediately after the current node
  const createNodeAfter = () => {
    // Calculate the position right after the current node
    const pos = getPos() + node.nodeSize;

    // Use the editor's command to insert a new "rootblock" node at the calculated position
    editor
      .chain()
      .insertContentAt(pos, {
        type: "rootblock",
        content: [
          {
            type: "paragraph",
          },
        ],
      })
      .focus(pos + 3) // Focus on the new block (you might need to adjust the position based on your exact requirements)
      .run();
  };

  // Render the custom node view
  return (
    <NodeViewWrapper
      as="div"
      className="group relative mx-auto flex w-full gap-2"
    >
      <div className="relative mx-auto w-full max-w-4xl bg-slate-50">
        {/* Container for buttons that appear on hover */}
        <div
          className="absolute -left-12 -top-0 flex w-12 gap-1 "
          aria-label="left-menu"
        >
          {/* Button to add a new node after the current node */}
          <button type="button" onClick={createNodeAfter} className="">
            +
          </button>
          {/* Draggable handle button to allow rearranging nodes */}
          <button draggable data-drag-handle className="cursor-grab">
            <AiOutlineHolder className="h-3 w-5" />
          </button>
        </div>
        {/* Area where the node's actual content will be rendered */}
        <NodeViewContent className="w-full" />
      </div>
    </NodeViewWrapper>
  );
};
