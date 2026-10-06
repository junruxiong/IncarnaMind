import Popper from "@mui/material/Popper";
import { Editor } from "@tiptap/react";

// Defining types for your items and command function parameters

const getSuggestionItems = (query: any) => {
  const queryString = typeof query === "string" ? query : "";
  console.log("query", query);

  return [
    {
      title: "Heading 1",
      command: ({ editor, range }: { editor: Editor; range: any }) => {
        console.log("range", range);
        editor
          .chain()
          .focus()
          .deleteRange(range)
          .setNode("heading", { level: 1 })
          .run();
      },
    },
    {
      title: "Heading 2",
      command: ({ editor, range }: { editor: Editor; range: any }) => {
        console.log("range", range);
        editor
          .chain()
          .focus()
          .deleteRange(range)
          .setNode("heading", { level: 2 })
          .run();
      },
    },
    {
      title: "Bold",
      command: ({ editor, range }: { editor: Editor; range: any }) => {
        editor.chain().focus().deleteRange(range).setMark("bold").run();
      },
    },
    {
      title: "Text",
      command: ({ editor, range }: { editor: Editor; range: any }) => {
        editor
          .chain()
          .focus()
          .deleteRange(range)
          .setNode("paragraph")
          .unsetMark("bold")
          .unsetMark("italic")
          .run();
      },
    },
  ]
    .filter((item) =>
      item.title.toLowerCase().startsWith(queryString.toLowerCase())
    )
    .slice(0, 10);
};

export default getSuggestionItems;
