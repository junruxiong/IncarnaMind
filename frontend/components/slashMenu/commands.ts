import { Extension } from "@tiptap/core";
import Suggestion from "@tiptap/suggestion";

interface CommandArg {
  editor: any; // Replace 'any' with the actual type of 'editor'
  range: any; // Replace 'any' with the actual type of 'range'
  props: any; // Replace 'any' with the actual type of 'props'
}

const Commands = Extension.create({
  name: "mention",
  priority: 3000,
  addOptions() {
    return {
      suggestion: {
        char: "/",
        startOfLine: false,
        command: ({ editor, range, props }: CommandArg) => {
          props.command({ editor, range });
        },
      },
    };
  },

  addProseMirrorPlugins() {
    return [
      Suggestion({
        editor: this.editor,
        ...this.options.suggestion,
      }),
    ];
  },
});

export default Commands;
