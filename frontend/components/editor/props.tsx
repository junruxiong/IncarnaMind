import { EditorProps } from "@tiptap/pm/view";

const Props: EditorProps = {
  attributes: {
    class: [
      "prose-gray",
      "prose-headings:font-bold",
      "prose-img:rounded-[6px]",
      "prose-h1:text-3xl",
      "prose-h1:my-4",
      "prose-h2:text-2xl",
      "prose-h2:my-3",
      "prose-h3:text-xl",
      "prose-h3:my-2",
      "prose-h4:text-lg",
      "prose-h4:my-1",
      "prose-h5:text-base",
      "prose-h4:my-[2px]",
      "prose-code:text-sm",
      "prose-blockquote:ml-4",
      "prose-blockquote:pl-2",
      "prose-blockquote:border-l-2",
      "prose-blockquote:border-gray-200",
    ].join(" "),
  },
};

export default Props;
