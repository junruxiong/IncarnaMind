import type { Editor, JSONContent, Range } from "@tiptap/core";
import type { ReactNode } from "react";
import { QUESTION_BLOCK, type Skill } from "../../../core/api";
import type { MessageKey } from "../../../shared/i18n";
import { SkillIcon } from "../components/icons";

/**
 * One entry of the slash menu. To add one, add it to `noteSlashItems`, or
 * give `SlashMenu.configure({ items })` a list that includes it (as the
 * editor does with `skillSlashItems`).
 */
export interface SlashItem {
  /** A stable name. Tests find the item by it (`slash-item-<id>`), and typing it after the slash finds the item. */
  id: string;
  /** What the menu shows: a dictionary key, or text shown as it is (e.g. a Skill's name). */
  label: MessageKey | { text: string };
  /** Shown when the item is pointed at, e.g. what a Skill is for. */
  hint?: string;
  /** More words that find the item when typed after the slash, whatever the interface language. */
  keywords?: readonly string[];
  /** A small glyph before the label. */
  icon: ReactNode;
  /** Replaces the typed slash and query (`range`) with what the item inserts. */
  run(editor: Editor, range: Range): void;
}

const turnInto =
  (type: string, attrs?: Record<string, unknown>) =>
  (editor: Editor, range: Range): void => {
    editor.chain().focus().deleteRange(range).setNode(type, attrs).run();
  };

/**
 * Puts a new Block after the one the cursor is in, or in its place if that is
 * an empty paragraph, and returns where it starts. A Block added at the very end
 * gets an empty paragraph after it, to carry on writing in.
 */
function insertBlock(editor: Editor, range: Range, block: JSONContent): number | null {
  editor.commands.deleteRange(range);
  const { $from } = editor.state.selection;
  if ($from.depth === 0) return null;
  const replacesEmptyLine =
    $from.depth === 1 && $from.parent.type.name === "paragraph" && $from.parent.childCount === 0;
  const from = replacesEmptyLine ? $from.before(1) : $from.after(1);
  const to = replacesEmptyLine ? $from.after(1) : from;
  const atEnd = to === editor.state.doc.content.size;
  editor
    .chain()
    .insertContentAt({ from, to }, atEnd ? [block, { type: "paragraph" }] : block, {
      updateSelection: false,
    })
    .run();
  return from;
}

/** The slash menu's entries for writing Notes. */
export const noteSlashItems: readonly SlashItem[] = [
  {
    id: "text",
    label: "editor.slash.text",
    keywords: ["paragraph", "plain"],
    icon: "T",
    run: turnInto("paragraph"),
  },
  {
    id: "heading-1",
    label: "editor.slash.heading1",
    keywords: ["h1", "title"],
    icon: "H1",
    run: turnInto("heading", { level: 1 }),
  },
  {
    id: "heading-2",
    label: "editor.slash.heading2",
    keywords: ["h2", "subtitle"],
    icon: "H2",
    run: turnInto("heading", { level: 2 }),
  },
  {
    id: "heading-3",
    label: "editor.slash.heading3",
    keywords: ["h3"],
    icon: "H3",
    run: turnInto("heading", { level: 3 }),
  },
  {
    id: "code",
    label: "editor.slash.codeBlock",
    keywords: ["code", "codeblock", "pre"],
    icon: "{ }",
    run: turnInto("codeBlock"),
  },
  {
    id: "math",
    label: "editor.slash.math",
    keywords: ["math", "equation", "formula", "latex", "katex", "tex"],
    icon: "∑",
    run(editor, range) {
      const pos = insertBlock(editor, range, { type: "blockMath", attrs: { latex: "" } });
      if (pos !== null) editor.commands.editMath(pos);
    },
  },
  {
    id: "inline-math",
    label: "editor.slash.inlineMath",
    keywords: ["math", "equation", "formula", "latex", "katex", "tex", "inline"],
    icon: "x²",
    run(editor, range) {
      editor
        .chain()
        .deleteRange(range)
        .insertContent({ type: "inlineMath", attrs: { latex: "" } })
        .run();
      editor.commands.editMath(range.from);
    },
  },
];

/**
 * The slash menu's entries for the enabled Skills, labelled with their names
 * as they are. Choosing one forces the Skill on the Question the cursor is
 * in; in a Note, it goes to the composer as a chip, for the Question asked
 * there next (`onComposer`), and the composer takes the focus.
 */
export function skillSlashItems(
  skills: readonly Skill[],
  describe: (skill: Skill) => string = (skill) => skill.description,
  onComposer: (skill: string) => void = () => undefined,
): SlashItem[] {
  return skills
    .filter((skill) => skill.enabled)
    .map((skill) => ({
      id: `skill-${skill.name}`,
      label: { text: skill.name },
      hint: describe(skill),
      keywords: ["skill"],
      icon: <SkillIcon className="size-3.5" />,
      run(editor, range) {
        const inQuestion = editor.state.selection.$from.parent.type.name === QUESTION_BLOCK;
        if (!inQuestion) {
          editor.chain().focus().deleteRange(range).run();
          onComposer(skill.name);
          return;
        }
        editor
          .chain()
          .focus()
          .deleteRange(range)
          .updateAttributes(QUESTION_BLOCK, { forcedSkill: skill.name })
          .run();
      },
    }));
}

/**
 * The items `query` finds, in their order: those with a word of their label,
 * ID or keywords starting with it. A query in Chinese (or any non-Latin script)
 * finds the labels that contain it, as Chinese doesn't separate words.
 */
export function matchSlashItems(
  items: readonly SlashItem[],
  query: string,
  labelOf: (item: SlashItem) => string,
): SlashItem[] {
  const wanted = query.trim().toLowerCase();
  if (!wanted) return [...items];
  const nonLatin = /[\u0080-￿]/.test(wanted);
  return items.filter((item) => {
    const label = labelOf(item).toLowerCase();
    if (nonLatin) return label.includes(wanted);
    return [label, item.id, ...(item.keywords ?? [])]
      .flatMap((text) => text.toLowerCase().split(/[\s-]+/))
      .some((word) => word.startsWith(wanted));
  });
}
