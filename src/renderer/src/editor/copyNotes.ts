/**
 * Copying Blocks with their Citations (#102): a copy's Citations as footnotes. The HTML keeps
 * each Citation as it was, so it pastes back into IncarnaMind as a Citation, and shows it as
 * Word writes a footnote: a numbered marker in the text and a footnote list after it. Word
 * pastes those as footnotes; where footnotes can't be pasted, they read as numbered markers with
 * a list of Sources. The plain text has the same markers and a Sources list.
 */
import { getTextBetween, getTextSerializersFromSchema, type TextSerializer } from "@tiptap/core";
import type { Fragment, Schema } from "@tiptap/pm/model";
import { CITATION_NODE, type CitationAttributes, FOOTNOTE_NODE } from "../../../core/api";
import { translate } from "../../../shared/i18n";
import { type SourceContext, sourceText } from "../../../shared/sources";

type Cited = Partial<CitationAttributes>;

/** The list the footnotes are kept in; pasting it back into a Mind skips it (see `Citation`). */
export const NOTES_ATTRIBUTE = "data-citation-notes";

/** The Citations in a fragment that take a number, top to bottom: not those in a footnote the User wrote. */
export function citationsIn(fragment: Fragment): Cited[] {
  const found: Cited[] = [];
  fragment.descendants((node) => {
    if (node.type.name === FOOTNOTE_NODE) return false;
    if (node.type.name === CITATION_NODE) found.push(node.attrs as Cited);
    return true;
  });
  return found;
}

/** A marker "[1]" and what it points at, as Word's clipboard names a footnote's two ends. */
function marker(document: Document, number: number, end: "reference" | "note"): HTMLElement {
  const anchor = document.createElement("a");
  anchor.setAttribute("style", `mso-footnote-id:ftn${number}`);
  anchor.setAttribute("href", end === "reference" ? `#_ftn${number}` : `#_ftnref${number}`);
  anchor.setAttribute("name", end === "reference" ? `_ftnref${number}` : `_ftn${number}`);
  anchor.setAttribute("title", "");
  const mark = document.createElement(end === "reference" ? "sup" : "span");
  mark.setAttribute("class", "MsoFootnoteReference");
  mark.textContent = `[${number}]`;
  anchor.append(mark);
  return anchor;
}

/** A footnote's text, its address a link when it has one. */
function noteText(document: Document, text: string, link: string | null): Node[] {
  const at = link === null ? -1 : text.indexOf(link);
  if (link === null || at < 0) return [document.createTextNode(text)];
  const a = document.createElement("a");
  a.setAttribute("href", link);
  a.textContent = link;
  return [
    document.createTextNode(text.slice(0, at)),
    a,
    document.createTextNode(text.slice(at + link.length)),
  ];
}

/**
 * Turns the Citations in serialized HTML into numbered footnote markers and adds the footnote
 * list after it. Each Citation keeps its own element and its data, so it pastes back as one.
 * `citations` are the fragment's own, in the same order as the elements.
 */
export function addFootnotes(
  root: ParentNode,
  citations: readonly Cited[],
  context: SourceContext,
): void {
  const spans = [
    ...root.querySelectorAll<HTMLElement>(`span[data-type="${CITATION_NODE}"]`),
  ].filter((span) => !span.closest(`[data-type="${FOOTNOTE_NODE}"]`));
  if (spans.length === 0 || spans.length !== citations.length) return;
  const document = spans[0]?.ownerDocument ?? window.document;
  const list = document.createElement("div");
  list.setAttribute("style", "mso-element:footnote-list");
  list.setAttribute(NOTES_ATTRIBUTE, "");
  spans.forEach((span, index) => {
    const number = index + 1;
    const citation = citations[index] ?? {};
    span.replaceChildren(marker(document, number, "reference"));
    const { text, link } = sourceText(citation, context);
    const note = document.createElement("div");
    note.setAttribute("style", "mso-element:footnote");
    note.setAttribute("id", `ftn${number}`);
    const paragraph = document.createElement("p");
    paragraph.setAttribute("class", "MsoFootnoteText");
    paragraph.append(marker(document, number, "note"), " ", ...noteText(document, text, link));
    note.append(paragraph);
    list.append(note);
  });
  root.append(list);
}

/**
 * The text of a copied fragment with its Citations as numbered markers, "[1]", and the list of
 * Sources after it. "" when it holds no Citation, so the editor's own text stands.
 */
export function textWithSources(
  fragment: Fragment,
  schema: Schema,
  context: SourceContext,
): string {
  const citations = citationsIn(fragment);
  if (citations.length === 0) return "";
  let number = 0;
  const textSerializers: Record<string, TextSerializer> = {
    ...getTextSerializersFromSchema(schema),
    [CITATION_NODE]: () => `[${++number}]`,
  };
  const copied = schema.topNodeType.create(null, fragment);
  const body = getTextBetween(
    copied,
    { from: 0, to: copied.content.size },
    { textSerializers },
  ).trim();
  const sources = citations
    .map((citation, index) => `[${index + 1}] ${sourceText(citation, context).text}`)
    .join("\n");
  return `${body}\n\n${translate(context.language, "citedSource.title")}\n${sources}`;
}
