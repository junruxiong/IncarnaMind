import type { NodeViewRenderer } from "@tiptap/core";
import { TextSelection } from "@tiptap/pm/state";
import { type Editor, useEditorState } from "@tiptap/react";
import { useRef } from "react";
import type { CitationAttributes } from "../../../core/api";
import { citationState } from "../../../shared/citations";
import { citationLocation, formatLocation } from "../../../shared/locations";
import { noteOfCitation, useMindFormatting } from "../citationFormatting";
import { useT } from "../i18n";
import { useAppStore } from "../store";
import { numberedCitations, numberedFootnotes } from "./citationNumbers";
import { FootnoteText } from "./FootnoteText";
import { focusNow, returnToFootnote } from "./footnoteSchema";
import { CheckMarkIcon } from "./icons";
import { useMindId } from "./mindContext";
import { StyledRuns } from "./StyledRuns";

/** One footnote of the area, in the Mind's one numbering. */
type Row =
  | { kind: "footnote"; number: number; id: string }
  | { kind: "citation"; number: number; key: string; attributes: CitationAttributes };

/** The Mind's footnotes in order, or none while the User has written no footnote of their own. */
function rowsOf(editor: Editor): Row[] {
  const footnotes = numberedFootnotes(editor.state);
  if (footnotes.length === 0) return [];
  const rows: Row[] = [
    ...footnotes.map(({ number, id }): Row => ({ kind: "footnote", number, id })),
    // Only the Citations the Mind's citation style makes footnotes of (#289).
    ...numberedCitations(editor.state)
      .filter((citation) => citation.footnote)
      .map(({ number, key, attributes }): Row => ({ kind: "citation", number, key, attributes })),
  ];
  return rows.sort((a, b) => a.number - b.number);
}

const sameRows = (a: readonly Row[], b: readonly Row[] | null) =>
  b !== null &&
  a.length === b.length &&
  a.every((row, index) => {
    const other = b[index];
    if (!other || row.kind !== other.kind || row.number !== other.number) return false;
    return row.kind === "footnote"
      ? other.kind === "footnote" && row.id === other.id
      : other.kind === "citation" && row.key === other.key && row.attributes === other.attributes;
  });

/**
 * The footnote area at the end of the Mind, once the User has written a
 * footnote: every footnote in the Mind's one numbering, as the export will
 * have them. The User's own are edited in place (`FootnoteText`); a
 * Citation's is its reference and its check, quieter. A footnote's number
 * takes the cursor to its marker in the text.
 */
export function FootnoteArea({
  editor,
  citationView,
}: {
  editor: Editor;
  citationView: NodeViewRenderer | null;
}) {
  const t = useT();
  const rows = useEditorState({
    editor,
    selector: ({ editor: current }) => rowsOf(current),
    equalityFn: sameRows,
  });
  const numbers = useRef<Map<string, HTMLButtonElement>>(new Map());
  if (rows.length === 0) return null;

  return (
    <section
      className="footnote-area"
      data-testid="footnote-area"
      aria-label={t("editor.footnotes.title")}
    >
      <h2 className="footnote-area-title">{t("editor.footnotes.title")}</h2>
      <ol className="footnote-area-list">
        {rows.map((row) =>
          row.kind === "footnote" ? (
            <li
              key={row.id}
              className="footnote-area-row"
              data-testid="footnote-row"
              data-number={row.number}
              data-footnote-id={row.id}
            >
              <button
                ref={(element) => {
                  if (element) numbers.current.set(row.id, element);
                  else numbers.current.delete(row.id);
                }}
                type="button"
                className="footnote-area-number"
                data-testid="footnote-row-number"
                aria-label={t("editor.footnotes.goTo", { number: row.number })}
                title={t("editor.footnotes.goTo", { number: row.number })}
                onClick={() => returnToFootnote(editor, row.id)}
              >
                {row.number}
              </button>
              <FootnoteText
                mind={editor}
                id={row.id}
                label={t("editor.footnote.text", { number: row.number })}
                placeholder={t("editor.footnote.placeholder")}
                variant="area"
                citationView={citationView}
                // Back on its number, from where Tab goes on and Enter goes to its marker.
                onLeave={() => numbers.current.get(row.id)?.focus()}
              />
            </li>
          ) : (
            <CitationRow key={row.key} editor={editor} row={row} />
          ),
        )}
      </ol>
    </section>
  );
}

/** A Citation's footnote in the area: its number, which goes to its marker, and its reference. */
function CitationRow({ editor, row }: { editor: Editor; row: Extract<Row, { kind: "citation" }> }) {
  const t = useT();
  const documents = useAppStore((state) =>
    state.status.kind === "ready" ? state.documents : null,
  );
  const kept = useAppStore((state) => state.keptCitationTexts);
  const { attributes } = row;
  const state = citationState(attributes, documents, kept);
  const location = citationLocation(attributes);
  const where = location ? formatLocation(location, t) : null;
  const name = attributes.documentName ?? "";
  const source = where ? t("citation.where", { document: name, location: where }) : name;
  // In a note style, the note as the style writes it (#289), once it is formatted.
  const formatting = useMindFormatting(useMindId());
  const note = noteOfCitation(editor.state, formatting, row.key);
  return (
    <li
      className="footnote-area-row footnote-area-row--citation"
      data-testid="footnote-row-citation"
      data-number={row.number}
      data-check={state.check}
    >
      <button
        type="button"
        className="footnote-area-number"
        aria-label={t("editor.footnotes.goToCitation", { number: row.number })}
        title={t("editor.footnotes.goToCitation", { number: row.number })}
        onClick={() => {
          // Where it is now: the row may be drawn from before the last keystroke.
          const citation = numberedCitations(editor.state).find((each) => each.key === row.key);
          if (!citation) return;
          const after = TextSelection.create(editor.state.doc, citation.pos + 1);
          editor.view.dispatch(editor.state.tr.setSelection(after).scrollIntoView());
          focusNow(editor);
        }}
      >
        {row.number}
      </button>
      <span className="footnote-area-source">
        <CheckMarkIcon check={state.check} className="footnote-area-check" />
        {note ? <StyledRuns runs={note} /> : source}
      </span>
    </li>
  );
}
