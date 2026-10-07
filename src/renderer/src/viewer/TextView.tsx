import { type ReactNode, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import type { Document } from "../../../core/api";
import { decodeText } from "../../../core/documents/decode";
import { findQuote, type TextRange } from "../../../shared/quoteMatch";
import { useT } from "../i18n";
import type { ViewerTarget } from "../store";
import { loadDocumentBytes, MissingDocumentError } from "./documentBytes";
import { type Block, type Inline, parseMarkdown } from "./markdown";
import { CitationMark, markTopFor, quoteMarkOf, quoteTone } from "./quoteMark";
import { ViewerHeader } from "./ViewerHeader";
import { DocumentRemoved, ViewerMessage } from "./ViewerMessage";

type Loaded =
  | { kind: "loading" }
  | { kind: "ready"; text: string }
  | { kind: "missing" }
  | { kind: "failed"; message: string };

/** Nothing to highlight. */
const NO_HIGHLIGHT: readonly TextRange[] = [];

/**
 * A TXT or Markdown Document, readable in the viewer: set in the Mind's serif
 * on a page, like a PDF's. Opened with a quote, it highlights the quote (each
 * part of a quote with an ellipsis) and scrolls to it; otherwise it opens at
 * the top.
 */
export function TextView({ document, target }: { document: Document; target: ViewerTarget }) {
  const t = useT();
  const [loaded, setLoaded] = useState<Loaded>({ kind: "loading" });

  useEffect(() => {
    const controller = new AbortController();
    loadDocumentBytes(document.id, controller.signal).then(
      (bytes) => {
        // Decoded the way processing decodes it, so quotes from its Passages match.
        const text = decodeText(bytes);
        setLoaded(text === null ? { kind: "failed", message: "" } : { kind: "ready", text });
      },
      (error: unknown) => {
        if (controller.signal.aborted) return;
        if (error instanceof MissingDocumentError) setLoaded({ kind: "missing" });
        else setLoaded({ kind: "failed", message: error instanceof Error ? error.message : "" });
      },
    );
    return () => controller.abort();
  }, [document.id]);

  return (
    <>
      <ViewerHeader openable={loaded.kind !== "missing"} />
      {loaded.kind === "loading" ? (
        <ViewerMessage>{t("viewer.loading")}</ViewerMessage>
      ) : loaded.kind === "missing" ? (
        <DocumentRemoved quote={target.quote} />
      ) : loaded.kind === "failed" ? (
        <ViewerMessage tone="error">
          {loaded.message ? t("viewer.failed", { message: loaded.message }) : t("viewer.notText")}
        </ViewerMessage>
      ) : (
        <TextContent text={loaded.text} markdown={document.kind === "markdown"} target={target} />
      )}
    </>
  );
}

function TextContent({
  text,
  markdown,
  target,
}: {
  text: string;
  markdown: boolean;
  target: ViewerTarget;
}) {
  const scroller = useRef<HTMLDivElement>(null);
  const column = useRef<HTMLDivElement>(null);
  const appliedRequest = useRef<number | null>(null);
  const highlight = useMemo(
    () => (target.quote ? findQuote(text, target.quote) : null) ?? NO_HIGHLIGHT,
    [text, target.quote],
  );
  const blocks = useMemo(() => (markdown ? parseMarkdown(text) : null), [text, markdown]);
  const mark = quoteMarkOf(target.citation);
  const [markTop, setMarkTop] = useState<number | null>(null);

  // The Citation's mark sits beside the quote's first line, wherever the text wraps it.
  const showMark = mark !== null && highlight.length > 0;
  // biome-ignore lint/correctness/useExhaustiveDependencies: re-placed for each new highlight
  useLayoutEffect(() => {
    const element = column.current;
    if (!element || !showMark) {
      setMarkTop(null);
      return;
    }
    const place = () => {
      const first = element.querySelector("[data-quote-highlight]");
      setMarkTop(first ? markTopFor(first, element) : null);
    };
    place();
    const observer = new ResizeObserver(place);
    observer.observe(element);
    return () => observer.disconnect();
  }, [showMark, highlight]);

  // Each open request goes to its quote, or to the top if it has none or it isn't found.
  // Opening the Document again without a quote leaves the reader where they were.
  useLayoutEffect(() => {
    const container = scroller.current;
    if (!container || appliedRequest.current === target.request) return;
    const first = appliedRequest.current === null;
    appliedRequest.current = target.request;
    const mark = container.querySelector<HTMLElement>("[data-quote-highlight]");
    if (mark) {
      const top =
        mark.getBoundingClientRect().top -
        container.getBoundingClientRect().top +
        container.scrollTop;
      container.scrollTop = Math.max(0, top - container.clientHeight / 3);
    } else if (target.quote || first) {
      container.scrollTop = 0;
    }
  }, [target.request, target.quote]);

  return (
    <div
      ref={scroller}
      data-testid="viewer-text"
      data-quote-tone={quoteTone(target.citation)}
      className="viewer-backdrop viewer-text-backdrop min-h-0 flex-1 overflow-auto select-text"
    >
      <article className="viewer-page viewer-text-page">
        <div ref={column} className="relative">
          {blocks ? (
            <div className="document-markdown">
              {blocks.map((block, index) => (
                // biome-ignore lint/suspicious/noArrayIndexKey: the blocks never reorder
                <MarkdownBlock key={index} block={block} source={text} highlight={highlight} />
              ))}
            </div>
          ) : (
            <pre className="document-plain">
              <Slice source={text} start={0} end={text.length} highlight={highlight} />
            </pre>
          )}
          {mark && markTop !== null && (
            <CitationMark mark={mark} style={{ top: markTop, left: "calc(100% + 12px)" }} />
          )}
        </div>
      </article>
    </div>
  );
}

/** A span of the source, with the parts inside `highlight` (ranges in order) marked. */
function Slice({
  source,
  start,
  end,
  highlight,
}: {
  source: string;
  start: number;
  end: number;
  highlight: readonly TextRange[];
}) {
  const inside = highlight.filter((range) => range.end > start && range.start < end);
  if (inside.length === 0) return source.slice(start, end);
  const parts: ReactNode[] = [];
  let at = start;
  for (const range of inside) {
    const from = Math.max(at, range.start);
    const to = Math.min(end, range.end);
    if (from >= to) continue;
    parts.push(source.slice(at, from));
    // A part split across inlines (e.g. at a line break) joins up square at the seam.
    const joins = `${from > range.start ? " quote-highlight--joins-before" : ""}${
      to < range.end ? " quote-highlight--joins-after" : ""
    }`;
    parts.push(
      <mark key={from} data-quote-highlight="" className={`quote-highlight${joins}`}>
        {source.slice(from, to)}
      </mark>,
    );
    at = to;
  }
  parts.push(source.slice(at, end));
  return <>{parts}</>;
}

function renderInlines(
  inlines: readonly Inline[],
  source: string,
  highlight: readonly TextRange[],
): ReactNode[] {
  return inlines.map((inline, index) => renderInline(inline, index, source, highlight));
}

function renderInline(
  inline: Inline,
  key: number,
  source: string,
  highlight: readonly TextRange[],
): ReactNode {
  switch (inline.kind) {
    case "text":
      return (
        <Slice
          key={key}
          source={source}
          start={inline.start}
          end={inline.end}
          highlight={highlight}
        />
      );
    case "code":
      return (
        <code key={key}>
          <Slice source={source} start={inline.start} end={inline.end} highlight={highlight} />
        </code>
      );
    case "strong":
      return <strong key={key}>{renderInlines(inline.children, source, highlight)}</strong>;
    case "em":
      return <em key={key}>{renderInlines(inline.children, source, highlight)}</em>;
    case "link":
      return inline.href ? (
        // Opens in the User's browser: the main process sends every new window there.
        <a key={key} href={inline.href} target="_blank" rel="noreferrer">
          {renderInlines(inline.children, source, highlight)}
        </a>
      ) : (
        <span key={key}>{renderInlines(inline.children, source, highlight)}</span>
      );
  }
}

function MarkdownBlock({
  block,
  source,
  highlight,
}: {
  block: Block;
  source: string;
  highlight: readonly TextRange[];
}) {
  switch (block.kind) {
    case "heading": {
      const Heading = `h${block.level}` as const;
      return <Heading>{renderInlines(block.inlines, source, highlight)}</Heading>;
    }
    case "paragraph":
      return <p>{renderInlines(block.inlines, source, highlight)}</p>;
    case "quote":
      return <blockquote>{renderInlines(block.inlines, source, highlight)}</blockquote>;
    case "list": {
      const List = block.ordered ? "ol" : "ul";
      return (
        <List>
          {block.items.map((item, index) => (
            // biome-ignore lint/suspicious/noArrayIndexKey: the items never reorder
            <li key={index}>{renderInlines(item, source, highlight)}</li>
          ))}
        </List>
      );
    }
    case "verbatim":
      return (
        <pre>
          <Slice source={source} start={block.start} end={block.end} highlight={highlight} />
        </pre>
      );
    case "rule":
      return <hr />;
  }
}
