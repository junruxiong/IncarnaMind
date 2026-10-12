import type { KeyboardEvent } from "react";
import { useCallback, useEffect, useId, useLayoutEffect, useMemo, useRef, useState } from "react";
import type { Document } from "../../../core/api";
import {
  type Cue,
  readVtt,
  type Transcript,
  type TranscriptGap,
  transcriptUnits,
} from "../../../core/documents/formats/vtt";
import { clockTime } from "../../../shared/clock";
import type { MessageKey } from "../../../shared/i18n";
import { transcriptHandle, useHandle } from "../commands/handles";
import { PlusLineIcon } from "../components/lineIcons";
import { useDismissOnEscape } from "../escapeDismiss";
import { useT } from "../i18n";
import { useMeetings } from "../meetings";
import { durationLabel } from "../meetingTime";
import type { ViewerTarget } from "../store";
import { addLineToNote, type LineAdded } from "../transcriptNotes";
import { loadDocumentBytes, MissingDocumentError } from "./documentBytes";
import { markQuote, textNodes, unmarkQuote } from "./domQuote";
import { FileStates } from "./FileStates";
import { CitationMark, marginRoom, markTopFor, quoteMarkOf, quoteTone } from "./quoteMark";
import type { LoadedFile } from "./useDocumentFile";
import { ViewerHeader } from "./ViewerHeader";

/** Within this of the bottom, the view follows new lines as they come. */
const FOLLOW_PX = 48;

const GAP_REASONS: Record<TranscriptGap["reason"], MessageKey> = {
  restarted: "transcript.gap.restarted",
  sleep: "transcript.gap.sleep",
  interrupted: "transcript.gap.interrupted",
  lost: "transcript.gap.lost",
};

const read = (bytes: Uint8Array): Transcript => readVtt(new TextDecoder().decode(bytes));

/**
 * The Transcript's file, read again whenever `version` changes, the last read
 * kept on screen meanwhile: a Meeting's lines arrive every few seconds.
 */
function useTranscriptFile(documentId: string, version: string): LoadedFile<Transcript> {
  const [loaded, setLoaded] = useState<LoadedFile<Transcript>>({ kind: "loading" });
  // biome-ignore lint/correctness/useExhaustiveDependencies: `version` says when to read again
  useEffect(() => {
    const controller = new AbortController();
    loadDocumentBytes(documentId, controller.signal)
      .then((bytes) => {
        if (!controller.signal.aborted) setLoaded({ kind: "ready", value: read(bytes) });
      })
      .catch((error: unknown) => {
        if (controller.signal.aborted) return;
        if (error instanceof MissingDocumentError) setLoaded({ kind: "missing" });
        else
          setLoaded({
            kind: "failed",
            message: error instanceof Error ? error.message : String(error),
          });
      });
    return () => controller.abort();
  }, [documentId, version]);
  return loaded;
}

/**
 * A Transcript (#136): its lines, each with its time, and its speaker once
 * speakers are told apart (#140). While its Meeting records, lines arrive a
 * few seconds behind, the speech being transcribed shows greyed at the end,
 * and the view follows them unless the User has scrolled up. Opened at a
 * Citation, the quote is washed in the line it was said in.
 */
export function TranscriptView({ document, target }: { document: Document; target: ViewerTarget }) {
  const meeting = useMeetings((state) =>
    state.meetings.find((each) => each.transcript?.documentId === document.id),
  );
  const loaded = useTranscriptFile(
    document.id,
    `${document.contentHash}:${meeting?.transcript?.lines ?? 0}`,
  );
  return (
    <FileStates
      loaded={loaded}
      quote={target.quote}
      ready={(transcript) => (
        <TranscriptContent
          document={document}
          transcript={transcript}
          target={target}
          meetingId={meeting?.id ?? null}
        />
      )}
    />
  );
}

type Row = { kind: "cue"; cue: Cue } | { kind: "gap"; gap: TranscriptGap };

/** How long "Added to the note" shows in the header. */
const ADDED_SHOWN_MS = 3000;

/** The line after or before this one: the Transcript's gaps between them aren't lines. */
function lineBeside(from: Element, direction: "next" | "previous"): HTMLElement | null {
  let each = direction === "next" ? from.nextElementSibling : from.previousElementSibling;
  while (each) {
    if (each instanceof HTMLElement && each.dataset.start !== undefined) return each;
    each = direction === "next" ? each.nextElementSibling : each.previousElementSibling;
  }
  return null;
}

function TranscriptContent({
  document,
  transcript,
  target,
  meetingId,
}: {
  document: Document;
  transcript: Transcript;
  target: ViewerTarget;
  meetingId: string | null;
}) {
  const t = useT();
  const meeting = useMeetings((state) => state.meetings.find((each) => each.id === meetingId));
  const live = meeting?.status === "recording";
  const state = meeting?.transcript?.state;
  const partial = live && state === "live" ? (meeting?.transcript?.partial ?? null) : null;
  const scroller = useRef<HTMLDivElement>(null);
  const column = useRef<HTMLDivElement>(null);
  const list = useRef<HTMLOListElement>(null);
  const following = useRef(true);
  const [atEnd, setAtEnd] = useState(true);
  const [highlighted, setHighlighted] = useState(0);
  const [markTop, setMarkTop] = useState<number | null>(null);
  const [markRoom, setMarkRoom] = useState(0);
  const mark = quoteMarkOf(target.citation);
  const hintId = useId();
  // The line the User is on, by where it starts: the one Enter, the button and ⌘K add.
  const [active, setActive] = useState<number | null>(null);
  const activeRef = useRef<Cue | null>(null);
  const [added, setAdded] = useState<LineAdded | null>(null);

  const rows = useMemo<Row[]>(
    () =>
      [
        ...transcript.cues.map((cue): Row & { at: number } => ({
          kind: "cue",
          cue,
          at: cue.startMs,
        })),
        ...transcript.gaps.map((gap): Row & { at: number } => ({
          kind: "gap",
          gap,
          at: gap.fromMs,
        })),
      ].sort((a, b) => a.at - b.at),
    [transcript],
  );
  const units = useMemo(() => transcriptUnits(transcript), [transcript]);

  const addLine = useCallback(
    async (cue: Cue) => {
      if (!meetingId) return;
      const unit = units.find(
        ({ label }) =>
          label?.from !== undefined &&
          label.to !== undefined &&
          cue.startMs >= label.from &&
          cue.startMs <= label.to,
      );
      setAdded(await addLineToNote({ document, meetingId, cue, page: unit?.page ?? null }));
    },
    [document, meetingId, units],
  );
  useEffect(() => {
    if (added === null || added === "failed") return;
    const timer = setTimeout(() => setAdded(null), ADDED_SHOWN_MS);
    return () => clearTimeout(timer);
  }, [added]);
  useDismissOnEscape(added !== null && added !== "failed", () => setAdded(null));
  const lend = useMemo(
    () => ({
      hasLine: () => meetingId !== null && activeRef.current !== null,
      addLine: async () => {
        if (activeRef.current) await addLine(activeRef.current);
      },
    }),
    [addLine, meetingId],
  );
  useHandle(transcriptHandle, lend);
  const firstStart = transcript.cues[0]?.startMs;

  const onLineKey = (event: KeyboardEvent<HTMLElement>, cue: Cue) => {
    if (event.target !== event.currentTarget) return;
    if (event.key === "Enter") {
      event.preventDefault();
      void addLine(cue);
    } else if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      const next = lineBeside(event.currentTarget, event.key === "ArrowDown" ? "next" : "previous");
      if (!next) return;
      event.preventDefault();
      next.focus();
      next.scrollIntoView({ block: "nearest" });
    }
  };

  // Opened at a Citation: the quote washed in the cited lines, else anywhere; the view goes there,
  // once for each open request: lines arriving afterwards don't pull the view back.
  const shownRequest = useRef<string | null>(null);
  useLayoutEffect(() => {
    const body = list.current;
    const container = scroller.current;
    const request = `${target.request}:${target.quote}:${target.pageFrom}:${target.pageTo}`;
    if (!body || !container || !target.quote || shownRequest.current === request) return;
    if (units.length === 0) return;
    shownRequest.current = request;
    unmarkQuote(body);
    const cited = units.filter(
      (unit) =>
        target.pageFrom !== undefined &&
        unit.page >= target.pageFrom &&
        unit.page <= (target.pageTo ?? target.pageFrom),
    );
    const from = cited[0]?.label?.from;
    const to = cited.at(-1)?.label?.to;
    const lines = [...body.querySelectorAll<HTMLElement>("[data-start]")].filter((line) => {
      const start = Number(line.dataset.start);
      return from !== undefined && to !== undefined && start >= from && start <= to;
    });
    let range: Range | undefined;
    const first = lines[0];
    const last = lines.at(-1);
    if (first && last) {
      range = body.ownerDocument.createRange();
      range.setStartBefore(first);
      range.setEndAfter(last);
    }
    const marks =
      (range && markQuote(textNodes(body, range), target.quote)) ??
      markQuote(textNodes(body), target.quote);
    setHighlighted((count) => (marks ? count + 1 : 0));
    const anchor = marks?.[0] ?? first ?? null;
    if (anchor) {
      following.current = false;
      const top =
        anchor.getBoundingClientRect().top -
        container.getBoundingClientRect().top +
        container.scrollTop;
      container.scrollTop = Math.max(0, top - container.clientHeight / 3);
    }
  }, [units, target.request, target.quote, target.pageFrom, target.pageTo]);

  // New lines while live: followed, unless the User has scrolled up to read.
  // biome-ignore lint/correctness/useExhaustiveDependencies: placed again for each line
  useLayoutEffect(() => {
    const container = scroller.current;
    if (container && following.current) container.scrollTop = container.scrollHeight;
  }, [rows.length, partial?.text]);

  useLayoutEffect(() => {
    const element = column.current;
    if (!element || !mark || highlighted === 0) {
      setMarkTop(null);
      return;
    }
    const place = () => {
      const first = element.querySelector("[data-quote-highlight]");
      setMarkTop(first ? markTopFor(first, element) : null);
      setMarkRoom(marginRoom(element));
    };
    place();
    const observer = new ResizeObserver(place);
    observer.observe(element);
    return () => observer.disconnect();
  }, [mark, highlighted]);

  const onScroll = () => {
    const container = scroller.current;
    if (!container) return;
    const end = container.scrollHeight - container.scrollTop - container.clientHeight <= FOLLOW_PX;
    following.current = end;
    setAtEnd(end);
  };

  const toEnd = () => {
    const container = scroller.current;
    if (!container) return;
    following.current = true;
    setAtEnd(true);
    container.scrollTop = container.scrollHeight;
  };

  return (
    <>
      <ViewerHeader>
        {live && state === "live" && (
          <span data-testid="transcript-live" className="transcript-live">
            <span aria-hidden="true" className="recording-dot" />
            {t("transcript.live")}
          </span>
        )}
        {live && state === "behind" && (
          <span data-testid="transcript-behind" className="transcript-live">
            {t("transcript.recording")}
          </span>
        )}
        {added && added !== "failed" && (
          <span role="status" data-testid="transcript-added" className="viewer-meta">
            {t(added === "added" ? "transcript.addToNote.added" : "transcript.addToNote.already")}
          </span>
        )}
        <span data-testid="transcript-count" className="viewer-meta">
          {t(transcript.cues.length === 1 ? "transcript.lines.one" : "transcript.lines.other", {
            count: transcript.cues.length,
          })}
        </span>
      </ViewerHeader>
      <div className="relative flex min-h-0 flex-1">
        <div
          ref={scroller}
          data-testid="viewer-transcript"
          data-quote-tone={quoteTone(target.citation)}
          onScroll={onScroll}
          className="viewer-backdrop viewer-text-backdrop min-h-0 min-w-0 flex-1 overflow-auto select-text"
        >
          <article className="viewer-page viewer-text-page transcript-page">
            <div ref={column} className="relative">
              {rows.length === 0 && !partial && (
                <p data-testid="transcript-empty" className="transcript-empty">
                  {t(live ? "transcript.waiting" : "transcript.none")}
                </p>
              )}
              {meetingId && (
                <p id={hintId} className="sr-only">
                  {t("transcript.line.hint")}
                </p>
              )}
              <ol ref={list} data-testid="transcript-lines" className="transcript-lines">
                {rows.map((row) =>
                  row.kind === "cue" ? (
                    <li
                      key={`c${row.cue.startMs}-${row.cue.endMs}`}
                      data-testid="transcript-line"
                      data-start={row.cue.startMs}
                      tabIndex={
                        meetingId
                          ? (active ?? firstStart) === row.cue.startMs
                            ? 0
                            : -1
                          : undefined
                      }
                      aria-describedby={meetingId ? hintId : undefined}
                      onFocus={() => {
                        setActive(row.cue.startMs);
                        activeRef.current = row.cue;
                      }}
                      onKeyDown={meetingId ? (event) => onLineKey(event, row.cue) : undefined}
                      className="transcript-line group"
                    >
                      <span data-quote-skip="" className="transcript-time">
                        {clockTime(row.cue.startMs)}
                      </span>
                      <div className="min-w-0">
                        {row.cue.speaker && (
                          <p data-quote-skip="" className="transcript-speaker">
                            {row.cue.speaker}
                          </p>
                        )}
                        <p className="transcript-text">{row.cue.text}</p>
                      </div>
                      {meetingId && (
                        <button
                          type="button"
                          tabIndex={-1}
                          data-testid="transcript-add"
                          data-quote-skip=""
                          aria-label={t("transcript.addToNote.line", {
                            time: clockTime(row.cue.startMs),
                            text: row.cue.text,
                          })}
                          onClick={() => void addLine(row.cue)}
                          className="transcript-add"
                        >
                          <PlusLineIcon className="size-3.5" />
                          {t("transcript.addToNote")}
                        </button>
                      )}
                    </li>
                  ) : (
                    <li
                      key={`g${row.gap.fromMs}-${row.gap.toMs}`}
                      data-testid="transcript-gap"
                      data-quote-skip=""
                      className="transcript-gap"
                    >
                      <span className="transcript-time">{clockTime(row.gap.fromMs)}</span>
                      <span>
                        {row.gap.toMs > row.gap.fromMs
                          ? t("transcript.gap", {
                              duration: durationLabel(row.gap.toMs - row.gap.fromMs, t),
                              reason: t(GAP_REASONS[row.gap.reason]),
                            })
                          : t(GAP_REASONS[row.gap.reason])}
                      </span>
                    </li>
                  ),
                )}
                {partial && (
                  <li
                    data-testid="transcript-partial"
                    data-quote-skip=""
                    aria-live="polite"
                    className="transcript-line transcript-line--partial"
                  >
                    <span className="transcript-time">{clockTime(partial.startMs)}</span>
                    <p className="transcript-text">{partial.text}</p>
                  </li>
                )}
              </ol>
              {mark && markTop !== null && (
                <CitationMark
                  mark={mark}
                  room={markRoom}
                  style={{ top: markTop, left: "calc(100% + 12px)" }}
                />
              )}
            </div>
          </article>
        </div>
        {live && !atEnd && (
          <button
            type="button"
            data-testid="transcript-follow"
            onClick={toEnd}
            className="transcript-follow"
          >
            {t("transcript.follow")}
          </button>
        )}
      </div>
    </>
  );
}
