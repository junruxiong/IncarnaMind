import { useEffect, useState } from "react";
import type { ConversationAnswer, MarkedCitation, MeetingHelpKind } from "../../../../core/api";
import { citedLocation } from "../../../../shared/citations";
import { isTranscriptLocation } from "../../../../shared/locations";
import { helpLabel } from "../../../../shared/meetingHelp";
import { core } from "../../core";
import { type ReplyItem, type ReplyPiece, readHelpReply } from "../../helpReply";
import { useT } from "../../i18n";
import { CheckLineIcon, DocumentLineIcon } from "../lineIcons";

/** A help Answer (#839), as the core writes it: loaded once, then each write as it streams. */
export function useHelpAnswer(
  help: { conversationId: string; answerId: string } | null,
): ConversationAnswer | null {
  const [answer, setAnswer] = useState<ConversationAnswer | null>(null);
  const conversationId = help?.conversationId;
  const answerId = help?.answerId;
  useEffect(() => {
    setAnswer(null);
    if (!conversationId || !answerId) return;
    let live = true;
    const take = (message: unknown) => {
      const found = message as ConversationAnswer;
      if (live && found.kind === "answer" && found.id === answerId) setAnswer(found);
    };
    const stop = core.on("conversation.message", take);
    core.getConversation(conversationId).then(
      ({ messages }) => {
        const found = messages.find((message) => message.id === answerId);
        if (found) setAnswer((now) => now ?? (found as ConversationAnswer));
      },
      // Deleted meanwhile: the panel shows no suggestion.
      () => undefined,
    );
    return () => {
      live = false;
      stop();
    };
  }, [conversationId, answerId]);
  return answer;
}

/** A Citation's marker as a chip: its number, and a tick once its quote is found. */
function CitationChip({ citation, marker }: { citation?: MarkedCitation; marker: number }) {
  const t = useT();
  const where = citation ? citedLocation(citation, t) : null;
  const found = citation?.check === "found";
  return (
    <span
      data-testid="help-citation"
      data-check={citation?.check ?? "checking"}
      title={citation ? [citation.documentName, where].filter(Boolean).join(", ") : undefined}
      className="help-citation"
    >
      {marker}
      {found && <CheckLineIcon aria-hidden="true" className="help-citation-found" />}
    </span>
  );
}

function Pieces({ pieces, answer }: { pieces: readonly ReplyPiece[]; answer: ConversationAnswer }) {
  return (
    <>
      {pieces.map((piece, index) => {
        const key = `${piece.kind}-${index}`;
        if (piece.kind === "text") return <span key={key}>{piece.text}</span>;
        if (piece.kind === "figure") {
          const figure = answer.figures.find((each) => each.marker === piece.marker);
          return <span key={key}>{figure?.shown ?? ""}</span>;
        }
        const citation = answer.citations.find((each) => each.marker === piece.marker);
        return <CitationChip key={key} citation={citation} marker={piece.marker} />;
      })}
    </>
  );
}

function Item({ item, answer }: { item: ReplyItem; answer: ConversationAnswer }) {
  const t = useT();
  return (
    <div
      data-testid="help-option"
      data-recommended={item.recommended}
      className={`help-option${item.recommended ? " help-option--recommended" : ""}`}
    >
      <span className="help-option-mark">
        {item.recommended && (
          <span role="img" aria-label={t("meetings.help.recommended")}>
            ★
          </span>
        )}
      </span>
      <span data-testid="help-option-say" className="help-option-say">
        <Pieces pieces={item.lead} answer={answer} />
      </span>
      {item.notes.map((note, index) => (
        // biome-ignore lint/suspicious/noArrayIndexKey: a reading's notes are in the order written.
        <span key={index} data-testid="help-option-note" className="help-option-note">
          <Pieces pieces={note} answer={answer} />
        </span>
      ))}
      {item.effect && (
        <span data-testid="help-option-effect" className="help-option-effect">
          → <Pieces pieces={item.effect} answer={answer} />
        </span>
      )}
    </div>
  );
}

/** The outline of options still to come: shown within 100 ms of a press. */
export function HelpOutline({ count }: { count: number }) {
  return (
    <>
      {Array.from({ length: count }, (_, index) => (
        // biome-ignore lint/suspicious/noArrayIndexKey: placeholders, all alike.
        <div key={index} data-testid="help-outline" aria-hidden="true" className="help-option">
          <span className="help-option-mark" />
          <span className="help-outline-lines">
            <span className="help-outline-line" style={{ width: "88%" }} />
            <span className="help-outline-line" style={{ width: "52%" }} />
          </span>
        </div>
      ))}
    </>
  );
}

/** The heading over a suggestion: the button, and for "What do they want?" what it reads from. */
export function HelpHeading({ kind }: { kind: MeetingHelpKind }) {
  const t = useT();
  return (
    <p data-testid="help-heading" className="help-heading">
      <b>{t(helpLabel(kind))}</b>
      {kind === "want" && <span> · {t("meetings.help.readFrom")}</span>}
    </p>
  );
}

/** The Documents and transcript moments a suggestion cites, one line each. */
function Sources({ answer, markers }: { answer: ConversationAnswer; markers: readonly number[] }) {
  const t = useT();
  const cited = markers.flatMap((marker) => {
    const citation = answer.citations.find((each) => each.marker === marker);
    return citation ? [citation] : [];
  });
  if (cited.length === 0) return null;
  return (
    <ul data-testid="help-sources" className="help-sources">
      {cited.map((citation) => {
        const where = citedLocation(citation, t);
        return (
          <li key={citation.marker} data-testid="help-source" className="help-source">
            <DocumentLineIcon
              kind={isTranscriptLocation(citation.location) ? "transcript" : "text"}
              className="size-[13px] shrink-0"
            />
            <span className="truncate">
              {[String(citation.marker), citation.documentName, where].filter(Boolean).join(" · ")}
            </span>
          </li>
        );
      })}
    </ul>
  );
}

/**
 * A help button's Answer, laid out to be read in a second or two: the options with the
 * recommended first and starred, each with what it leads to; the Citations they rest on.
 * While it streams, the options still to come keep their outline.
 */
export function HelpSuggestion({
  kind,
  answer,
}: {
  kind: MeetingHelpKind;
  answer: ConversationAnswer;
}) {
  const t = useT();
  const items = readHelpReply(answer.text);
  const streaming = answer.status === "streaming";
  const markers = [
    ...new Set(
      items.flatMap((item) =>
        [item.lead, ...item.notes, item.effect ?? []].flatMap((pieces) =>
          pieces.flatMap((piece) => (piece.kind === "citation" ? [piece.marker] : [])),
        ),
      ),
    ),
  ];
  return (
    <div
      data-testid="help-suggestion"
      data-status={answer.status}
      aria-busy={streaming}
      className="help-suggestion"
    >
      <HelpHeading kind={kind} />
      {items.map((item, index) => (
        // biome-ignore lint/suspicious/noArrayIndexKey: the options are in the order written.
        <Item key={index} item={item} answer={answer} />
      ))}
      {streaming && items.length < 2 && <HelpOutline count={items.length === 0 ? 2 : 1} />}
      {answer.status === "failed" && (
        <p role="alert" data-testid="help-failed" className="help-failed">
          {t("meetings.help.failed", { message: answer.errorMessage ?? "" })}
        </p>
      )}
      <Sources answer={answer} markers={markers} />
    </div>
  );
}
