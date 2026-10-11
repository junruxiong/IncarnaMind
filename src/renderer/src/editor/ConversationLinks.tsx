import type { Editor } from "@tiptap/react";
import { useCallback, useEffect, useRef, useState } from "react";
import type { BlockOrigin, ReferenceTarget } from "../../../core/api";
import { ConversationLineIcon } from "../components/lineIcons";
import { conversationLabel, openConversation, useConversations } from "../conversations";
import { core } from "../core";
import { useT } from "../i18n";
import { useAppStore } from "../store";
import { hasOrigins, type OriginRun, originRuns } from "./blockOrigins";
import { Popover } from "./Popover";

/** How long the pointer rests on added text before its card shows, and may leave before it goes. */
const SHOW_MS = 300;
const HIDE_MS = 200;

interface Placed extends OriginRun {
  /** From the top of the frame, to its first line. */
  top: number;
}

/** The conversation an origin points into, as last listed: its name while it isn't deleted. */
const listedName = (origin: BlockOrigin) => {
  const conversation = useConversations
    .getState()
    .list.find((each) => each.id === origin.from.artifactId);
  return conversation ? conversationLabel(conversation) : null;
};

/** Opens the conversation a Note came from in the assistant, at its Answer, flashed. */
export function openOrigin(origin: BlockOrigin): void {
  if (origin.from.artifactKind !== "conversation") return;
  void openConversation(origin.from.artifactId, origin.from.anchor.messageId);
}

/**
 * The links back of Notes added from Answers (#201): a small conversation mark
 * in the left margin beside each run of Blocks added from one Answer, and on
 * hover (over the mark or the text) a card naming the conversation and the
 * Question, "From the conversation “…”", which opens it in the assistant at
 * that Answer. A conversation in Recently deleted, or deleted for good, is
 * said to be. Clicking the mark opens the conversation, or says why it can't.
 * Under 900px the margin folds away; hovering the text still shows the card.
 */
export function ConversationLinks({ editor }: { editor: Editor }) {
  const t = useT();
  const layer = useRef<HTMLDivElement>(null);
  const [runs, setRuns] = useState<readonly Placed[]>([]);
  /** The run whose card shows, and what it is anchored to. */
  const [card, setCard] = useState<{ run: OriginRun; anchor: Element } | null>(null);
  const timer = useRef<number | undefined>(undefined);
  // Listed conversations change their names and whether they are there.
  useConversations((state) => state.list);

  useEffect(() => {
    let frame = 0;
    const measure = () => {
      frame = 0;
      const column = layer.current;
      if (!column || editor.isDestroyed) return;
      const origin = column.getBoundingClientRect().top;
      const placed: Placed[] = [];
      for (const run of originRuns(editor.state)) {
        const dom = editor.view.nodeDOM(run.pos);
        if (!(dom instanceof HTMLElement)) continue;
        placed.push({ ...run, top: Math.round(dom.getBoundingClientRect().top - origin) });
      }
      setRuns((drawn) =>
        drawn.length === placed.length &&
        drawn.every((each, index) => {
          const other = placed[index];
          return (
            other?.blockId === each.blockId &&
            other.top === each.top &&
            other.origin === each.origin
          );
        })
          ? drawn
          : placed,
      );
    };
    const request = () => {
      if (!frame) frame = requestAnimationFrame(measure);
    };
    request();
    // Only a Mind with added text has anything to place: typing elsewhere costs nothing.
    const onTransaction = () => {
      if (hasOrigins(editor.state) || layer.current?.childElementCount) request();
    };
    editor.on("transaction", onTransaction);
    const resizes = new ResizeObserver(request);
    resizes.observe(editor.view.dom);
    return () => {
      cancelAnimationFrame(frame);
      editor.off("transaction", onTransaction);
      resizes.disconnect();
    };
  }, [editor]);

  const show = useCallback((run: OriginRun, anchor: Element) => {
    window.clearTimeout(timer.current);
    timer.current = window.setTimeout(() => setCard({ run, anchor }), SHOW_MS);
  }, []);
  const hide = useCallback(() => {
    window.clearTimeout(timer.current);
    timer.current = window.setTimeout(() => setCard(null), HIDE_MS);
  }, []);
  const keep = useCallback(() => window.clearTimeout(timer.current), []);

  // Hovering added text shows the card of the run it belongs to.
  useEffect(() => {
    const dom = editor.view.dom;
    const runOf = (target: EventTarget | null): { run: OriginRun; block: Element } | null => {
      if (!(target instanceof Element) || editor.isDestroyed) return null;
      const block = target.closest("[data-origin]");
      if (!block || block.parentElement !== dom) return null;
      let pos: number;
      try {
        pos = editor.view.posAtDOM(block, 0) - 1;
      } catch {
        return null;
      }
      const run = originRuns(editor.state)
        .filter((each) => each.pos <= pos)
        .at(-1);
      return run ? { run, block } : null;
    };
    const over = (event: PointerEvent) => {
      const found = runOf(event.target);
      if (found) show(found.run, found.block);
      else hide();
    };
    dom.addEventListener("pointerover", over);
    dom.addEventListener("pointerleave", hide);
    return () => {
      dom.removeEventListener("pointerover", over);
      dom.removeEventListener("pointerleave", hide);
      window.clearTimeout(timer.current);
    };
  }, [editor, show, hide]);

  return (
    <div ref={layer} className="conversation-links" data-testid="conversation-links">
      {runs.map((run) => {
        const name = listedName(run.origin);
        const label = name ? t("origin.from", { title: name }) : t("origin.open");
        return (
          <button
            key={run.blockId}
            type="button"
            data-testid="conversation-link"
            // Its card says the same on hover; a tooltip as well would cover it.
            aria-label={label}
            style={{ top: run.top }}
            // The note keeps its selection and focus.
            onMouseDown={(event) => event.preventDefault()}
            onPointerEnter={(event) => show(run, event.currentTarget)}
            onPointerLeave={hide}
            onFocus={(event) => show(run, event.currentTarget)}
            onBlur={hide}
            onClick={(event) => {
              if (name) {
                openOrigin(run.origin);
                return;
              }
              // Not listed (yet, just after the app opens, or any more): asked of the core, it
              // opens if it is there, and the card says why if it isn't.
              const anchor = event.currentTarget;
              core.resolveReference(run.origin.from).then(
                (target) =>
                  target.status === "available" ? openOrigin(run.origin) : setCard({ run, anchor }),
                (error: unknown) => useAppStore.getState().reportError(error),
              );
            }}
            className="conversation-link"
          >
            <ConversationLineIcon className="size-3.5" />
          </button>
        );
      })}
      {card && (
        <OriginCard
          origin={card.run.origin}
          anchor={card.anchor}
          onEnter={keep}
          onLeave={hide}
          onClose={() => setCard(null)}
        />
      )}
    </div>
  );
}

/**
 * Where a Note came from: "From the conversation “…”" (opening it), with the
 * Question asked; or that the conversation is in Recently deleted, or gone.
 */
function OriginCard({
  origin,
  anchor,
  onEnter,
  onLeave,
  onClose,
}: {
  origin: BlockOrigin;
  anchor: Element;
  onEnter(): void;
  onLeave(): void;
  onClose(): void;
}) {
  const t = useT();
  const [target, setTarget] = useState<ReferenceTarget | null>(null);
  useEffect(() => {
    let current = true;
    core.resolveReference(origin.from).then(
      (found) => {
        if (current) setTarget(found);
      },
      (error: unknown) => useAppStore.getState().reportError(error),
    );
    return () => {
      current = false;
    };
  }, [origin]);
  const title =
    target?.status === "available" || target?.status === "deleted"
      ? target.name || t("conversation.untitled")
      : "";
  return (
    <Popover
      anchor={anchor}
      placement="top-start"
      onDismiss={onClose}
      role="dialog"
      aria-label={t("origin.open")}
      data-testid="origin-card"
      data-status={target?.status ?? "loading"}
      onPointerEnter={onEnter}
      onPointerLeave={onLeave}
      className="origin-card"
    >
      {target?.status === "available" ? (
        <button
          type="button"
          data-testid="origin-open"
          onClick={() => {
            onClose();
            openOrigin(origin);
          }}
          className="origin-card-link"
        >
          <ConversationLineIcon className="size-3.5 shrink-0" />
          <span className="min-w-0">{t("origin.from", { title })}</span>
        </button>
      ) : (
        <p data-testid="origin-gone" className="origin-card-gone">
          <ConversationLineIcon className="size-3.5 shrink-0" />
          <span className="min-w-0">
            {target?.status === "deleted"
              ? t("origin.deleted", { title })
              : target
                ? t("origin.gone")
                : t("origin.open")}
          </span>
        </p>
      )}
      {origin.question && (
        <p data-testid="origin-question" className="origin-card-question">
          {t("origin.asked", { question: origin.question })}
        </p>
      )}
    </Popover>
  );
}
