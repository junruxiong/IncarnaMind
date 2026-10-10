import { useEffect, useRef, useState } from "react";
import { core } from "../core";
import { useT } from "../i18n";
import { sentencesEnd, spokenText } from "./spokenText";

/** How often, at most, more of a streaming Answer is read out. */
const ANNOUNCE_EVERY_MS = 1_500;

/**
 * Reads out the Answers being written in this Mind, in a polite live region
 * (it waits for a pause): that one is being written, then its sentences as
 * they arrive, a few at a time, then that it is done or stopped. A failed
 * Answer says so itself, as an alert under it.
 */
export function AnswerAnnouncer({ mindId }: { mindId: string }) {
  const t = useT();
  const words = useRef(t);
  words.current = t;
  const [said, setSaid] = useState("");

  useEffect(() => {
    const t = (key: Parameters<typeof words.current>[0]) => words.current(key);
    /** By Answer: its Markdown so far, and how much of it has been read out. */
    const answers = new Map<string, { text: string; told: number }>();
    let timer: ReturnType<typeof setTimeout> | null = null;
    /** What is still to be read out of each Answer, to the end of its last whole sentence (or all). */
    const take = (final: boolean): string => {
      const pieces: string[] = [];
      for (const answer of answers.values()) {
        const end = final ? answer.text.length : sentencesEnd(answer.text, answer.told);
        if (end <= answer.told) continue;
        pieces.push(spokenText(answer.text.slice(answer.told, end)));
        answer.told = end;
      }
      return pieces.filter(Boolean).join(" ");
    };
    const later = () => {
      if (timer) return;
      timer = setTimeout(() => {
        timer = null;
        const words = take(false);
        if (words) setSaid(words);
      }, ANNOUNCE_EVERY_MS);
    };
    const stops = [
      core.on("answer.started", (event) => {
        if (event.mindId !== mindId) return;
        answers.set(event.answerId, { text: "", told: 0 });
        setSaid(t("composer.live.writing"));
      }),
      core.on("answer.delta", (event) => {
        const answer = answers.get(event.answerId);
        if (!answer) return;
        answer.text += event.text;
        later();
      }),
      core.on("answer.finished", (event) => {
        if (!answers.has(event.answerId)) return;
        const rest = take(true);
        answers.delete(event.answerId);
        const end = t(event.status === "stopped" ? "composer.live.stopped" : "composer.live.done");
        setSaid(rest ? `${rest} ${end}` : end);
      }),
      core.on("answer.failed", (event) => {
        answers.delete(event.answerId);
      }),
    ];
    return () => {
      if (timer) clearTimeout(timer);
      for (const stop of stops) stop();
    };
  }, [mindId]);

  return (
    <p role="status" aria-live="polite" data-testid="composer-live" className="sr-only">
      {said}
    </p>
  );
}
