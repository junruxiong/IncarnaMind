import { useEffect, useState } from "react";
import type { ConversationAnswer } from "../../core/api";
import { silentWaitMs } from "./answerWaitWords";
import { core } from "./core";

const hasWords = (answer: ConversationAnswer): boolean =>
  answer.text.trim() !== "" || answer.toolCalls.some((call) => call.mindChange !== undefined);

/**
 * Whether an Answer being written has had no words for long enough (counted
 * from when its Question went to the model, never while it waits for the
 * User), and what its model usually takes in seconds, null when not known.
 */
export function useAnswerWait(
  answer: ConversationAnswer,
  streaming: boolean,
  waitingForUser: boolean,
): { silent: boolean; usual: number | null } {
  const [silent, setSilent] = useState(false);
  const [usual, setUsual] = useState<number | null>(null);
  const words = hasWords(answer);
  const sentAt = Date.parse(answer.steps?.sentAt ?? "");
  const silentNow = streaming && !words && !Number.isNaN(sentAt);

  useEffect(() => {
    if (!silentNow) {
      setSilent(false);
      return;
    }
    const left = sentAt + silentWaitMs() - Date.now();
    if (left <= 0) {
      setSilent(true);
      return;
    }
    setSilent(false);
    const timer = setTimeout(() => setSilent(true), left);
    return () => clearTimeout(timer);
  }, [silentNow, sentAt]);

  useEffect(() => {
    if (!streaming) return;
    let current = true;
    core.usualAnswerWait(answer.id).then(
      (seconds) => {
        if (current) setUsual(seconds);
      },
      (error: unknown) =>
        console.error(`How long Answers of ${answer.model?.modelId} take couldn't be read.`, error),
    );
    return () => {
      current = false;
    };
  }, [streaming, answer.id, answer.model?.modelId]);

  return { silent: silent && !waitingForUser, usual };
}
