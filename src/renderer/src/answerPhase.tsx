import { useAnswers } from "./answers";
import { useT } from "./i18n";

/**
 * What an Answer being written is doing, for its meta line: "Loading the
 * model…", "Searching your Documents…" or "Writing…", from the core's
 * "answer.phase" events; "Writing…" until the first one arrives.
 */
export function AnswerPhaseText({ answerId }: { answerId: string | null }) {
  const t = useT();
  const phase = useAnswers((state) => (answerId ? state.phases[answerId] : undefined));
  return t(phase ? `answer.phase.${phase}` : "answer.status.streaming");
}
