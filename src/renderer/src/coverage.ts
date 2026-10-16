/**
 * Answers asked again from an earlier Answer's coverage line: its
 * Question's Search scope is the Documents the earlier one didn't use. Only
 * data.
 */
import type { ConversationMessage } from "../../core/api";

const sameSet = (a: readonly string[], b: readonly string[]) =>
  a.length === b.length && a.every((id) => b.includes(id));

/** For each Answer asked again with an earlier Answer's other Documents, how many those were. */
export function askedAgainCounts(messages: readonly ConversationMessage[]): Map<string, number> {
  const counts = new Map<string, number>();
  const offered: string[][] = [];
  const questions = new Map(
    messages.flatMap((message) => (message.kind === "question" ? [[message.id, message]] : [])),
  );
  for (const message of messages) {
    if (message.kind !== "answer") continue;
    const scope = questions.get(message.questionId)?.scope;
    const ids =
      scope && scope.folderIds.length + scope.tagIds.length === 0 ? scope.documentIds : [];
    if (ids.length > 0 && offered.some((rest) => sameSet(rest, ids)))
      counts.set(message.id, ids.length);
    const rest = message.steps?.coverage?.rest;
    if (rest && rest.length > 0) offered.push(rest);
  }
  return counts;
}
