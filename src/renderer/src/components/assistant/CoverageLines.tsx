import type { ConversationAnswer, ConversationQuestion } from "../../../../core/api";
import { askInAssistant } from "../../conversations";
import { useLanguage, useT } from "../../i18n";
import { MetaLine } from "../MetaLine";
import { LineAction } from "../ProblemLine";

/**
 * Above an Answer that covered only some of the Documents in scope: which,
 * and an offer of the rest, asked again in a new Answer under this one, which
 * stays as it is: "I'll summarise the 8 largest of your 14 Documents
 * now. · Summarise the other 6", "I took “10-K” and “Facility agreement” as
 * the main ones. · Use others".
 */
export function CoverageLine({
  answer,
  question,
}: {
  answer: Pick<ConversationAnswer, "steps">;
  question: ConversationQuestion | undefined;
}) {
  const t = useT();
  const language = useLanguage();
  const coverage = answer.steps?.coverage;
  if (!coverage || !question) return null;
  const { used, rest } = coverage;
  const text =
    coverage.kind === "largest"
      ? t("coverage.largest", { count: used.length, total: used.length + rest.length })
      : used.length === 1
        ? t("coverage.main.one", { document: `“${used[0]?.name ?? ""}”` })
        : t("coverage.main", {
            documents: new Intl.ListFormat(language, { type: "conjunction" }).format(
              used.map((each) => `“${each.name}”`),
            ),
          });
  const askAgain = () =>
    void askInAssistant({
      text: question.text,
      mindId: question.askedFrom && !question.askedFrom.deleted ? question.askedFrom.mindId : null,
      focus: question.focus,
      scope: { folderIds: [], tagIds: [], documentIds: rest },
      model: question.model,
      forcedSkill: question.forcedSkill,
    });
  return (
    <MetaLine
      data-testid="answer-coverage"
      className="answer-coverage"
      parts={[
        text,
        rest.length > 0 && (
          <LineAction
            action={{
              label:
                coverage.kind === "largest"
                  ? t("coverage.largest.rest", { count: rest.length })
                  : t("coverage.main.others"),
              testId: "answer-coverage-rest",
              onClick: askAgain,
            }}
          />
        ),
      ]}
    />
  );
}
