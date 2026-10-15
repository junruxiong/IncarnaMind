import { useMemo } from "react";
import type { ConversationAnswer, ConversationQuestion } from "../../../../core/api";
import { mindName } from "../../../../shared/naming";
import { type ChangedMind, changedMinds } from "../../changedMinds";
import { askInAssistant } from "../../conversations";
import { useT } from "../../i18n";
import { resolveAllChanges, showChanges, useWaitingChanges, type Waiting } from "../../mindChanges";
import type { Translate } from "../../objects";
import { useAppStore } from "../../store";
import { MindLineIcon, PencilLineIcon } from "../lineIcons";

/** The names of Minds seen in a card, for a row whose Mind has since been removed. */
const seenNames = new Map<string, string>();

/** What a card says of one Mind: made or changed, by its name, and how many of its changes wait. */
function oneMindText(t: Translate, row: ChangedMind, name: string, waiting: Waiting) {
  if (waiting === "gone") return t("mindChanges.one.removed", { mind: name });
  if (row.created) {
    if (waiting === null || waiting === 0) return t("mindChanges.one.created.none", { mind: name });
    return waiting === 1
      ? t("mindChanges.one.created.one", { mind: name })
      : t("mindChanges.one.created", { mind: name, count: waiting });
  }
  if (waiting === null) return t("mindChanges.one.edited", { mind: name });
  if (waiting === 0) return t("mindChanges.one.none", { mind: name });
  return waiting === 1
    ? t("mindChanges.one.one", { mind: name })
    : t("mindChanges.one", { mind: name, count: waiting });
}

function countText(t: Translate, waiting: number): string {
  if (waiting === 0) return t("mindChanges.row.none");
  return waiting === 1 ? t("mindChanges.row.one") : t("mindChanges.row.count", { count: waiting });
}

/**
 * The card under the reply of an Answer that changed Minds (#202, D148; the
 * boards Asst-Changes and Asst-Change-In-Mind): "2 Minds changed · Review",
 * a row for each Mind, Created or Edited, with how many of its changes wait
 * and Open, then Accept all · Reject all. For one Mind, its row is the card's
 * head: "3 changes in “Q3 review” · Review". Review and Open show the Mind
 * at its first change; Accept all and Reject all offer Undo on the Answer's
 * line. Rejecting a Mind the Answer made removes it.
 */
export function MindChangesCard({
  answer,
  streaming,
}: {
  answer: Pick<ConversationAnswer, "id" | "toolCalls">;
  /** Changes may still come: Accept all and Reject all wait for the Answer to end. */
  streaming: boolean;
}) {
  const t = useT();
  const rows = useMemo(() => changedMinds(answer), [answer]);
  const waiting = useWaitingChanges(rows);
  const minds = useAppStore((state) => state.minds);
  if (rows.length === 0) return null;

  const nameOf = (row: ChangedMind) => {
    const mind = minds.find((each) => each.id === row.mindId);
    const name = (mind && mindName(mind)) || seenNames.get(row.mindId) || t("mind.untitled");
    if (mind) seenNames.set(row.mindId, name);
    return name;
  };
  const counted = (index: number) => {
    const count = waiting[index];
    return typeof count === "number" ? count : 0;
  };
  const total = rows.reduce((sum, _, index) => sum + counted(index), 0);
  const first = rows.find((_, index) => counted(index) > 0);
  const [only] = rows;
  const head =
    rows.length === 1 && only
      ? oneMindText(t, only, nameOf(only), waiting[0] ?? null)
      : t("mindChanges.minds", { count: rows.length });

  return (
    <section aria-label={head} data-testid="mind-changes" className="mind-changes">
      <div className="mind-changes-head">
        <PencilLineIcon aria-hidden="true" className="mind-changes-icon" />
        <span data-testid="mind-changes-title" className="mind-changes-title" title={head}>
          {head}
        </span>
        {first && (
          <>
            <span aria-hidden="true" className="mind-changes-dot">
              ·
            </span>
            <button
              type="button"
              data-testid="mind-changes-review"
              onClick={() => showChanges(first)}
              className="mind-changes-review"
            >
              {t("mindChanges.review")}
            </button>
          </>
        )}
      </div>
      {rows.length > 1 &&
        rows.map((row, index) => {
          const count = waiting[index] ?? null;
          const name = nameOf(row);
          const label =
            count === "gone"
              ? t("mindChanges.row.removed", { mind: name })
              : t(row.created ? "mindChanges.row.created" : "mindChanges.row.edited", {
                  mind: name,
                });
          return (
            <div
              key={row.mindId}
              data-testid="mind-changes-row"
              data-created={row.created || undefined}
              className="mind-changes-row"
            >
              <MindLineIcon aria-hidden="true" className="mind-changes-icon" />
              <span className="mind-changes-name" title={label}>
                {label}
              </span>
              {typeof count === "number" && !(row.created && count === 0) && (
                <span className="mind-changes-count">{countText(t, count)}</span>
              )}
              <span className="flex-1" />
              {count !== "gone" && (
                <button
                  type="button"
                  data-testid="mind-changes-open"
                  aria-label={`${t("mindChanges.open")} ${name}`}
                  onClick={() => showChanges(row)}
                  className="mind-changes-action"
                >
                  {t("mindChanges.open")}
                </button>
              )}
            </div>
          );
        })}
      {total > 0 && !streaming && (
        <div className="mind-changes-foot">
          <button
            type="button"
            data-testid="mind-changes-accept-all"
            onClick={() => void resolveAllChanges(answer.id, rows, "accept")}
            className="mind-changes-accept"
          >
            {t("proposal.acceptAll")}
          </button>
          <button
            type="button"
            data-testid="mind-changes-reject-all"
            onClick={() => void resolveAllChanges(answer.id, rows, "reject")}
            className="mind-changes-action"
          >
            {t("proposal.rejectAll")}
          </button>
        </div>
      )}
    </section>
  );
}

/** Whether an Answer tried to change the Mind in focus, and there was none. */
export function wantedAMind(answer: Pick<ConversationAnswer, "toolCalls">): boolean {
  return answer.toolCalls.some(
    (call) =>
      call.source === "minds" &&
      call.status === "failed" &&
      (call.tool === "edit_mind" || call.tool === "add_to_mind") &&
      typeof call.input.mind !== "string",
  );
}

/**
 * Under an Answer asked with no Mind in focus that tried to change one
 * (#202): it says so, and offers New Mind, which makes one (in the Folder of
 * the Question's Search scope, when that is one Folder) and asks the
 * Question again from it, so the change goes there.
 */
export function NoMindLine({ question }: { question: ConversationQuestion }) {
  const t = useT();
  const { scope } = question;
  const folderId =
    scope && scope.folderIds.length === 1 && scope.tagIds.length + scope.documentIds.length === 0
      ? (scope.folderIds[0] ?? null)
      : null;
  const askInNewMind = () =>
    void useAppStore.getState().createMind({
      folderId,
      focusBody: true,
      made: (mindId) =>
        void askInAssistant({
          mindId,
          text: question.text,
          scope,
          model: question.model,
          forcedSkill: question.forcedSkill,
        }),
    });
  return (
    <p data-testid="mind-changes-no-mind" className="mind-changes-no-mind">
      {t("mindChanges.noMind")}
      <button
        type="button"
        data-testid="mind-changes-new-mind"
        onClick={askInNewMind}
        className="mind-changes-review"
      >
        {t("sidebar.newMind")}
      </button>
    </p>
  );
}
