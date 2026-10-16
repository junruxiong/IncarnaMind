import { type AskedOutcome, reviewAsked, undoAsked } from "../../askedChanges";
import type { ChangedMind } from "../../changedMinds";
import { changeFact } from "../../changeFacts";
import { useT } from "../../i18n";
import { showChanges, type Waiting } from "../../mindChanges";
import { PencilLineIcon } from "../lineIcons";

/**
 * The card under an Answer whose changes the User asked for applied at once
 * (#880, D224): a line for each Mind with the count its own line shows,
 * "Changed 12 places in “Credit note: draft” · Undo · Review changes", where
 * Undo takes the whole turn back; then a line of their own for the changes
 * it made beyond the request, which wait: "I also suggested 2 changes in
 * “Sources” · Review".
 */
export function AskedChangesCard({
  answerId,
  rows,
  outcomes,
  waiting,
  nameOf,
}: {
  answerId: string;
  rows: readonly ChangedMind[];
  outcomes: Readonly<Record<string, AskedOutcome>>;
  waiting: readonly Waiting[];
  nameOf(row: ChangedMind): string;
}) {
  const t = useT();
  const applied = rows.filter((row) => outcomes[row.mindId] !== undefined);
  const also = rows.flatMap((row, index) => {
    const count = waiting[index];
    return outcomes[row.mindId] === undefined && typeof count === "number" && count > 0
      ? [{ row, count }]
      : [];
  });
  return (
    <section data-testid="mind-changes" className="mind-changes">
      {applied.map((row) => {
        const outcome = outcomes[row.mindId];
        return (
          <div key={row.mindId} className="mind-changes-head">
            <PencilLineIcon aria-hidden="true" className="mind-changes-icon" />
            <p className="mind-changes-line">
              <span data-testid="mind-changes-title" className="mind-changes-title">
                {changeFact(t, row, nameOf(row))}
              </span>
              {outcome === "applied" && (
                <>
                  <Dot />
                  <button
                    type="button"
                    data-testid="mind-changes-undo"
                    onClick={() => undoAsked(answerId)}
                    className="mind-changes-review"
                  >
                    {t("edit.applied.undo")}
                  </button>
                  <Dot />
                  <button
                    type="button"
                    data-testid="mind-changes-review"
                    onClick={() => void reviewAsked(row.mindId)}
                    className="mind-changes-review"
                  >
                    {t("edit.applied.review")}
                  </button>
                </>
              )}
              {outcome === "undone" && (
                <>
                  <Dot />
                  <span data-testid="mind-changes-undone" className="mind-changes-count">
                    {t("mindChanges.undone")}
                  </span>
                </>
              )}
              {outcome === "held" && (
                <>
                  <Dot />
                  <button
                    type="button"
                    data-testid="mind-changes-review"
                    onClick={() => showChanges(row)}
                    className="mind-changes-review"
                  >
                    {t("mindChanges.review")}
                  </button>
                </>
              )}
            </p>
          </div>
        );
      })}
      {also.map(({ row, count }) => (
        <p key={row.mindId} data-testid="mind-changes-also" className="mind-changes-line">
          <span className="mind-changes-title">
            {count === 1
              ? t("mindChanges.also.one", { mind: nameOf(row) })
              : t("mindChanges.also", { mind: nameOf(row), count })}
          </span>
          <Dot />
          <button
            type="button"
            data-testid="mind-changes-also-review"
            onClick={() => showChanges(row)}
            className="mind-changes-review"
          >
            {t("mindChanges.also.review")}
          </button>
        </p>
      ))}
    </section>
  );
}

const Dot = () => (
  <span aria-hidden="true" className="mind-changes-dot">
    {" · "}
  </span>
);
