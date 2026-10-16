/**
 * What an Answer's calls did in a Mind, in words: on the Answer's card, with
 * the Mind's name ("Changed 3 places in “Q3 review”"), and on the Mind's own
 * line, without it ("Changed 3 places"). One function, so the two counts
 * always agree (#880).
 */
import type { MessageKey, MessageParams } from "../../shared/i18n";

type Translate = (key: MessageKey, params?: MessageParams) => string;

/** What was done in one Mind: places changed, paragraphs written in whole Blocks, tables added. */
export interface ChangeCounts {
  created: boolean;
  changed: number;
  wrote: number;
  tables: number;
}

const counted = (t: Translate, count: number, what: "tables" | "paragraphs" | "places") =>
  count === 1 ? t(`mindChanges.count.${what}.one`) : t(`mindChanges.count.${what}`, { count });

/** The facts in words, with the Mind's name, or without it (`null`) for the Mind's own line. */
export function changeFact(t: Translate, row: ChangeCounts, name: string | null): string {
  const count = row.wrote + row.changed;
  if (row.tables > 0) return tablesFact(t, row, name);
  if (name === null) {
    if (row.changed === 0 && count > 0) {
      return count === 1 ? t("edit.wrote.one") : t("edit.wrote", { count });
    }
    return count === 1 ? t("edit.applied.one") : t("edit.applied", { count });
  }
  if (row.created) {
    if (count === 0) return t("mindChanges.one.created.none", { mind: name });
    return count === 1
      ? t("mindChanges.created.one", { mind: name })
      : t("mindChanges.created", { mind: name, count });
  }
  if (count === 0) return t("mindChanges.one.edited", { mind: name });
  if (row.changed === 0) {
    return count === 1
      ? t("mindChanges.wrote.one", { mind: name })
      : t("mindChanges.wrote", { mind: name, count });
  }
  return count === 1
    ? t("mindChanges.changed.one", { mind: name })
    : t("mindChanges.changed", { mind: name, count });
}

function tablesFact(t: Translate, row: ChangeCounts, name: string | null): string {
  const tables = counted(t, row.tables, "tables");
  const count = row.wrote + row.changed;
  if (name === null) {
    if (count === 0) return t("edit.tables", { tables });
    return row.changed === 0
      ? t("edit.wrote.tables", { paragraphs: counted(t, count, "paragraphs"), tables })
      : t("edit.applied.tables", { places: counted(t, count, "places"), tables });
  }
  if (count === 0) {
    return row.created
      ? t("mindChanges.created.tables", { mind: name, tables })
      : t("mindChanges.tables", { mind: name, tables });
  }
  const paragraphs = counted(t, count, "paragraphs");
  if (row.created) return t("mindChanges.created.wrote.tables", { mind: name, paragraphs, tables });
  if (row.changed === 0) return t("mindChanges.wrote.tables", { mind: name, paragraphs, tables });
  return t("mindChanges.changed.tables", {
    mind: name,
    places: counted(t, count, "places"),
    tables,
  });
}
