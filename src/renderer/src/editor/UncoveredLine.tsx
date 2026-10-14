import { ProblemLine } from "../components/ProblemLine";
import { useT } from "../i18n";
import { useAppStore } from "../store";

/**
 * "Your Documents don't cover most of this · Add files… · Check against all
 * Documents" (#764): said once, above the results, when most sentences of a
 * check or a Verify run found nothing, so the User isn't left to read a wall
 * of marks. Check against all Documents only when some were checked in a Folder.
 */
export function UncoveredLine({
  onCheckAll,
  className = "",
}: {
  onCheckAll?: (() => void) | undefined;
  className?: string;
}) {
  const t = useT();
  return (
    <ProblemLine
      role="status"
      testId="checks-uncovered"
      className={className}
      action={{
        label: t("checks.addFiles"),
        testId: "checks-add-files",
        onClick: () => void useAppStore.getState().pickDocuments(),
      }}
      also={
        onCheckAll && {
          label: t("checks.checkAll"),
          testId: "checks-check-all",
          onClick: onCheckAll,
        }
      }
    >
      {t("checks.uncovered")}
    </ProblemLine>
  );
}
