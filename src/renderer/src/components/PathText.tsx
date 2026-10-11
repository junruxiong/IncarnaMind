/**
 * A path on one line that, when it doesn't fit, is cut at its start, so the
 * folder's own name stays in view ("…/Documents/IncarnaMind"); the whole path
 * is its tooltip. The right-to-left box puts the ellipsis on the left; the
 * path inside it stays left to right.
 */
export function PathText({
  path,
  testId,
  className = "",
}: {
  path: string;
  testId?: string;
  className?: string;
}) {
  return (
    <span
      dir="rtl"
      data-testid={testId}
      title={path}
      className={`block min-w-0 truncate text-left ${className}`}
    >
      <span dir="ltr">{path}</span>
    </span>
  );
}
