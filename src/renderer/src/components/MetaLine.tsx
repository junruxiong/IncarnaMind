import type { HTMLAttributes, ReactNode } from "react";

/**
 * A line of parts joined by middle dots ("Checked 4 claims · 3 need a look ·
 * Check again"). It wraps only between parts, and no line starts or ends with
 * a dot: each dot is drawn in the gap before its part and clipped where that
 * part starts a line (`.meta-line` in styles.css). Use it, or `.meta-line`
 * and `.meta-part` on a flex row, wherever parts are joined by dots.
 */
export function MetaLine({
  parts,
  as: Tag = "p",
  className = "",
  ...rest
}: { parts: readonly ReactNode[]; as?: "p" | "div" } & Omit<
  HTMLAttributes<HTMLElement>,
  "children"
>) {
  return (
    <Tag {...rest} className={`meta-line ${className}`}>
      {parts
        .filter((part) => part !== null && part !== undefined && part !== false && part !== "")
        .map((part, index) => (
          // biome-ignore lint/suspicious/noArrayIndexKey: parts have no identity, and the line is rebuilt whole.
          <span key={index} className="meta-part">
            {index > 0 && (
              <span className="meta-dot" aria-hidden="true">
                {" · "}
              </span>
            )}
            {part}
          </span>
        ))}
    </Tag>
  );
}
