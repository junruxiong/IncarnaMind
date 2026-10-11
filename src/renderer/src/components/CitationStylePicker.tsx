import { useRef } from "react";
import { CITATION_STYLES, type CitationStyleId } from "../../../shared/citationStyles";
import { translate } from "../../../shared/i18n";
import { chooseCitationStyle, useCitationStylePicker } from "../citationStylePicker";
import { useT } from "../i18n";
import { SearchPicker } from "./SearchPicker";

/**
 * The citation style picker (#289), mounted once in the window and opened through
 * `useCitationStylePicker`: a field to search the styles by name (in either language, or by
 * what they're also called: "Cite Them Right", "国标"), the shipped styles in the order of the
 * common list, the one the Mind is in ticked, and the credit the CSL project's licence asks for.
 */
export function CitationStylePickerHost() {
  const t = useT();
  const open = useCitationStylePicker((state) => state.open);
  const close = useCitationStylePicker((state) => state.close);
  /** Where the picker opens when nothing was given to open beside: the top of the window. */
  const fallback = useRef<HTMLSpanElement>(null);
  const anchor = open ? (open.anchor?.isConnected ? open.anchor : fallback.current) : null;
  const options = CITATION_STYLES.map((style) => ({
    id: style.id,
    label: t(style.label),
    keywords: [translate("en", style.label), translate("zh-CN", style.label), ...style.keywords],
    data: { "data-style": style.id },
  }));
  return (
    <>
      <span
        ref={fallback}
        aria-hidden="true"
        className="pointer-events-none fixed top-12 left-1/2 h-0 w-0"
      />
      <SearchPicker<CitationStyleId>
        anchor={anchor}
        title={t("citationStyle.title")}
        searchLabel={t("citationStyle.search")}
        noMatch={(query) => t("citationStyle.noMatch", { query })}
        options={options}
        current={open?.current?.style}
        currentLabel={t("citationStyle.current")}
        onChoose={(style) => {
          if (open) void chooseCitationStyle(open.mindId, style, open.current);
        }}
        onClose={close}
        testId="citation-style"
        className="w-72"
        footer={
          <>
            {t("citationStyle.credit")}{" "}
            <a
              href="https://citationstyles.org/"
              target="_blank"
              rel="noreferrer"
              className="text-accent underline-offset-2 hover:underline"
            >
              citationstyles.org
            </a>
          </>
        }
      />
    </>
  );
}
