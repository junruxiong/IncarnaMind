import { NodeViewContent, NodeViewWrapper, type ReactNodeViewProps } from "@tiptap/react";
import { useT } from "../i18n";
import { CODE_LANGUAGES } from "./codeLanguages";

/** A code block with its language picker in the top-right corner. */
export function CodeBlockView({ node, updateAttributes }: ReactNodeViewProps) {
  const t = useT();
  const language = typeof node.attrs.language === "string" ? node.attrs.language : "";
  const known = Object.hasOwn(CODE_LANGUAGES, language);

  return (
    <NodeViewWrapper className="code-block">
      <select
        contentEditable={false}
        aria-label={t("editor.code.language")}
        value={language}
        onChange={(event) => updateAttributes({ language: event.target.value || null })}
        className="code-language"
      >
        <option value="">{t("editor.code.auto")}</option>
        {/* A language set some other way, e.g. a pasted ```js fence. */}
        {language && !known && <option value={language}>{language}</option>}
        {Object.entries(CODE_LANGUAGES).map(([name, label]) => (
          <option key={name} value={name}>
            {label}
          </option>
        ))}
      </select>
      <pre>
        <NodeViewContent<"code"> as="code" />
      </pre>
    </NodeViewWrapper>
  );
}
