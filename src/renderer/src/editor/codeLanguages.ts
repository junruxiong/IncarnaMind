import bash from "highlight.js/lib/languages/bash";
import cpp from "highlight.js/lib/languages/cpp";
import css from "highlight.js/lib/languages/css";
import go from "highlight.js/lib/languages/go";
import java from "highlight.js/lib/languages/java";
import javascript from "highlight.js/lib/languages/javascript";
import json from "highlight.js/lib/languages/json";
import markdown from "highlight.js/lib/languages/markdown";
import python from "highlight.js/lib/languages/python";
import r from "highlight.js/lib/languages/r";
import rust from "highlight.js/lib/languages/rust";
import sql from "highlight.js/lib/languages/sql";
import typescript from "highlight.js/lib/languages/typescript";
import xml from "highlight.js/lib/languages/xml";
import yaml from "highlight.js/lib/languages/yaml";
import { createLowlight } from "lowlight";

/**
 * The languages code blocks highlight, by highlight.js name, with the name the
 * language picker shows. A few common ones rather than all of highlight.js;
 * each also answers to its usual aliases (e.g. `js`, `py`, `html`).
 */
export const CODE_LANGUAGES = {
  bash: "Bash",
  cpp: "C/C++",
  css: "CSS",
  go: "Go",
  java: "Java",
  javascript: "JavaScript",
  json: "JSON",
  markdown: "Markdown",
  python: "Python",
  r: "R",
  rust: "Rust",
  sql: "SQL",
  typescript: "TypeScript",
  xml: "HTML/XML",
  yaml: "YAML",
} as const;

export const lowlight = createLowlight({
  bash,
  cpp,
  css,
  go,
  java,
  javascript,
  json,
  markdown,
  python,
  r,
  rust,
  sql,
  typescript,
  xml,
  yaml,
});
