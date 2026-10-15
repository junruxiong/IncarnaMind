/**
 * Checks of the User's own sentences in the editor (#96): the mark that
 * holds a sentence's result (`CLAIM_CHECK_MARK`), where each check is now,
 * how it shows (`claimCheckShown`), and the wash behind a sentence found
 * unsupported or contradicted.
 */
import { Extension, Mark } from "@tiptap/core";
import type { Node as ProseMirrorNode } from "@tiptap/pm/model";
import { type EditorState, Plugin, PluginKey } from "@tiptap/pm/state";
import { Decoration, DecorationSet, type EditorView } from "@tiptap/pm/view";
import { CLAIM_CHECK_MARK, type ClaimCheckAttributes } from "../../../core/api";
import { type ClaimCheckShown, claimCheckShown, readClaimCheck } from "../../../shared/claimChecks";
import { INLINE_NODE, sentenceSpans } from "../../../shared/sentences";

/** Every attribute a check's mark keeps: all of `ClaimCheckAttributes`, never read from pasted HTML. */
const ATTRIBUTES: readonly (keyof ClaimCheckAttributes)[] = [
  "id",
  "request",
  "action",
  "state",
  "reason",
  "detail",
  "text",
  "evidence",
  "says",
  "calculation",
  "scope",
  "versions",
  "excluded",
  "checkedAt",
];

/**
 * A sentence's check, on its text. A copied sentence pastes without it: its
 * check belongs where it was made.
 */
export const ClaimCheck = Mark.create({
  name: CLAIM_CHECK_MARK,
  inclusive: false,
  addAttributes: () =>
    Object.fromEntries(
      ATTRIBUTES.map((name) => [
        name,
        { default: null, parseHTML: () => null, renderHTML: () => ({}) },
      ]),
    ),
  parseHTML: () => [],
  renderHTML: ({ mark }) => ["span", { "data-claim-check": String(mark.attrs.id ?? "") }, 0],
});

/** A paragraph's text as checks read it: an inline node as U+FFFC, a line break as "\n". */
export function paragraphText(block: ProseMirrorNode): string {
  let text = "";
  block.forEach((child) => {
    text += child.isText
      ? (child.text ?? "")
      : child.type.name === "hardBreak"
        ? "\n"
        : INLINE_NODE;
  });
  return text;
}

/** A check where it is in the editor now. */
export interface ClaimSpan {
  check: ClaimCheckAttributes;
  /** The first marked character, and after the last. */
  from: number;
  to: number;
  /** The whole sentences the marked text is in, as they read now. */
  sentence: { from: number; to: number; text: string };
}

/** Every check in the document, top to bottom. */
export function claimSpans(doc: ProseMirrorNode): ClaimSpan[] {
  const spans: ClaimSpan[] = [];
  doc.descendants((block, pos) => {
    if (!block.isTextblock) return true;
    const found = new Map<string, { check: ClaimCheckAttributes; from: number; to: number }>();
    let offset = 0;
    block.forEach((child) => {
      const mark = child.marks.find((each) => each.type.name === CLAIM_CHECK_MARK);
      const check = mark ? readClaimCheck(mark.attrs) : null;
      if (check) {
        const known = found.get(check.id);
        if (known) known.to = offset + child.nodeSize;
        else found.set(check.id, { check, from: offset, to: offset + child.nodeSize });
      }
      offset += child.nodeSize;
    });
    if (found.size === 0) return false;
    const text = paragraphText(block);
    const sentences = sentenceSpans(text);
    for (const { check, from, to } of found.values()) {
      const touched = sentences.filter((span) => span.from < to && span.to > from);
      const start = Math.min(from, touched[0]?.from ?? from);
      const end = Math.max(to, touched.at(-1)?.to ?? to);
      spans.push({
        check,
        from: pos + 1 + from,
        to: pos + 1 + to,
        sentence: { from: pos + 1 + start, to: pos + 1 + end, text: text.slice(start, end) },
      });
    }
    return false;
  });
  return spans;
}

/** What a check is compared with besides its sentence: the Documents' versions and the Mind's Folder. */
export interface ClaimContext {
  /** Each live Document's current version, by id; null while they aren't known. */
  versions: ReadonlyMap<string, string> | null;
  folderId: string | null;
}

/** A check where it is, and how it shows. */
export interface ShownClaim extends ClaimSpan {
  shown: ClaimCheckShown;
}

interface ClaimState {
  context: ClaimContext;
  claims: readonly ShownClaim[];
  decorations: DecorationSet;
}

const claimsKey = new PluginKey<ClaimState>("claimChecks");

/** A transaction with this meta compares the checks with new Documents or a new Folder (`setClaimContext`). */
const CONTEXT = "claimContext";

const NO_CONTEXT: ClaimContext = { versions: null, folderId: null };

function stateOf(doc: ProseMirrorNode, context: ClaimContext): ClaimState {
  const claims = claimSpans(doc).map((span) => ({
    ...span,
    shown: claimCheckShown(span.check, { ...context, text: span.sentence.text }),
  }));
  const washes = claims.flatMap(({ shown, from, to }) =>
    shown.state === "unsupported" || shown.state === "contradicted"
      ? [Decoration.inline(from, to, { class: `claim-wash claim-wash--${shown.state}` })]
      : [],
  );
  return { context, claims, decorations: DecorationSet.create(doc, washes) };
}

/** How the checks of a Mind's sentences cover its Documents (#764). */
export interface Coverage {
  /** More than half of the sentences that got a verdict found nothing in the Documents. */
  mostlyUncovered: boolean;
  /** Some of them were checked in a Folder only, so every Document could still say something. */
  canWiden: boolean;
  /** From the first to the last sentence that found nothing: where "Check against all Documents" goes. */
  range: { from: number; to: number } | null;
  /** How the first of them was checked. */
  action: ShownClaim["check"]["action"] | null;
}

const NO_COVERAGE: Coverage = {
  mostlyUncovered: false,
  canWiden: false,
  range: null,
  action: null,
};

/** Whether the checks, taken together, say the Documents don't cover most of what was checked. */
export function coverageOf(claims: readonly ShownClaim[]): Coverage {
  let checked = 0;
  const lacking: ShownClaim[] = [];
  for (const claim of claims) {
    const { state } = claim.shown;
    if (state === "supported" || state === "contradicted") checked++;
    else if (state === "unsupported") {
      checked++;
      lacking.push(claim);
    }
  }
  const first = lacking[0];
  const last = lacking.at(-1);
  if (!first || !last || checked < 2 || lacking.length * 2 <= checked) return NO_COVERAGE;
  return {
    mostlyUncovered: true,
    canWiden: lacking.some(
      (claim) => !claim.check.scope.widened && claim.check.scope.folderId !== null,
    ),
    range: { from: first.from, to: last.to },
    action: first.check.action,
  };
}

/** Whether two readings of the coverage say the same. */
export const sameCoverage = (a: Coverage, b: Coverage | null | undefined): boolean =>
  b !== null &&
  b !== undefined &&
  a.mostlyUncovered === b.mostlyUncovered &&
  a.canWiden === b.canWiden &&
  a.range?.from === b.range?.from &&
  a.range?.to === b.range?.to;

/** The checks in the editor, as they show now. Empty without `ClaimChecks`. */
export function shownClaims(state: EditorState): readonly ShownClaim[] {
  return claimsKey.getState(state)?.claims ?? [];
}

/** Compares the checks with the Documents and the Mind's Folder as they are now. */
export function setClaimContext(view: EditorView, context: ClaimContext): void {
  const now = claimsKey.getState(view.state)?.context;
  if (now?.versions === context.versions && now.folderId === context.folderId) return;
  view.dispatch(view.state.tr.setMeta(CONTEXT, context).setMeta("addToHistory", false));
}

/** Keeps the checks, how they show, and the washes behind sentences found unsupported or contradicted. */
export const ClaimChecks = Extension.create({
  name: "claimChecks",

  addProseMirrorPlugins() {
    return [
      new Plugin<ClaimState>({
        key: claimsKey,
        state: {
          init: (_config, state) => stateOf(state.doc, NO_CONTEXT),
          apply: (tr, claims) => {
            const context = tr.getMeta(CONTEXT) as ClaimContext | undefined;
            if (context) return stateOf(tr.doc, context);
            return tr.docChanged ? stateOf(tr.doc, claims.context) : claims;
          },
        },
        props: {
          decorations: (state) => claimsKey.getState(state)?.decorations,
        },
      }),
    ];
  },
});
