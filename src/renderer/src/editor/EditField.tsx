import type { Editor } from "@tiptap/core";
import type { FocusEvent, KeyboardEvent } from "react";
import { ChatLineIcon } from "../components/lineIcons";
import { ProblemLine } from "../components/ProblemLine";
import { stopEdit, useEdits } from "../editWithAi";
import { useT } from "../i18n";
import { isComposing } from "./composing";
import { EditProblemLine } from "./EditStatus";
import { holdSelection } from "./heldSelection";
import type { EditWithAiField, useEditField } from "./useEditField";

type Field = ReturnType<typeof useEditField>;

const CARET_KEYS: ReadonlySet<string> = new Set(["ArrowLeft", "ArrowRight", "Home", "End"]);

/**
 * "Edit with AI…" at the foot of the selection menu: what to change is typed
 * here, and Enter asks for it, as the composer would. The control at the
 * field's end moves the request and the selection to the composer instead.
 */
export function EditField({
  editor,
  edit,
  field,
  keys,
  onLeave,
  onHandOff,
}: {
  editor: Editor;
  edit: EditWithAiField;
  field: Field;
  /** ⌘J, as the menu's keys write it. */
  keys: string;
  onLeave(event: FocusEvent<HTMLInputElement>): void;
  onHandOff(): void;
}) {
  const t = useT();
  const problem = useEdits((state) => state.problems[edit.mindId]);

  const onKeyDown = (event: KeyboardEvent<HTMLInputElement>) => {
    if (isComposing(event)) return;
    if (CARET_KEYS.has(event.key)) {
      // These move the caret here; the menu would move to another row.
      event.stopPropagation();
      return;
    }
    if (event.key === "Enter" && !event.shiftKey && !event.altKey && !event.metaKey) {
      event.preventDefault();
      field.send();
    }
  };

  return (
    <>
      <div className="selection-edit">
        <input
          ref={field.input}
          type="text"
          data-menu-field=""
          data-testid="selection-edit-field"
          aria-label={t("editor.ai.edit")}
          placeholder={t("editor.ai.edit")}
          value={field.text}
          spellCheck={false}
          onChange={(event) => field.setText(event.target.value)}
          onKeyDown={onKeyDown}
          onFocus={() => holdSelection(editor, true, "edit-field")}
          onBlur={(event) => {
            holdSelection(editor, false, "edit-field");
            onLeave(event);
          }}
          className="selection-edit-input"
        />
        {field.text === "" && (
          <span aria-hidden="true" className="editor-menu-shortcut">
            {keys}
          </span>
        )}
        {edit.handOff && (
          <button
            type="button"
            role="menuitem"
            data-testid="selection-edit-handoff"
            aria-label={t("editor.edit.handOff")}
            title={t("editor.edit.handOff.tip")}
            onMouseDown={(event) => event.preventDefault()}
            onClick={onHandOff}
            className="selection-edit-handoff"
          >
            <ChatLineIcon className="size-4" />
          </button>
        )}
      </div>
      {field.phase === "settled" && problem && (
        <EditProblemLine
          mindId={edit.mindId}
          problem={problem}
          className="selection-problem"
          onRetry={field.asked}
        />
      )}
    </>
  );
}

/** What takes the menu's place while the AI works on the change: one line, with Stop. */
export function EditWorking({ mindId }: { mindId: string }) {
  const t = useT();
  return (
    // biome-ignore lint/a11y/noStaticElementInteractions: it only keeps the cursor in the text when Stop is clicked.
    <div onMouseDown={(event) => event.preventDefault()} className="editor-menu selection-working">
      <ProblemLine
        role="status"
        testId="selection-edit-working"
        action={{
          label: t("edit.stop"),
          testId: "selection-edit-stop",
          onClick: () => stopEdit(mindId),
        }}
      >
        <span className="edit-pending-dot" aria-hidden="true" />
        {t("edit.pending.instruction")}
      </ProblemLine>
    </div>
  );
}
