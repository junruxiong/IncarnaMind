import { type KeyboardEvent, useEffect, useId, useState } from "react";
import { useT } from "../i18n";
import { useAppStore } from "../store";
import { fieldClass, sectionNoteClass, sectionTitleClass } from "./ui";

/**
 * Settings → General → Your name (#107): the name the User's edits go under
 * when a Word file shows them as tracked changes, the name the computer knows
 * them by until they type another. Edited in place: Enter or leaving the field
 * saves, Esc puts it back, and an empty field goes back to the computer's.
 */
export function AuthorNameSettings() {
  const t = useT();
  const id = useId();
  const author = useAppStore((state) => state.settings?.author);
  const updateSettings = useAppStore((state) => state.updateSettings);
  const [draft, setDraft] = useState(author ?? "");
  useEffect(() => {
    if (author !== undefined) setDraft(author);
  }, [author]);
  if (author === undefined) return null;

  const commit = () => {
    const name = draft.trim();
    if (name === author) return;
    void updateSettings({ user: { authorName: name === "" ? null : name } });
  };
  const onKeyDown = (event: KeyboardEvent<HTMLInputElement>) => {
    if (event.key === "Enter") commit();
    if (event.key === "Escape" && draft !== author) {
      // Esc puts a changed name back, and doesn't close Settings as well.
      event.preventDefault();
      setDraft(author);
    }
  };

  return (
    <section data-testid="author-name-settings">
      <label htmlFor={id} className={sectionTitleClass}>
        {t("settings.authorName")}
      </label>
      <p id={`${id}-note`} className={sectionNoteClass}>
        {t("settings.authorName.body")}
      </p>
      <input
        id={id}
        type="text"
        data-testid="author-name"
        value={draft}
        maxLength={200}
        aria-describedby={`${id}-note`}
        onChange={(event) => setDraft(event.target.value)}
        onBlur={commit}
        onKeyDown={onKeyDown}
        className={`${fieldClass} w-full max-w-80`}
      />
    </section>
  );
}
