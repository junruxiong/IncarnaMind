import { nameCharacter, type StrayCharacter } from "../../../shared/apiKey";
import { useT } from "../i18n";
import { ProblemMark } from "./ProblemLine";
import { errorTextClass } from "./ui";

/** Under a key field: the one character keys don't use, and never any more of the key. */
export function KeyCharacterLine({ id, stray }: { id: string; stray: StrayCharacter }) {
  const t = useT();
  return (
    <span
      id={id}
      role="alert"
      data-testid="key-character-line"
      className={`${errorTextClass} mt-1 block`}
    >
      <ProblemMark />
      {t("error.provider.keyCharacter", { character: nameCharacter(stray.character) })}
    </span>
  );
}
