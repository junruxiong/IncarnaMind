import { nameCharacter } from "../../../shared/apiKey";
import { useT } from "../i18n";
import { ProblemMark } from "./ProblemLine";
import { errorTextClass } from "./ui";

export function KeyCharacterLine({ id, stray }: { id: string; stray: string }) {
  const t = useT();
  return (
    <span
      id={id}
      role="alert"
      data-testid="key-character-line"
      className={`${errorTextClass} mt-1 block`}
    >
      <ProblemMark />
      {t("error.provider.keyCharacter", { character: nameCharacter(stray) })}
    </span>
  );
}
