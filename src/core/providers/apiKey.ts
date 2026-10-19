import { cleanApiKey, nameCharacter, strayKeyCharacter } from "../../shared/apiKey";
import { InvalidInputError } from "../errors";

/** A key holds a character that keys don't use, named by itself and never with the rest of the key. */
export class KeyCharacterError extends InvalidInputError {
  constructor(character: string) {
    super({
      key: "error.provider.keyCharacter",
      values: { character: nameCharacter(character) },
    });
  }
}

/** A key to store or send, cleaned; "" when blank. */
export function readApiKey(text: string): string {
  const key = cleanApiKey(text);
  const stray = strayKeyCharacter(key);
  if (stray) throw new KeyCharacterError(stray.character);
  return key;
}

/** undefined: keep the stored key. Blank text counts as "keep". */
export function parseApiKey(value: unknown): string | undefined {
  if (value === undefined) return undefined;
  if (typeof value !== "string") throw new InvalidInputError("The API key must be text.");
  return readApiKey(value) || undefined;
}
