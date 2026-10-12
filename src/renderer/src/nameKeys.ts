/** What a key does in the field: never anything while an input method is composing (keyCode 229 is its Enter). */
export function nameKeyAction(event: {
  key: string;
  keyCode: number;
  isComposing: boolean;
}): "save" | "cancel" | null {
  if (event.isComposing || event.keyCode === 229) return null;
  return event.key === "Enter" ? "save" : event.key === "Escape" ? "cancel" : null;
}
