import { type ComponentProps, useLayoutEffect, useRef } from "react";

/**
 * A text area that shows `value` but leaves the DOM's text alone while the User
 * types. A controlled `<textarea>` makes React rewrite its default value on every
 * keystroke, which makes Chromium end the typing run, so Undo takes back one
 * character at a time instead of the run.
 */
export function TypingTextarea({
  value,
  ref,
  ...props
}: Omit<ComponentProps<"textarea">, "value" | "defaultValue"> & { value: string }) {
  const own = useRef<HTMLTextAreaElement | null>(null);
  useLayoutEffect(() => {
    if (own.current && own.current.value !== value) own.current.value = value;
  }, [value]);
  return (
    <textarea
      {...props}
      ref={(element) => {
        own.current = element;
        if (typeof ref === "function") ref(element);
        else if (ref) ref.current = element;
      }}
    />
  );
}
