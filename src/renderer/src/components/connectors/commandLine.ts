/** A command line as the User would type it: arguments with spaces in quotes. */
export function commandLine(command: string, args: readonly string[]): string {
  return [command, ...args]
    .map((part) => (/\s|^$/.test(part) ? JSON.stringify(part) : part))
    .join(" ");
}

/**
 * Splits a command line as a terminal would: at spaces, except inside "…" or
 * '…' or after a backslash, which are taken off. The inverse of `commandLine`.
 */
export function splitCommandLine(text: string): string[] {
  const parts: string[] = [];
  let current = "";
  let started = false;
  let quote: '"' | "'" | null = null;
  for (let index = 0; index < text.length; index++) {
    const char = text[index] as string;
    if (quote) {
      if (char === quote) quote = null;
      else if (char === "\\" && quote === '"' && index + 1 < text.length) {
        index++;
        current += text[index];
      } else current += char;
    } else if (char === '"' || char === "'") {
      quote = char;
      started = true;
    } else if (char === "\\" && index + 1 < text.length) {
      index++;
      current += text[index];
      started = true;
    } else if (/\s/.test(char)) {
      if (started) parts.push(current);
      current = "";
      started = false;
    } else {
      current += char;
      started = true;
    }
  }
  if (started) parts.push(current);
  return parts;
}
