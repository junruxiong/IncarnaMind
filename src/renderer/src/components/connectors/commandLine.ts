/** A command line as the User would type it: arguments with spaces in quotes. */
export function commandLine(command: string, args: readonly string[]): string {
  return [command, ...args]
    .map((part) => (/\s|^$/.test(part) ? JSON.stringify(part) : part))
    .join(" ");
}
