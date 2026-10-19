import { statSync } from "node:fs";
import { isAbsolute, resolve } from "node:path";

/**
 * The files a second launch was asked to open: the arguments that are the path of a file, made
 * absolute from where that launch ran. Switches, the app's own folder and anything that isn't
 * there are left out.
 */
export function filesAskedFor(argv: readonly string[], workingDirectory: string): string[] {
  return argv.slice(1).flatMap((argument) => {
    if (argument.startsWith("-")) return [];
    const path = isAbsolute(argument) ? argument : resolve(workingDirectory, argument);
    try {
      return statSync(path).isFile() ? [path] : [];
    } catch {
      // An argument that names no file: a switch's value, or a file that has gone.
      return [];
    }
  });
}
