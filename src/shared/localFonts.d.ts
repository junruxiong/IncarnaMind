/** The Local Font Access API: Electron's Chromium has it, TypeScript's DOM types don't. */
interface LocalFont {
  readonly family: string;
  readonly style: string;
}

interface Window {
  queryLocalFonts(): Promise<LocalFont[]>;
}
