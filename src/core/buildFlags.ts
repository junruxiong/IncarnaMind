/** Replaced at build time (electron.vite.config.ts); undefined where nothing builds the code, as in tests. */
declare const __EXPERIMENTAL_SIGN_IN_ROUTES__: boolean | undefined;

/**
 * Whether this build has the experimental sign-in routes. Built with it
 * false, their profiles aren't in the bundle at all.
 */
export const EXPERIMENTAL_SIGN_IN_ROUTES: boolean =
  typeof __EXPERIMENTAL_SIGN_IN_ROUTES__ === "boolean" ? __EXPERIMENTAL_SIGN_IN_ROUTES__ : true;
