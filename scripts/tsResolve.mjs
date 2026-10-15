// Node runs the catalog script's TypeScript itself but, unlike the app's bundler, doesn't
// resolve a relative import written without its extension, as the catalog's modules are.
import { registerHooks } from "node:module";

registerHooks({
  resolve(specifier, context, nextResolve) {
    try {
      return nextResolve(specifier, context);
    } catch (error) {
      const relative = specifier.startsWith("./") || specifier.startsWith("../");
      if (error?.code !== "ERR_MODULE_NOT_FOUND" || !relative || /\.[cm]?[jt]s$/.test(specifier)) {
        throw error;
      }
      return nextResolve(`${specifier}.ts`, context);
    }
  },
});
