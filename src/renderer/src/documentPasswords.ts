/**
 * Which PDF Documents have their password kept (#763), for the row's menu to
 * offer Forget password. Only their ids: a password never comes here.
 */
import { create } from "zustand";
import { core } from "./core";

export const useDocumentPasswords = create<{ kept: readonly string[] }>(() => ({ kept: [] }));

const keep = (kept: unknown) => {
  if (Array.isArray(kept)) useDocumentPasswords.setState({ kept });
};

core.on("documentPasswords.changed", keep);
Promise.resolve(core.listDocumentPasswords()).then(keep, (error: unknown) =>
  console.error("The Documents with a kept PDF password couldn't be listed.", error),
);
