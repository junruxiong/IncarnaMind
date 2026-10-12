/** The longest Skill name the SKILL.md format allows. */
export const MAX_SKILL_NAME_LENGTH = 64;

/**
 * A name in the format's characters: lowercase letters and digits in runs
 * joined by hyphens, as "Supplier contract review" becomes
 * "supplier-contract-review". Empty if nothing of it is left (a name in
 * Chinese alone, say). The editor shows this before the core writes it.
 */
export function skillNameFrom(text: string): string {
  return text
    .normalize("NFKD")
    .replace(/\p{M}/gu, "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, MAX_SKILL_NAME_LENGTH)
    .replace(/-+$/, "");
}
