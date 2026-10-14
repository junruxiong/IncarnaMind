/** A top-level section of electron-builder.yml: its key's line and the indented lines under it. */
export function section(yaml: string, key: string): string {
  const lines = yaml.split(/\r?\n/);
  const start = lines.indexOf(`${key}:`);
  if (start < 0) return "";
  const end = lines.findIndex((line, index) => index > start && /^\S/.test(line));
  return lines.slice(start, end < 0 ? undefined : end).join("\n");
}
