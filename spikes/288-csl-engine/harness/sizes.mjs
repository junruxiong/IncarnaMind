// Sizes of a wasm build: raw, gzip -9, brotli; and how much is code vs data.
// usage: node sizes.mjs <file.wasm>...
import fs from "node:fs";
import zlib from "node:zlib";

const NAMES = { 0: "custom", 1: "type", 2: "import", 3: "function", 5: "memory", 6: "global", 7: "export", 10: "code", 11: "data" };
for (const file of process.argv.slice(2)) {
  const buf = fs.readFileSync(file);
  const sections = {};
  let p = 8;
  const leb = () => {
    let r = 0;
    let s = 0;
    for (;;) {
      const b = buf[p++];
      r |= (b & 0x7f) << s;
      if (!(b & 0x80)) return r >>> 0;
      s += 7;
    }
  };
  while (p < buf.length) {
    const id = buf[p++];
    const len = leb();
    sections[NAMES[id] ?? id] = (sections[NAMES[id] ?? id] ?? 0) + len;
    p += len;
  }
  const gz = zlib.gzipSync(buf, { level: 9 }).length;
  const br = zlib.brotliCompressSync(buf, { params: { [zlib.constants.BROTLI_PARAM_QUALITY]: 11 } }).length;
  console.log(JSON.stringify({ file, raw: buf.length, gzip9: gz, brotli11: br, code: sections.code, data: sections.data, custom: sections.custom ?? 0 }));
}
