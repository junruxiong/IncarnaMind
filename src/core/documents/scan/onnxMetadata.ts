/**
 * An ONNX model's metadata (`metadata_props`, field 14 of its `ModelProto`),
 * read straight from the file's protocol buffer: onnxruntime-node doesn't
 * give it. The recognition model keeps its character list there, under
 * "character" (#397). Only the top level of the message is walked; the
 * graph, which is most of the file, is skipped over whole.
 */

const METADATA_PROPS = 14;

/** A varint at `at`, and where the next field starts. Lengths here stay within 2^53. */
function varint(bytes: Uint8Array, at: number): [number, number] {
  let value = 0;
  let scale = 1;
  let index = at;
  for (;;) {
    const byte = bytes[index++];
    if (byte === undefined) throw new Error("The model file ends inside a number.");
    value += (byte & 0x7f) * scale;
    if (byte < 0x80) return [value, index];
    scale *= 128;
  }
}

/** The fields of one protocol-buffer message, by number: the length-delimited ones' bytes. */
function* fields(bytes: Uint8Array): Generator<[number, Uint8Array | null]> {
  let at = 0;
  while (at < bytes.length) {
    const [key, next] = varint(bytes, at);
    at = next;
    const field = Math.floor(key / 8);
    switch (key % 8) {
      case 0:
        at = varint(bytes, at)[1];
        yield [field, null];
        break;
      case 1:
        at += 8;
        yield [field, null];
        break;
      case 5:
        at += 4;
        yield [field, null];
        break;
      case 2: {
        const [length, start] = varint(bytes, at);
        at = start + length;
        if (at > bytes.length) throw new Error("The model file ends inside a field.");
        yield [field, bytes.subarray(start, at)];
        break;
      }
      default:
        throw new Error("The model file isn't an ONNX model.");
    }
  }
}

/** The model's metadata, key by value. */
export function onnxMetadata(model: Uint8Array): Record<string, string> {
  const metadata: Record<string, string> = {};
  const decoder = new TextDecoder();
  for (const [field, value] of fields(model)) {
    if (field !== METADATA_PROPS || !value) continue;
    let key = "";
    let text = "";
    for (const [entry, bytes] of fields(value)) {
      if (entry === 1 && bytes) key = decoder.decode(bytes);
      if (entry === 2 && bytes) text = decoder.decode(bytes);
    }
    metadata[key] = text;
  }
  return metadata;
}
