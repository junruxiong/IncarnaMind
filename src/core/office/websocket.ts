/**
 * The few parts of RFC 6455 a task pane's WebSocket needs, so the spike adds no
 * dependency: the handshake, whole text frames up to 65,535 bytes, ping and close. A
 * client that fragments, sends binary or doesn't mask is closed.
 */
import { createHash } from "node:crypto";
import type { IncomingMessage } from "node:http";
import type { Duplex } from "node:stream";

const HANDSHAKE_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
export const MAX_FRAME_BYTES = 65_535;

const TEXT = 0x1;
const CLOSE = 0x8;
const PING = 0x9;
const PONG = 0xa;

export interface WebSocketPeer {
  send(text: string): void;
  close(code?: number): void;
  onText(listener: (text: string) => void): void;
  onClose(listener: () => void): void;
}

function frame(opcode: number, payload: Buffer): Buffer {
  const size = payload.length;
  const header =
    size < 126
      ? Buffer.from([0x80 | opcode, size])
      : Buffer.from([0x80 | opcode, 126, size >> 8, size & 0xff]);
  return Buffer.concat([header, payload]);
}

export function wantsWebSocket(request: IncomingMessage): boolean {
  return (
    request.headers.upgrade?.toLowerCase() === "websocket" &&
    request.headers["sec-websocket-version"] === "13" &&
    typeof request.headers["sec-websocket-key"] === "string"
  );
}

/** Completes the handshake on `socket`; the caller has already checked who is asking. */
export function acceptWebSocket(
  request: IncomingMessage,
  socket: Duplex,
  head: Buffer,
): WebSocketPeer {
  const key = String(request.headers["sec-websocket-key"]);
  const accept = createHash("sha1")
    .update(key + HANDSHAKE_GUID)
    .digest("base64");
  socket.write(
    [
      "HTTP/1.1 101 Switching Protocols",
      "Upgrade: websocket",
      "Connection: Upgrade",
      `Sec-WebSocket-Accept: ${accept}`,
      "",
      "",
    ].join("\r\n"),
  );

  let textListener: (text: string) => void = () => undefined;
  let closeListener: () => void = () => undefined;
  let closed = false;
  let buffered = Buffer.alloc(0);

  const close = (code = 1000) => {
    if (closed) return;
    closed = true;
    const payload = Buffer.alloc(2);
    payload.writeUInt16BE(code);
    socket.end(frame(CLOSE, payload));
    closeListener();
  };

  const consume = () => {
    while (!closed && buffered.length >= 2) {
      const first = buffered[0] ?? 0;
      const second = buffered[1] ?? 0;
      const opcode = first & 0x0f;
      let size = second & 0x7f;
      let offset = 2;
      if (size === 126) {
        if (buffered.length < 4) return;
        size = buffered.readUInt16BE(2);
        offset = 4;
      } else if (size === 127) {
        return close(1009);
      }
      if (size > MAX_FRAME_BYTES) return close(1009);
      if (!(second & 0x80)) return close(1002);
      if (buffered.length < offset + 4 + size) return;
      const mask = buffered.subarray(offset, offset + 4);
      const payload = Buffer.from(buffered.subarray(offset + 4, offset + 4 + size));
      for (let index = 0; index < payload.length; index += 1) {
        payload[index] = (payload[index] ?? 0) ^ (mask[index % 4] ?? 0);
      }
      buffered = buffered.subarray(offset + 4 + size);
      if (!(first & 0x80)) return close(1003);
      if (opcode === TEXT) textListener(payload.toString("utf8"));
      else if (opcode === CLOSE) return close(1000);
      else if (opcode === PING) socket.write(frame(PONG, payload));
      else if (opcode !== PONG) return close(1003);
    }
  };

  socket.on("data", (chunk: Buffer) => {
    buffered = Buffer.concat([buffered, chunk]);
    consume();
  });
  socket.on("close", () => {
    if (closed) return;
    closed = true;
    closeListener();
  });
  socket.on("error", () => socket.destroy());
  if (head.length > 0) {
    buffered = Buffer.from(head);
    consume();
  }

  return {
    send: (text) => {
      if (!closed) socket.write(frame(TEXT, Buffer.from(text, "utf8")));
    },
    close,
    onText: (listener) => {
      textListener = listener;
    },
    onClose: (listener) => {
      closeListener = listener;
    },
  };
}
