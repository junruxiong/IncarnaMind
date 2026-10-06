import type { SecretProtection } from "../../src/core";
import type { SecretCipher } from "../../src/main/secretsFile";

/**
 * A stand-in for Electron's safeStorage: XORs with a fixed key, so the
 * ciphertext never contains the secret. Like safeStorage on Linux's
 * "basic_text" backend, it refuses to encrypt until plain text is allowed.
 */
export function createFakeCipher(protection: SecretProtection = "os") {
  let plainTextAllowed = false;
  const xor = (bytes: Buffer) => Buffer.from(bytes.map((byte) => byte ^ 0x5a));
  const cipher: SecretCipher & { readonly plainTextAllowed: boolean } = {
    get plainTextAllowed() {
      return plainTextAllowed;
    },
    protection: () => protection,
    allowPlainText: () => {
      plainTextAllowed = true;
    },
    encrypt(plainText) {
      if (protection === "unavailable" || (protection === "plain-text" && !plainTextAllowed)) {
        throw new Error("Encryption is not available.");
      }
      return xor(Buffer.from(plainText, "utf8"));
    },
    decrypt: (cipherText) => xor(cipherText).toString("utf8"),
  };
  return cipher;
}
