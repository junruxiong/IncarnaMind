import { describe, expect, test } from "vitest";
import { SecretStorageError } from "../../src/core";
import { createMemoryKeychain, createTempDataFolder, startCore } from "../helpers/core";

const OPENAI = { kind: "openai", apiKey: "sk-test-openai", modelId: "gpt-5.4-mini" } as const;

describe("Secret storage", () => {
  test("with the OS keychain, API keys can be saved", async () => {
    const core = startCore(await createTempDataFolder());

    expect(await core.getSecretStorage()).toEqual({
      protection: "os",
      plainTextAccepted: false,
      canSave: true,
    });
  });

  test("on Linux without a keyring, keys are refused until the User accepts plain-text storage", async () => {
    // safeStorage's "basic_text" backend: GNOME Keyring and KWallet aren't running.
    const keychain = createMemoryKeychain("plain-text");
    const core = startCore(await createTempDataFolder(), { keychain });

    expect(await core.getSecretStorage()).toEqual({
      protection: "plain-text",
      plainTextAccepted: false,
      canSave: false,
    });
    await expect(core.saveChatProvider(OPENAI)).rejects.toThrow(SecretStorageError);
    expect(keychain.secrets.size).toBe(0);
    expect(await core.listChatProviders()).toEqual([]);
    expect(await core.getChatReadiness()).toEqual({ ready: false, reason: "no-provider" });

    expect(await core.acceptPlainTextSecretStorage()).toEqual({
      protection: "plain-text",
      plainTextAccepted: true,
      canSave: true,
    });
    await core.saveChatProvider(OPENAI);

    expect([...keychain.secrets.values()]).toEqual(["sk-test-openai"]);
  });

  test("accepting plain-text storage is remembered on this device", async () => {
    const dataDir = await createTempDataFolder();
    const before = startCore(dataDir, { keychain: createMemoryKeychain("plain-text") });
    await before.acceptPlainTextSecretStorage();
    before.close();

    const keychain = createMemoryKeychain("plain-text");
    const after = startCore(dataDir, { keychain });

    expect(await after.getSecretStorage()).toMatchObject({
      plainTextAccepted: true,
      canSave: true,
    });
    await after.saveChatProvider(OPENAI);
    expect(keychain.secrets.size).toBe(1);
    // It's kept for the core itself: it isn't one of the settings the UI can change.
    expect((await after.getSettings()).device).not.toHaveProperty("plainTextSecretsAccepted");
  });

  test("providers without a key work without a keyring", async () => {
    const core = startCore(await createTempDataFolder(), {
      keychain: createMemoryKeychain("plain-text"),
    });

    await core.saveChatProvider({ kind: "ollama", modelId: "qwen3:4b" });

    expect(await core.getChatReadiness()).toMatchObject({ ready: true });
  });

  test("when encryption isn't available at all, keys can't be saved even after accepting", async () => {
    const core = startCore(await createTempDataFolder(), {
      keychain: createMemoryKeychain("unavailable"),
    });

    await core.acceptPlainTextSecretStorage();

    expect(await core.getSecretStorage()).toMatchObject({ canSave: false });
    await expect(core.saveChatProvider(OPENAI)).rejects.toThrow(SecretStorageError);
  });
});
