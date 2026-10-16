/** What counts as on this computer, and the service an address sends data to: the one rule, for Local mode and consent. */
import type { ExternalService } from "../api";

/** Loopback addresses: data sent there stays on this computer. */
export function isLoopbackHost(hostname: string): boolean {
  const host = hostname.toLowerCase().replace(/^\[|\]$/g, "");
  return (
    host === "localhost" ||
    host.endsWith(".localhost") ||
    host === "::1" ||
    host === "0.0.0.0" ||
    /^127(\.\d{1,3}){3}$/.test(host)
  );
}

/** The service at an address, by its origin: null when it is on this computer. Shared by Ollama, a server the User gives and the presets. */
export function serviceAt(address: string, name: string): ExternalService | null {
  const url = new URL(address);
  return isLoopbackHost(url.hostname) ? null : { id: url.origin, name };
}

/** The service a server URL sends data to, or null when the server runs on this computer. */
export const serviceForUrl = (baseUrl: string): ExternalService | null =>
  serviceAt(baseUrl, new URL(baseUrl).host);
