/**
 * Which addresses the web fetcher may connect to (#128): public unicast ones
 * only, so a page can't make IncarnaMind reach this computer or the User's
 * local network. Intranet pages are out of scope for web search (#74).
 */
import { promises as dns, type LookupAddress } from "node:dns";
import { isIP, type LookupFunction } from "node:net";
import ipaddr from "ipaddr.js";

/** IPv6's global unicast space (RFC 4291): an address on the internet is in it. */
const GLOBAL_UNICAST = ipaddr.IPv6.parseCIDR("2000::/3");

/**
 * Whether an IP address is on the public internet: unicast, outside every
 * special-purpose range ipaddr.js knows (loopback, private, link-local,
 * carrier-grade NAT, documentation, benchmarking, 6to4, Teredo, NAT64…), and
 * for IPv6 in the global unicast space. An IPv4-mapped IPv6 address is
 * refused whatever it maps to. Anything that isn't an IP address is refused.
 */
export function isPublicAddress(address: string): boolean {
  const bare = address.replace(/^\[(.*)\]$/, "$1");
  if (isIP(bare) === 0) return false;
  const parsed = ipaddr.parse(bare);
  if (parsed.range() !== "unicast") return false;
  return parsed.kind() === "ipv4" || parsed.match(GLOBAL_UNICAST);
}

/** A host's addresses, as the system's resolver gives them. */
export type Resolve = (hostname: string) => Promise<readonly LookupAddress[]>;

export const systemResolve: Resolve = (hostname) =>
  dns.lookup(hostname, { all: true, verbatim: true });

/** A connection the guard refused: the host is, or resolves to, an address that isn't public. */
export class NotPublicError extends Error {
  override name = "NotPublicError";
  readonly code = "ENOTPUBLIC";
  constructor(readonly host: string) {
    super(`${host} is not a public address on the internet.`);
  }
}

/**
 * A host's addresses, if every one is public. A host written as an IP address
 * is checked as it is. Rejects with `NotPublicError` if any isn't public.
 */
export async function publicAddresses(host: string, resolve: Resolve): Promise<LookupAddress[]> {
  const bare = host.replace(/^\[(.*)\]$/, "$1");
  const family = isIP(bare);
  const addresses = family === 0 ? await resolve(bare) : [{ address: bare, family }];
  if (addresses.length === 0 || !addresses.every((each) => isPublicAddress(each.address))) {
    throw new NotPublicError(host);
  }
  return [...addresses];
}

/**
 * A `lookup` for `net.connect` and `tls.connect` that gives only public
 * addresses. It runs as each connection is made, and the connection goes to
 * the addresses it checked, so a name that resolved to a public address
 * earlier and resolves to a local one now (DNS rebinding) is refused.
 */
export function publicOnlyLookup(resolve: Resolve): LookupFunction {
  return (hostname, options, callback) => {
    publicAddresses(hostname, resolve).then(
      (addresses) => {
        if (options.all) {
          callback(null, addresses);
          return;
        }
        const family =
          options.family === "IPv4" ? 4 : options.family === "IPv6" ? 6 : options.family;
        const wanted = addresses.find((each) => !family || each.family === family) ?? addresses[0];
        if (wanted) callback(null, wanted.address, wanted.family);
        else callback(new NotPublicError(hostname), "");
      },
      (error: unknown) => callback(error instanceof Error ? error : new Error(String(error)), ""),
    );
  };
}
