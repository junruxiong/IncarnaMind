/**
 * The certificate the app makes for itself, so an Office task pane served from
 * `https://localhost` loads without a warning once the User trusts it. Two
 * certificates: a local authority the User trusts once, which may only vouch
 * for `localhost`, `127.0.0.1` and `::1` (name constraints, so a stolen key
 * can't impersonate any other site), and a short-lived server certificate it
 * signs, made again at each start so the User isn't asked again. Only
 * `node:crypto`: Node has no X.509 writer, so the few ASN.1 pieces are built here.
 */
import {
  createHash,
  createPrivateKey,
  createPublicKey,
  generateKeyPairSync,
  type KeyObject,
  randomBytes,
  sign,
} from "node:crypto";

export interface LocalAuthority {
  certificatePem: string;
  keyPem: string;
}

export interface LoopbackCertificate {
  certificatePem: string;
  keyPem: string;
  notAfter: Date;
}

export const LOOPBACK_DNS_NAMES = ["localhost"] as const;
export const LOOPBACK_ADDRESSES = ["127.0.0.1", "::1"] as const;

/** Apple's limit for a server certificate is 825 days; shorter is safer and costs the User nothing. */
const SERVER_CERTIFICATE_DAYS = 90;
const AUTHORITY_YEARS = 10;
const DAY_MS = 86_400_000;
const BACKDATE_MS = DAY_MS;

const OID = {
  commonName: "2.5.4.3",
  organization: "2.5.4.10",
  ecdsaWithSha256: "1.2.840.10045.4.3.2",
  subjectKeyIdentifier: "2.5.29.14",
  keyUsage: "2.5.29.15",
  subjectAltName: "2.5.29.17",
  basicConstraints: "2.5.29.19",
  nameConstraints: "2.5.29.30",
  authorityKeyIdentifier: "2.5.29.35",
  extendedKeyUsage: "2.5.29.37",
  serverAuth: "1.3.6.1.5.5.7.3.1",
} as const;

function lengthBytes(length: number): Buffer {
  if (length < 0x80) return Buffer.from([length]);
  const bytes: number[] = [];
  for (let rest = length; rest > 0; rest = Math.floor(rest / 256)) bytes.unshift(rest % 256);
  return Buffer.from([0x80 | bytes.length, ...bytes]);
}

function der(tag: number, ...parts: Buffer[]): Buffer {
  const body = Buffer.concat(parts);
  return Buffer.concat([Buffer.from([tag]), lengthBytes(body.length), body]);
}

const sequence = (...parts: Buffer[]) => der(0x30, ...parts);
const set = (...parts: Buffer[]) => der(0x31, ...parts);
const octets = (bytes: Buffer) => der(0x04, bytes);
const boolean = (value: boolean) => der(0x01, Buffer.from([value ? 0xff : 0x00]));
const utf8 = (text: string) => der(0x0c, Buffer.from(text, "utf8"));
const bitString = (bytes: Buffer, unusedBits = 0) => der(0x03, Buffer.from([unusedBits]), bytes);

function integer(bytes: Buffer): Buffer {
  const first = bytes[0] ?? 0;
  return der(0x02, first & 0x80 ? Buffer.concat([Buffer.from([0]), bytes]) : bytes);
}

function objectId(dotted: string): Buffer {
  const [first = 0, second = 0, ...rest] = dotted.split(".").map(Number);
  const bytes = [first * 40 + second];
  for (const arc of rest) {
    const group = [arc % 128];
    for (
      let remaining = Math.floor(arc / 128);
      remaining > 0;
      remaining = Math.floor(remaining / 128)
    ) {
      group.unshift((remaining % 128) | 0x80);
    }
    bytes.push(...group);
  }
  return der(0x06, Buffer.from(bytes));
}

function utcTime(date: Date): Buffer {
  const pad = (value: number) => String(value).padStart(2, "0");
  const text = `${pad(date.getUTCFullYear() % 100)}${pad(date.getUTCMonth() + 1)}${pad(date.getUTCDate())}${pad(date.getUTCHours())}${pad(date.getUTCMinutes())}${pad(date.getUTCSeconds())}Z`;
  return der(0x17, Buffer.from(text, "ascii"));
}

function name(commonName: string): Buffer {
  const attribute = (oid: string, value: string) => set(sequence(objectId(oid), utf8(value)));
  return sequence(
    attribute(OID.organization, "IncarnaMind"),
    attribute(OID.commonName, commonName),
  );
}

function extension(oid: string, critical: boolean, value: Buffer): Buffer {
  return sequence(objectId(oid), ...(critical ? [boolean(true)] : []), octets(value));
}

function ipBytes(address: string): Buffer {
  if (address.includes(":")) {
    if (address !== "::1") throw new Error("Only ::1 is supported as an IPv6 address.");
    return Buffer.concat([Buffer.alloc(15), Buffer.from([1])]);
  }
  return Buffer.from(address.split(".").map(Number));
}

const dnsName = (value: string) => der(0x82, Buffer.from(value, "ascii"));
const ipAddress = (bytes: Buffer) => der(0x87, bytes);

function nameConstraints(): Buffer {
  const subtree = (generalName: Buffer) => sequence(generalName);
  return sequence(
    der(
      0xa0,
      ...LOOPBACK_DNS_NAMES.map((value) => subtree(dnsName(value))),
      ...LOOPBACK_ADDRESSES.map((address) => {
        const bytes = ipBytes(address);
        return subtree(ipAddress(Buffer.concat([bytes, Buffer.alloc(bytes.length, 0xff)])));
      }),
    ),
  );
}

/** The SHA-1 of the key's bits, as RFC 5280 names the key in its identifier extensions. */
function keyIdentifier(publicKey: KeyObject): Buffer {
  const spki = publicKey.export({ type: "spki", format: "der" });
  // A P-256 key's SubjectPublicKeyInfo is a fixed 26-byte header and the 65-byte point.
  return createHash("sha1")
    .update(spki.subarray(spki.length - 65))
    .digest();
}

function pem(label: string, derBytes: Buffer): string {
  const lines = derBytes.toString("base64").match(/.{1,64}/g) ?? [];
  return `-----BEGIN ${label}-----\n${lines.join("\n")}\n-----END ${label}-----\n`;
}

interface CertificateParts {
  subject: string;
  issuer: string;
  issuerKey: KeyObject;
  subjectKey: KeyObject;
  notBefore: Date;
  notAfter: Date;
  extensions: Buffer[];
}

function certificate(parts: CertificateParts): string {
  const algorithm = sequence(objectId(OID.ecdsaWithSha256));
  const serial = randomBytes(16);
  serial[0] = (serial[0] ?? 0) & 0x7f;
  const toBeSigned = sequence(
    der(0xa0, integer(Buffer.from([2]))),
    integer(serial),
    algorithm,
    name(parts.issuer),
    sequence(utcTime(parts.notBefore), utcTime(parts.notAfter)),
    name(parts.subject),
    Buffer.from(parts.subjectKey.export({ type: "spki", format: "der" })),
    der(0xa3, sequence(...parts.extensions)),
  );
  const signature = sign("sha256", toBeSigned, { key: parts.issuerKey, dsaEncoding: "der" });
  return pem("CERTIFICATE", sequence(toBeSigned, algorithm, bitString(signature)));
}

const AUTHORITY_NAME = "IncarnaMind local add-in authority";
const SERVER_NAME = "IncarnaMind local add-in";

export function createAuthority(now: Date): LocalAuthority {
  const { privateKey, publicKey } = generateKeyPairSync("ec", { namedCurve: "P-256" });
  const notAfter = new Date(now.getTime());
  notAfter.setUTCFullYear(notAfter.getUTCFullYear() + AUTHORITY_YEARS);
  const certificatePem = certificate({
    subject: AUTHORITY_NAME,
    issuer: AUTHORITY_NAME,
    issuerKey: privateKey,
    subjectKey: publicKey,
    notBefore: new Date(now.getTime() - BACKDATE_MS),
    notAfter,
    extensions: [
      extension(OID.basicConstraints, true, sequence(boolean(true), integer(Buffer.from([0])))),
      // keyCertSign (bit 5) and cRLSign (bit 6).
      extension(OID.keyUsage, true, bitString(Buffer.from([0x06]), 1)),
      extension(OID.nameConstraints, true, nameConstraints()),
      extension(OID.subjectKeyIdentifier, false, octets(keyIdentifier(publicKey))),
    ],
  });
  return {
    certificatePem,
    keyPem: String(privateKey.export({ type: "pkcs8", format: "pem" })),
  };
}

export function issueLoopbackCertificate(
  authority: LocalAuthority,
  now: Date,
  names: { dns: readonly string[]; addresses: readonly string[] } = {
    dns: LOOPBACK_DNS_NAMES,
    addresses: LOOPBACK_ADDRESSES,
  },
): LoopbackCertificate {
  const issuerKey = createPrivateKey(authority.keyPem);
  const authorityKeyId = keyIdentifier(createPublicKey(issuerKey));
  const { privateKey, publicKey } = generateKeyPairSync("ec", { namedCurve: "P-256" });
  const notAfter = new Date(now.getTime() + SERVER_CERTIFICATE_DAYS * DAY_MS);
  const certificatePem = certificate({
    subject: SERVER_NAME,
    issuer: AUTHORITY_NAME,
    issuerKey,
    subjectKey: publicKey,
    notBefore: new Date(now.getTime() - BACKDATE_MS),
    notAfter,
    extensions: [
      extension(OID.basicConstraints, true, sequence()),
      // digitalSignature (bit 0): all an ECDHE key exchange needs.
      extension(OID.keyUsage, true, bitString(Buffer.from([0x80]), 7)),
      extension(OID.extendedKeyUsage, false, sequence(objectId(OID.serverAuth))),
      extension(
        OID.subjectAltName,
        false,
        sequence(
          ...names.dns.map(dnsName),
          ...names.addresses.map((address) => ipAddress(ipBytes(address))),
        ),
      ),
      extension(OID.subjectKeyIdentifier, false, octets(keyIdentifier(publicKey))),
      extension(OID.authorityKeyIdentifier, false, sequence(der(0x80, authorityKeyId))),
    ],
  });
  return {
    certificatePem,
    keyPem: String(privateKey.export({ type: "pkcs8", format: "pem" })),
    notAfter,
  };
}
