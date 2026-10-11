/**
 * The bake-off's command line (arm C): `incarnamind search|read|cite|check|ask`,
 * a thin layer over the same socket and the same MCP Tools the server gives
 * an agent (#351), so the CLI and the MCP server can't drift apart. Each
 * command connects, says which pairing it is, calls one Tool and prints its
 * result as JSON. Plain Node with no imports outside it, so an agent's shell
 * can run it from any folder.
 */
import { appendFileSync } from "node:fs";
import { createConnection } from "node:net";

const socketPath = process.env.INCARNAMIND_AGENT_SOCKET;
const token = process.env.INCARNAMIND_AGENT_TOKEN;
const logFile = process.env.INCARNAMIND_BAKEOFF_LOG;
/** The Tool each verb calls, as the server names them. */
const TOOL_NAMES: Record<string, string> = {
  search: "search",
  read: "read_passages",
  cite: "cite",
  check: "check_citations",
  ask: "ask",
};
/** The Tools this pairing's CLI offers (the bake-off's arm C leaves out `ask`); all when empty. */
const allowed = (process.env.INCARNAMIND_BAKEOFF_TOOLS ?? "")
  .split(",")
  .map((each) => each.trim())
  .filter(Boolean);

/** The same words as the MCP Tools' descriptions (src/core/agents/mcp.ts), so the routes differ only in transport. */
const VERBS: { verb: string; usage: string[]; text: string }[] = [
  {
    verb: "search",
    usage: ['incarnamind search "<words to look for>" [--page N]'],
    text: "Search the User's Documents. Returns the best Passages, each with its Location and a ready Citation (`cite`) of its best sentence (`quote`). Use the words a Document would use; ask for page 2 for more.",
  },
  {
    verb: "read",
    usage: [
      "incarnamind read <passageId> [--before N] [--after N]",
      "incarnamind read --document <documentId> [--after N]",
    ],
    text: "Read a Passage with up to 3 before and after it, or a Document from its start. Never a whole Document: read on with `previous` or `next`.",
  },
  {
    verb: "cite",
    usage: ['incarnamind cite <passageId> "<quote>"'],
    text: "Get a Citation for a quote, copied word for word from a Passage. A quote found gets `cite` to write after the claim; one not found gets none, and the Passage's text to copy from.",
  },
  {
    verb: "check",
    usage: ['incarnamind check "<draft>"   (or "-" to read the draft from stdin)'],
    text: "Check a draft before you finish: each [cite:…] in it is checked again against its Document, and sentences with none are listed.",
  },
  {
    verb: "ask",
    usage: ['incarnamind ask "<question>"'],
    text: "Ask a question about the User's Documents. IncarnaMind searches, answers on the User's own chat model and checks every Citation. Pass the answer on with its [cite:…] tokens as they are.",
  },
];

const offered = (verb: string) =>
  allowed.length === 0 || allowed.includes(TOOL_NAMES[verb] ?? verb);

function help(): string {
  const entries = VERBS.filter(({ verb }) => offered(verb)).map(({ usage, text }) =>
    [...usage.map((line) => `  ${line}`), `      ${text}`].join("\n"),
  );
  return `IncarnaMind: the User's own Documents, limited to the Folders the User shared with you.\n\n${entries.join("\n")}\n\nEvery command prints JSON.`;
}

function fail(message: string, code = 2): never {
  process.stderr.write(`${message}\n`);
  process.exit(code);
}

function option(args: string[], name: string): number | undefined {
  const index = args.indexOf(name);
  if (index < 0) return undefined;
  const value = Number(args[index + 1]);
  args.splice(index, 2);
  if (!Number.isInteger(value)) fail(`${name} takes a whole number.`);
  return value;
}

async function readStdin(): Promise<string> {
  let text = "";
  process.stdin.setEncoding("utf8");
  for await (const chunk of process.stdin) text += chunk;
  return text;
}

async function argumentsFor(verb: string, args: string[]): Promise<Record<string, unknown>> {
  switch (verb) {
    case "search": {
      const page = option(args, "--page");
      const query = args.join(" ").trim();
      if (!query) fail('Say what to look for: incarnamind search "<words>"');
      return { query, ...(page !== undefined && { page }) };
    }
    case "read": {
      const before = option(args, "--before");
      const after = option(args, "--after");
      const documentAt = args.indexOf("--document");
      const read: Record<string, unknown> = {
        ...(before !== undefined && { before }),
        ...(after !== undefined && { after }),
      };
      if (documentAt >= 0) return { ...read, documentId: args[documentAt + 1] };
      if (!args[0]) fail("Say which Passage to read: incarnamind read <passageId>");
      return { ...read, passageId: args[0] };
    }
    case "cite": {
      const [passageId, ...quote] = args;
      if (!passageId || quote.length === 0) fail('incarnamind cite <passageId> "<quote>"');
      return { citations: [{ passageId, quote: quote.join(" ") }] };
    }
    case "check": {
      const text = args[0] === "-" || args.length === 0 ? await readStdin() : args.join(" ");
      return { text };
    }
    case "ask": {
      const question = args.join(" ").trim();
      if (!question) fail('Ask a question: incarnamind ask "<question>"');
      return { question };
    }
    default:
      return fail(help());
  }
}

interface Reply {
  id?: number;
  result?: {
    structuredContent?: unknown;
    content?: { type: string; text?: string }[];
    isError?: boolean;
  };
  error?: { message?: string };
}

async function call(tool: string, input: Record<string, unknown>): Promise<Reply> {
  if (!socketPath || !token) {
    fail(
      "IncarnaMind isn't set up here: INCARNAMIND_AGENT_SOCKET and INCARNAMIND_AGENT_TOKEN are missing.",
    );
  }
  const socket = createConnection(socketPath);
  const replies = new Map<number, (reply: Reply) => void>();
  let buffer = "";
  socket.setEncoding("utf8");
  socket.on("data", (chunk: string) => {
    buffer += chunk;
    for (let end = buffer.indexOf("\n"); end >= 0; end = buffer.indexOf("\n")) {
      const line = buffer.slice(0, end);
      buffer = buffer.slice(end + 1);
      try {
        const reply = JSON.parse(line) as Reply;
        if (typeof reply.id === "number") replies.get(reply.id)?.(reply);
      } catch {
        // Not a reply to us.
      }
    }
  });
  await new Promise<void>((resolve, reject) => {
    socket.once("connect", resolve);
    socket.once("error", reject);
  }).catch((error: Error) =>
    fail(`IncarnaMind isn't running, or can't be reached: ${error.message}`, 1),
  );
  let next = 1;
  const request = (method: string, params: unknown) =>
    new Promise<Reply>((resolve) => {
      const id = next++;
      replies.set(id, resolve);
      socket.write(`${JSON.stringify({ jsonrpc: "2.0", id, method, params })}\n`);
    });
  socket.write(`${JSON.stringify({ incarnamind: 1, token })}\n`);
  await request("initialize", {
    protocolVersion: "2025-11-25",
    capabilities: {},
    clientInfo: { name: "incarnamind-cli", version: "0.0.0" },
  });
  socket.write(`${JSON.stringify({ jsonrpc: "2.0", method: "notifications/initialized" })}\n`);
  const started = performance.now();
  const reply = await request("tools/call", { name: tool, arguments: input });
  if (logFile) {
    try {
      appendFileSync(
        logFile,
        `${JSON.stringify({ method: "tools/call", tool, ms: Number((performance.now() - started).toFixed(3)), via: "cli", at: Date.now() })}\n`,
      );
    } catch {
      // Timing is a report only.
    }
  }
  socket.end();
  return reply;
}

async function main(): Promise<void> {
  const [verb, ...args] = process.argv.slice(2);
  if (!verb || verb === "--help" || verb === "-h" || verb === "help") {
    process.stdout.write(`${help()}\n`);
    return;
  }
  const tool = TOOL_NAMES[verb];
  if (!tool || !offered(verb)) fail(help());
  // Every command prints JSON; `--json` is accepted for agents that add it.
  const rest = args.filter((each) => each !== "--json");
  const reply = await call(tool, await argumentsFor(verb, rest));
  if (reply.error) fail(reply.error.message ?? "IncarnaMind refused the request.", 1);
  const result = reply.result;
  const output =
    result?.structuredContent ??
    (result?.content ?? [])
      .map((part) => part.text ?? "")
      .join("\n")
      .trim();
  process.stdout.write(
    `${typeof output === "string" ? output : JSON.stringify(output, null, 1)}\n`,
  );
  if (result?.isError) process.exit(1);
}

main().catch((error: unknown) => fail(error instanceof Error ? error.message : String(error), 1));
