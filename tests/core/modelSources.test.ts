import { existsSync } from "node:fs";
import { mkdtemp, readFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, test } from "vitest";
import type { ModelSourceChoice } from "../../src/core";
import { downloadModel, ModelDownloadError } from "../../src/core/embedding/download";
import {
  createModelSources,
  type ModelCopies,
  PROBE_KEPT_MS,
} from "../../src/core/embedding/modelSources";
import type { Language } from "../../src/core/language";
import { LOCAL_FETCH } from "../helpers/core";
import { type ModelServer, startModelServer } from "../helpers/embedding";

const FOLDER = "multilingual-e5-small";
const PINNED = "https://huggingface.co/intfloat/multilingual-e5-small/resolve/abc123/";
const FILE = { path: "onnx/model.onnx", size: 10, sha256: "0".repeat(64) };
const COPIES: ModelCopies = {
  modelscope: "https://www.modelscope.cn/models/IncarnaMind/{folder}/resolve/master/",
  r2: "https://models.example.com/{folder}/",
};

function probing(answers: { huggingface: boolean; modelscope: boolean }) {
  const asked: string[] = [];
  const fetcher: typeof fetch = (input) => {
    const url = new URL(input instanceof Request ? input.url : String(input));
    asked.push(url.host);
    const answered =
      (url.host === "huggingface.co" && answers.huggingface) ||
      (url.host === "www.modelscope.cn" && answers.modelscope);
    return answered
      ? Promise.resolve(new Response(null, { status: 200 }))
      : Promise.reject(new TypeError(`${url.host} can't be reached`));
  };
  return { fetcher, asked };
}

function sourcesWith(
  fetcher: typeof fetch,
  {
    copies = COPIES,
    language = "en",
    choice = "auto",
    clock = { now: 0 },
    proxy = { value: "DIRECT" },
    stallMs,
  }: {
    copies?: ModelCopies;
    language?: Language;
    choice?: ModelSourceChoice;
    clock?: { now: number };
    proxy?: { value: string };
    stallMs?: number;
  } = {},
) {
  return createModelSources({
    fetch: fetcher,
    now: () => clock.now,
    language: () => language,
    choice: () => choice,
    proxy: async () => proxy.value,
    copies,
    ...(stallMs !== undefined && { stallMs }),
  });
}

const order = async (sources: ReturnType<typeof sourcesWith>) =>
  (await sources.urlsFor(FOLDER, FILE, PINNED)).map((each) => each.source);

describe("Where a built-in model's files come from (#1578)", () => {
  // protects=outside mainland China Hugging Face leads, then IncarnaMind's copy on ModelScope, hf-mirror.com and IncarnaMind's copy on R2, each at the pinned file's own path; fails_when=the order differs, a URL leaves the pinned revision or path, or a copy's URL misses its folder; no_claim=whether the hosts serve the files, which the download tests cover.
  test("Hugging Face leads when it answers first, then the copies and the mirror in turn", async () => {
    const { fetcher } = probing({ huggingface: true, modelscope: false });
    const urls = await sourcesWith(fetcher).urlsFor(FOLDER, FILE, PINNED);
    expect(urls.map(({ source, url }) => [source, url.href])).toEqual([
      ["huggingface", `${PINNED}onnx/model.onnx`],
      [
        "modelscope",
        "https://www.modelscope.cn/models/IncarnaMind/multilingual-e5-small/resolve/master/onnx/model.onnx",
      ],
      [
        "hf-mirror",
        "https://hf-mirror.com/intfloat/multilingual-e5-small/resolve/abc123/onnx/model.onnx",
      ],
      ["r2", "https://models.example.com/multilingual-e5-small/onnx/model.onnx"],
    ]);
  });

  // protects=when ModelScope answers the probe first (mainland China), it leads, and hf-mirror comes before Hugging Face; fails_when=Hugging Face still leads, so every file waits on a blocked host first; no_claim=the probe's timing on a real network.
  test("ModelScope leads when it answers first, and the mirror comes before Hugging Face", async () => {
    const { fetcher } = probing({ huggingface: false, modelscope: true });
    expect(await order(sourcesWith(fetcher))).toEqual([
      "modelscope",
      "hf-mirror",
      "huggingface",
      "r2",
    ]);
  });

  // protects=when neither host answers the probe, the interface language is the tie-break: Chinese takes mainland China's order; fails_when=a failed probe always picks one order, or the language decides when a probe answered; no_claim=which order is right for a User abroad who reads Chinese, which only the probe can tell.
  test("when neither answers, the interface language decides", async () => {
    const { fetcher } = probing({ huggingface: false, modelscope: false });
    expect((await order(sourcesWith(fetcher, { language: "zh-CN" })))[0]).toBe("modelscope");
    expect((await order(sourcesWith(fetcher, { language: "en" })))[0]).toBe("huggingface");
    const answered = probing({ huggingface: true, modelscope: false });
    expect((await order(sourcesWith(answered.fetcher, { language: "zh-CN" })))[0]).toBe(
      "huggingface",
    );
  });

  // protects=the probe runs once for many files: its answer holds 10 minutes, and a change of the system proxy asks again; fails_when=each file probes, the answer never expires, or a proxy change keeps the old order; no_claim=how the host reads its proxy.
  test("the probe's answer holds ten minutes, and a new proxy asks again", async () => {
    const { fetcher, asked } = probing({ huggingface: true, modelscope: false });
    const clock = { now: 0 };
    const proxy = { value: "DIRECT" };
    const sources = sourcesWith(fetcher, { clock, proxy });
    await order(sources);
    await order(sources);
    expect(asked).toHaveLength(2);
    clock.now = PROBE_KEPT_MS - 1;
    await order(sources);
    expect(asked).toHaveLength(2);
    clock.now = PROBE_KEPT_MS;
    await order(sources);
    expect(asked).toHaveLength(4);
    proxy.value = "PROXY 127.0.0.1:7890";
    await order(sources);
    expect(asked).toHaveLength(6);
  });

  // protects=a source that failed a file goes last for the model's other files, and comes back once the probe runs again; the User's pick on a failed download's line leads; fails_when=every file waits on the failed source again, the demotion outlives the probe, or the pick is ignored; no_claim=the line itself.
  test("a source that failed goes last until the next probe, and the User's pick leads", async () => {
    const { fetcher } = probing({ huggingface: true, modelscope: false });
    const clock = { now: 0 };
    const sources = sourcesWith(fetcher, { clock });
    await order(sources);
    sources.failed("huggingface");
    expect(await order(sources)).toEqual(["modelscope", "hf-mirror", "r2", "huggingface"]);
    clock.now = PROBE_KEPT_MS;
    expect((await order(sources))[0]).toBe("huggingface");
    expect(await order(sourcesWith(fetcher, { choice: "modelscope" }))).toEqual([
      "modelscope",
      "huggingface",
      "hf-mirror",
      "r2",
    ]);
  });

  // protects=a copy that isn't set up is no source: until IncarnaMind's ModelScope and R2 copies exist, a file is tried at Hugging Face and hf-mirror only, and ModelScope isn't offered as a pick; fails_when=an empty copy yields a URL, or the pick lists a source that has nothing; no_claim=the copies' hosts.
  test("a copy that isn't set up is skipped, and isn't offered", async () => {
    const { fetcher } = probing({ huggingface: true, modelscope: false });
    const sources = sourcesWith(fetcher, { copies: {} });
    expect(await order(sources)).toEqual(["huggingface", "hf-mirror"]);
    expect(sources.choices()).toEqual(["auto", "huggingface"]);
    expect(sourcesWith(fetcher).choices()).toEqual(["auto", "huggingface", "modelscope"]);
  });
});

// Hugging Face's stand-in, and a copy of the same files on another server, as ModelScope's would be.
async function twoSources(files: Record<string, number>) {
  const pinned = await startModelServer(files);
  const copy = await startModelServer({}, { server: pinned, folder: FOLDER });
  const port = new URL(copy.source.baseUrl).port;
  return { pinned, copy, copies: { modelscope: `http://127.0.0.1:${port}/{folder}/` } };
}

async function download(
  server: ModelServer,
  copies: ModelCopies,
  {
    signal = new AbortController().signal,
    stallMs,
  }: { signal?: AbortSignal; stallMs?: number } = {},
) {
  const directory = join(await mkdtemp(join(tmpdir(), "incarnamind-models-")), FOLDER);
  const sources = sourcesWith(LOCAL_FETCH, { copies, ...(stallMs !== undefined && { stallMs }) });
  const run = (at: string, abort = signal) =>
    downloadModel({
      directory: at,
      source: server.source,
      signal: abort,
      onProgress: () => {},
      sources,
    });
  return { directory, run, done: run(directory) };
}

const failure = (done: Promise<unknown>) =>
  done.then(
    () => {
      throw new Error("The download didn't fail.");
    },
    (error: unknown) => error,
  );

describe("A built-in model downloads from the next source when one fails (#1578)", {
  timeout: 30_000,
}, () => {
  // protects=a source that serves other bytes is rejected by the pinned SHA-256, and the next source's file is installed; fails_when=the wrong bytes are kept, or the download stops at the first source; no_claim=which source is faster.
  test("a source that serves a wrong file is rejected, and the next one is used", async () => {
    const { pinned, copy, copies } = await twoSources({ "model.onnx": 2048 });
    pinned.behave("model.onnx", { corrupt: true });
    const { directory, done } = await download(pinned, copies);
    await done;
    expect(new Uint8Array(await readFile(join(directory, "model.onnx")))).toEqual(
      pinned.contents.get("model.onnx"),
    );
    expect(copy.requests.map((each) => each.path)).toEqual([`${FOLDER}/model.onnx`]);
  });

  // protects=a file that gets no bytes for the stall time moves to the next source, which resumes it with a Range request from where the first stopped, and the joined file passes its check; fails_when=the stall waits forever, starts over, or keeps a file that fails its SHA-256; no_claim=the 25 s itself, which the default carries.
  test("a stall moves to the next source, which resumes the file where it stopped", async () => {
    const { pinned, copy, copies } = await twoSources({ "model.onnx": 4096 });
    pinned.behave("model.onnx", { stallAfter: 1000 });
    const { directory, done } = await download(pinned, copies, { stallMs: 200 });
    await done;
    expect(copy.requests).toEqual([
      expect.objectContaining({ path: `${FOLDER}/model.onnx`, range: "bytes=1000-" }),
    ]);
    expect(new Uint8Array(await readFile(join(directory, "model.onnx")))).toEqual(
      pinned.contents.get("model.onnx"),
    );
  });

  // protects=a source that cuts the file short (a truncated download) moves to the next; fails_when=a short file is installed or the download gives up; no_claim=why the connection dropped.
  test("a file cut short moves to the next source", async () => {
    const { pinned, copies } = await twoSources({ "model.onnx": 4096 });
    pinned.behave("model.onnx", { cutAfter: 100 });
    const { directory, done } = await download(pinned, copies);
    await done;
    expect((await readFile(join(directory, "model.onnx"))).byteLength).toBe(4096);
  });

  // protects=with every source failing (one down, one serving wrong bytes), the error says what each said, in the order tried, and nothing is installed: no file in place and no record of a finished download; fails_when=the error names one source, a wrong file is kept, or verified.json is written; no_claim=the problem line's words, which the renderer shows.
  test("with every source failing, it says what each said, and nothing is installed", async () => {
    const { pinned, copy, copies } = await twoSources({ "model.onnx": 2048 });
    pinned.behave("model.onnx", { corrupt: true });
    await copy.close();
    const { directory, done } = await download(pinned, copies);
    const error = await failure(done);
    expect(error).toBeInstanceOf(ModelDownloadError);
    expect((error as ModelDownloadError).sources?.map((each) => each.source)).toEqual([
      "huggingface",
      "modelscope",
    ]);
    expect(existsSync(join(directory, "model.onnx"))).toBe(false);
    expect(existsSync(join(directory, "verified.json"))).toBe(false);
  });

  // protects=offline, every source fails to connect, and the download fails as a network problem naming each, rather than hanging; fails_when=it hangs, or calls it a broken file; no_claim=the operating system's own offline signal.
  test("offline, every source fails as a network problem", async () => {
    const { pinned, copy, copies } = await twoSources({ "model.onnx": 2048 });
    await pinned.close();
    await copy.close();
    const error = await failure((await download(pinned, copies)).done);
    expect(error).toMatchObject({ kind: "network" });
    expect((error as ModelDownloadError).sources).toHaveLength(2);
  });

  // protects=a quit halfway keeps what was downloaded, and the next start resumes it with a Range request instead of starting over; fails_when=the partial file is lost, or the next download asks for the whole file; no_claim=a quit during the SHA-256 check.
  test("a quit halfway keeps the partial file, and the next download resumes it", async () => {
    const { pinned, copies } = await twoSources({ "model.onnx": 10_000 });
    pinned.behave("model.onnx", { stallAfter: 3000 });
    const quit = new AbortController();
    const { directory, run, done } = await download(pinned, copies, { signal: quit.signal });
    await expect
      .poll(
        async () =>
          (await readFile(join(directory, "model.onnx.partial")).catch(() => null))?.byteLength,
      )
      .toBe(3000);
    quit.abort();
    await failure(done);
    pinned.behave("model.onnx", {});
    await run(directory, new AbortController().signal);
    expect(pinned.requests.at(-1)?.range).toBe("bytes=3000-");
    expect(new Uint8Array(await readFile(join(directory, "model.onnx")))).toEqual(
      pinned.contents.get("model.onnx"),
    );
  });
});
