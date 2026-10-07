# 🧠 IncarnaMind

## 👀 In a Nutshell

IncarnaMind enables you to chat with your personal documents 📁 (PDF, TXT) using Large Language Models (LLMs) like GPT ([architecture overview](#high-level-architecture)). While OpenAI has recently launched a fine-tuning API for GPT models, it doesn't enable the base pretrained models to learn new data, and the responses can be prone to factual hallucinations. Utilize our [Sliding Window Chunking](#sliding-window-chunking) mechanism and Ensemble Retriever enables efficient querying of both fine-grained and coarse-grained information within your ground truth documents to augment the LLMs.

## 🖥 Desktop app (in development)

IncarnaMind v1 is being rebuilt as a local-first desktop app (Electron and TypeScript). The Python command-line tool described below keeps working until the new app replaces it.

You need Node.js 24 or newer. After cloning:

```shell
npm install
npm run dev          # start the app in development
npm run typecheck    # TypeScript, strict
npm run lint         # Biome
npm test             # Vitest: drives the core's public interface
npm run test:smoke   # Playwright: builds the app and drives it in Electron
```

Code lives in `src/`: `core` (app logic, no Electron imports), `main` (Electron main process), `preload` (the typed bridge to the UI), `renderer` (React UI) and `shared` (i18n dictionaries, bridge names and the text normaliser).

Keep development data apart from your real data folder by pointing the app at another one: `INCARNAMIND_DATA_DIR=/tmp/incarnamind-dev npm run dev`.

Document search runs a built-in embedding model, multilingual-e5-small (int8 ONNX, 135 MB), on your CPU in an Electron utility process. The app downloads it from Hugging Face into the data folder (`models/`) the first time a Document needs it, and checks each file's SHA-256; after that, indexing works offline. The tests use a deterministic fake instead. `INCARNAMIND_REAL_MODEL=1 npm test` also runs one test with the real model, downloading it unless `INCARNAMIND_MODEL_DIR` points at a folder holding its files. On Linux x64, set `ONNXRUNTIME_NODE_INSTALL=skip` when you run `npm install`, or onnxruntime-node also downloads its CUDA libraries, which the app doesn't use.

### Installing a release

Download the installer for your system from [GitHub Releases](https://github.com/junruxiong/IncarnaMind/releases):

- **macOS:** `IncarnaMind-<version>-mac-arm64.dmg` for Apple silicon, or `-mac-x64.dmg` for Intel. Open it and drag IncarnaMind into Applications.
- **Windows:** `IncarnaMind-<version>-win-x64.exe`.
- **Linux:** `IncarnaMind-<version>-linux-x86_64.AppImage`. Make it executable (`chmod +x IncarnaMind-*.AppImage`) and run it. If it doesn't start on Ubuntu 22.04 or later, install FUSE 2: `sudo apt install libfuse2` (`libfuse2t64` on 24.04 and later).

IncarnaMind checks for a new version each time it starts. On Windows, in the Linux AppImage and in a signed macOS build, it downloads the update in the background and installs it when you quit, or straight away if you choose **Restart now**.

**Windows: "Windows protected your PC".** The installer isn't code-signed yet, so Microsoft Defender SmartScreen stops it the first time. Click **More info**, then **Run anyway**. If your browser flags the download, choose **Keep**.

**macOS: opening an unsigned build.** Until the macOS build is signed and notarized, macOS refuses to open it with "Apple could not verify 'IncarnaMind' is free of malware". To open it anyway:

1. Open IncarnaMind once, and click **Done** in the warning.
2. Open **System Settings → Privacy & Security**, scroll down to **Security**, and click **Open Anyway** next to the message about IncarnaMind.
3. Confirm with your password or Touch ID, then click **Open Anyway** again.

macOS remembers your choice. Or, in Terminal: `xattr -dr com.apple.quarantine /Applications/IncarnaMind.app`.

An unsigned macOS build can't update itself. When a new version is out, IncarnaMind tells you and offers its download page; install it the same way, replacing the old app.

### Packaging and releases

[electron-builder](https://www.electron.build) packages the app (config: `electron-builder.yml`), and electron-updater updates installed apps from GitHub Releases (`src/main/updater.ts`). To package locally, into `dist/`:

```shell
npm run dist:dir     # the unpacked app, for a quick check (e.g. dist/mac-arm64/IncarnaMind.app)
npm run dist         # the installers for the current system; never published
```

**Cutting a release:**

1. Set the version, commit it and tag it: `npm version 1.2.3`. This updates `package.json` and `package-lock.json`, commits, and creates the tag `v1.2.3`.
2. Push the commit and the tag: `git push && git push origin v1.2.3`.
3. The tag starts the **Release** workflow (`.github/workflows/release.yml`). It checks that the tag matches `package.json`, creates a draft GitHub Release, and builds and uploads:
   - for macOS, a dmg and a zip, each for Apple silicon and Intel;
   - for Windows, an NSIS installer;
   - for Linux, an AppImage;
   - the `latest*.yml` files that auto-update reads.
4. Review the draft, edit its notes, try the installers, and click **Publish release**. Installed apps only see an update once its release is published.

**macOS signing and notarization.** The workflow signs and notarizes the macOS app when these repository secrets exist. Without `CSC_LINK`, it builds an unsigned (ad-hoc signed) app and still succeeds.

| Secret | Value |
| --- | --- |
| `CSC_LINK` | The "Developer ID Application" certificate and its private key, exported from Keychain Access as a `.p12` file and base64-encoded (`base64 -i certificate.p12 \| pbcopy`) |
| `CSC_KEY_PASSWORD` | The password of that `.p12` file |
| `APPLE_ID`, `APPLE_APP_SPECIFIC_PASSWORD`, `APPLE_TEAM_ID` | To notarize with an Apple ID: the Apple ID, an [app-specific password](https://support.apple.com/102654) and the team ID |
| `APPLE_API_KEY`, `APPLE_API_KEY_ID`, `APPLE_API_ISSUER` | Or, to notarize with an App Store Connect API key: the contents of the `.p8` file, its key ID and the issuer ID |
The Windows installer stays unsigned in v1; signing through the SignPath Foundation is a follow-up.

**Crash reports.** IncarnaMind collects no usage data. Users can opt in to crash reports in **Settings → Privacy**; they go to Sentry, scrubbed of file paths, Document text, Mind content, Questions and Answers (`src/main/crashScrubber.ts`). Only a build made with a Sentry DSN offers them: set `MAIN_VITE_SENTRY_DSN` when building (electron-vite reads it from the environment or a `.env.local` file). The release workflow passes the `SENTRY_DSN` repository secret; without it, releases don't offer crash reports. Never commit a DSN.

## 1.2. Setup

Create Conda virtual environment:

```shell
conda create -n IncarnaMind python=3.10
```

Activate:

```shell
conda activate IncarnaMind
```

Install all requirements:

```shell
pip install -r requirements.txt
```

Install [llama-cpp](https://github.com/abetlen/llama-cpp-python) seperatly if you want to run quantized local LLMs:

- For `NVIDIA` GPUs support, use `cuBLAS`

```shell
CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python==0.1.83 --no-cache-dir
```

- For Apple Metal (`M1/M2`) support, use

```shell
CMAKE_ARGS="-DLLAMA_METAL=on"  FORCE_CMAKE=1 pip install llama-cpp-python==0.1.83 --no-cache-dir
```

Setup your one/all of API keys in **configparser.ini** file:

```shell
[tokens]
OPENAI_API_KEY = (replace_me)
ANTHROPIC_API_KEY = (replace_me)
TOGETHER_API_KEY = (replace_me)
# if you use full Meta-Llama models, you may need Huggingface token to access.
HUGGINGFACE_TOKEN = (replace_me)
```

(Optional) Setup your custom parameters in **configparser.ini** file:

```shell
[parameters]
PARAMETERS 1 = (replace_me)
PARAMETERS 2 = (replace_me)
...
PARAMETERS n = (replace_me)
```

### 2. Usage

#### 2.1. Upload and process your files

Put all your files (please name each file correctly to maximize the performance) into the **/data** directory and run the following command to ingest all data:
(You can delete example files in the **/data** directory before running the command)

```shell
python docs2db.py
```

#### 2.2. Run

In order to start the conversation, run a command like:

```shell
python main.py
```

#### 2.3. Chat and ask any questions

Wait for the script to require your input like the below.

```shell
Human:
```

#### 2.4. Others

When you start a chat, the system will automatically generate a **IncarnaMind.log** file.
If you want to edit the logging, please edit in the **configparser.ini** file.

```shell
[logging]
enabled = True
level = INFO
filename = IncarnaMind.log
format = %(asctime)s [%(levelname)s] %(name)s: %(message)s
```

## 🚫 Limitations

- 
## 📝 Upcoming Features

- 

## 🙌 Acknowledgements

Special thanks to [Langchain](https://github.com/langchain-ai/langchain), [Chroma DB](https://github.com/chroma-core/chroma), [LocalGPT](https://github.com/PromtEngineer/localGPT), [Llama-cpp](https://github.com/abetlen/llama-cpp-python) for their invaluable contributions to the open-source community. Their work has been instrumental in making the IncarnaMind project a reality.

## 🖋 Citation

If you want to cite our work, please use the following bibtex entry:

```bibtex
@misc{IncarnaMind2023,
  author = {Junru Xiong},
  title = {IncarnaMind},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/junruxiong/IncarnaMind}}
}
```

## 📑 License

[Apache 2.0 License](LICENSE)