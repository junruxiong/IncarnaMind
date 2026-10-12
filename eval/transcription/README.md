# Transcription on this computer

How well, and how fast, the first speech-to-text engine (#136) transcribes: Qwen3-ASR-0.6B int8 with Silero VAD, through sherpa-onnx-node, on the CPU. It runs the transcription process's own code (`createSpeechHost` in `src/core/transcription/speech.ts`, `createSherpaEngine` in `src/core/transcription/sherpa.ts`) on a stream made of read clips, fed in 100 ms frames at the pace it was spoken, as a Meeting's audio is.

This is a stand-in until #85's meeting audio set and scorer exist. #137 runs the bake-off on that set, on three machines, and sets the gate.

## Running it

```sh
python3 -I eval/transcription/fetch-fleurs.py ~/.cache/incarnamind-eval/fleurs 8
INCARNAMIND_EVAL_CLIPS=~/.cache/incarnamind-eval/fleurs npm run eval:transcription
```

- **Clips:** the first clips of Google's FLEURS dev split, in English (`en_us`) and Mandarin Chinese (`cmn_hans_cn`), with their reference transcripts, from a pinned revision on Hugging Face. FLEURS is under CC BY 4.0 (Conneau et al., "FLEURS: Few-shot Learning Evaluation of Universal Representations of Speech", 2022). `fetch-fleurs.py` reads only the start of each audio archive and stops once it has the clips. Nothing is committed.
- **The model:** downloaded the first time into `~/.cache/incarnamind-eval/models/` (about 985 MB), checked against the SHA-256 hashes pinned in `src/core/transcription/model.ts`, as the app downloads it; `INCARNAMIND_EVAL_MODELS` points at another folder laid out the same way.
- **The stream:** each language's clips in order, with 1.5 s of silence before each and after the last, as pauses between speakers.
- **Reported:** WER for English, CER for Chinese, after the normalisation below; how long decoding the final text took for the speech it decoded; how long all decoding took, partial text included, for the stream; how many partials there were; and how far after the audio ended the last line came. Also the machine's load average while it ran: the times are only as good as the machine was free. The report is written to `eval/results/`.

## Normalisation

The rules are in `scoring.ts` and tested in `tests/eval/transcriptionScoring.test.ts`:

- case is folded, and full-width letters and digits become ASCII (NFKC);
- punctuation and symbols are dropped, in both scripts;
- English is compared word by word; Chinese character by character, each run of Latin letters or digits as one unit;
- numbers are left as written: "6" and "六" differ. #85 fixes a rule for them.

## Results

2026-10-11, Apple M2 Max (12 cores, 32 GB), 4 threads, while the machine ran other work (load average 28 to 35 over the run):

| Language | Clips | Audio | Error | Decoding / speech decoded | All decoding / stream | Last line after the audio ended |
|---|---|---|---|---|---|---|
| English | 8 | 74.7 s | WER 12.8% (18 of 141 words) | 0.15× | 0.22× | 0.0 s |
| Chinese | 8 | 113.9 s | CER 1.7% (4 of 231 characters) | 0.23× | 0.17× | 0.0 s |

- Of the 4 Chinese errors, one is a number written as a word ("六英里" for "6 英里"), two are names written in other characters ("科龙" for "科隆", "加南" for "迦南"), and one a character said twice ("主主要").
- Most English errors are names (Javanese, archipelago, Lakkha Singh, bhajans, Mahayana) and one clip read badly.
- FLEURS is read speech, one speaker at a time, recorded close: easier than a meeting.
- Load matters. Earlier the same day, with the machine far busier (load average 124 to 183), decoding single clips took 1.3 to 1.7 times their length: slower than real time. Live text then falls behind, the meeting bar says "Recording…", and the rest is transcribed after Stop (#136).
- The FLEURS clips are recorded quietly (a peak of 0.5% to 2.4% of full scale for most English ones). Before the speech host raised quiet audio (up to 30 dB), Silero VAD missed four of the eight English clips: WER 48.2%.
