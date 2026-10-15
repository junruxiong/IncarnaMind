The echo canceller's test voices (`tests/core/echo.test.ts`) are four clips of Google's FLEURS dev split (CC BY 4.0, https://creativecommons.org/licenses/by/4.0/; Conneau et al., "FLEURS: Few-shot Learning Evaluation of Universal Representations of Speech", 2022), from revision `70bb2e84b976b7e960aa89f1c648e09c59f894dd` on Hugging Face, fetched with `eval/transcription/fetch-fleurs.py`:

| File | FLEURS clip | Speaker | What is said |
|---|---|---|---|
| `en-near.wav` | `en_us` 10280794863324544626 | male | Some festivals have special camping areas for families with young children. |
| `en-far.wav` | `en_us` 10146705666908229607 | female | The major religion in Moldova is Orthodox Christian. |
| `zh-near.wav` | `cmn_hans_cn` 10519077850963840950 | male | 他称，他制作了一个 WiFi 门铃。 |
| `zh-far.wav` | `cmn_hans_cn` 11184525298057700964 | female | 西班牙人开始了长达三个世纪的殖民时期。 |

Changed from FLEURS: the silence before and after the speech is cut to 0.1 s, each clip is kept to its first 4 seconds (so `en-near.wav` and `zh-far.wav` stop before the end of the sentence), raised to an RMS of −26 dBFS over its speech, and written as 16 kHz mono 16-bit WAV with a 44-byte header.
