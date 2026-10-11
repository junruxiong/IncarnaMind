#!/usr/bin/env python3
"""
Fetches the transcription evaluation's clips (eval/transcription/README.md):
the first clips of Google's FLEURS dev split (CC BY 4.0) in English and
Mandarin Chinese, with their reference transcripts, from a pinned revision on
Hugging Face. Only the start of each audio archive is read: the download stops
once enough clips are out. Nothing is committed: the clips go to <out-dir>.

    python3 -I eval/transcription/fetch-fleurs.py <out-dir> [clips per language]

Writes <out-dir>/<language>/<id>.wav (16 kHz mono) and <out-dir>/<language>/clips.tsv:
file, raw transcription, normalised transcription, samples, gender.
"""
import csv
import io
import os
import sys
import tarfile
import urllib.request

REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
LANGUAGES = {"en": "en_us", "zh": "cmn_hans_cn"}


def fetch(config, out, count):
    base = f"https://huggingface.co/datasets/google/fleurs/resolve/{REVISION}/data/{config}"
    os.makedirs(out, exist_ok=True)
    with urllib.request.urlopen(f"{base}/dev.tsv") as response:
        rows = {}
        reader = csv.reader(io.TextIOWrapper(response, "utf-8"), delimiter="\t", quoting=csv.QUOTE_NONE)
        for row in reader:
            rows[row[1]] = row
    kept = []
    with urllib.request.urlopen(f"{base}/audio/dev.tar.gz") as response:
        with tarfile.open(fileobj=response, mode="r|gz") as archive:
            for member in archive:
                name = os.path.basename(member.name)
                if not member.isfile() or name not in rows:
                    continue
                with open(os.path.join(out, name), "wb") as f:
                    f.write(archive.extractfile(member).read())
                kept.append(rows[name])
                if len(kept) >= count:
                    break
    with open(os.path.join(out, "clips.tsv"), "w", encoding="utf-8") as f:
        for row in kept:
            f.write("\t".join([row[1], row[2], row[3], row[5], row[6]]) + "\n")
    print(f"{len(kept)} {config} clips in {out}")


if __name__ == "__main__":
    target = sys.argv[1]
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    for language, config in LANGUAGES.items():
        fetch(config, os.path.join(target, language), count)
