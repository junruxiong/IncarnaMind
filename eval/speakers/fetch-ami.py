#!/usr/bin/env python3
"""
Fetches the speaker evaluation's meetings (eval/speakers/README.md): the AMI Meeting Corpus
(CC BY 4.0) as diarizers-community/ami keeps it on Hugging Face, from a pinned revision. The
first test shard of each microphone setting, six meetings of four people, three hours in all:
"Mix-Headset" (close microphones mixed, standing in for a call) and "Array1-01" (one distant
microphone, a meeting in a room). Nothing is committed: the meetings go to <out-dir>.

    uv run --with pyarrow python3 -I eval/speakers/fetch-ami.py <out-dir>

Writes <out-dir>/online/<meeting>.wav and .rttm, and <out-dir>/room/<meeting>.wav and .rttm:
16 kHz mono WAV, and AMI's speaker turns as RTTM.
"""
import os
import struct
import sys
import urllib.request

import pyarrow.parquet as pq

REVISION = "8cdaae2eaf968f3b000b6eb1204ab9b8db006ed0"
SETTINGS = {"online": "ihm", "room": "sdm"}
SHARD = "test-00000-of-00003.parquet"


def check_wav(data, name):
    channels, rate = struct.unpack("<HI", data[22:28])
    if data[:4] != b"RIFF" or channels != 1 or rate != 16000:
        raise SystemExit(f"{name} isn't 16 kHz mono WAV ({channels} channels, {rate} Hz)")


def fetch(setting, config, out):
    folder = os.path.join(out, setting)
    os.makedirs(folder, exist_ok=True)
    shard = os.path.join(out, f"{config}-{SHARD}")
    if not os.path.exists(shard):
        url = f"https://huggingface.co/datasets/diarizers-community/ami/resolve/{REVISION}/{config}/{SHARD}"
        urllib.request.urlretrieve(url, shard + ".partial")
        os.replace(shard + ".partial", shard)
    table = pq.ParquetFile(shard)
    for group in range(table.num_row_groups):
        for row in table.read_row_group(group).to_pylist():
            name = os.path.basename(row["audio"]["path"]).split(".")[0]
            data = row["audio"]["bytes"]
            check_wav(data, name)
            with open(os.path.join(folder, name + ".wav"), "wb") as f:
                f.write(data)
            with open(os.path.join(folder, name + ".rttm"), "w") as f:
                for start, end, speaker in zip(row["timestamps_start"], row["timestamps_end"], row["speakers"]):
                    f.write(f"SPEAKER {name} 1 {start:.3f} {end - start:.3f} <NA> <NA> {speaker} <NA> <NA>\n")
            print(f"{setting}: {name}, {len(set(row['speakers']))} speakers")
    os.remove(shard)


if __name__ == "__main__":
    for setting, config in SETTINGS.items():
        fetch(setting, config, sys.argv[1])
