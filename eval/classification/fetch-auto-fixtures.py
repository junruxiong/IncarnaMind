"""Fetch only the frozen public PDFs; no model calls, credentials, or app database."""
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
definition = json.loads((ROOT / "eval/classification/auto-set.json").read_text())
for sample in definition["samples"]:
    if "url" not in sample:
        continue
    target = ROOT / sample["path"]
    if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == sample["sha256"]:
        print(f"Already verified: {sample['id']}")
        continue
    with urlopen(Request(sample["url"], headers={"User-Agent": "IncarnaMind document evaluation"}), timeout=45) as response:
        data = response.read(40 * 1024 * 1024)
    if not data.startswith(b"%PDF") or hashlib.sha256(data).hexdigest() != sample["sha256"]:
        raise RuntimeError(f"Source version changed or unavailable: {sample['id']}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    print(f"Downloaded and verified: {sample['id']}")
