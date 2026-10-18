"""Fetches the Excel-saved test files listed in excel-files.json (commit-pinned URLs), checking each SHA-256.

  python3 -I fetch.py <excel-files.json> <folder>
"""
import hashlib, json, os, sys, urllib.request
manifest, folder = sys.argv[1:3]
os.makedirs(folder, exist_ok=True)
ok = 0
for entry in json.load(open(manifest)):
    path = os.path.join(folder, entry["name"])
    if not os.path.exists(path):
        with urllib.request.urlopen(entry["url"], timeout=60) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != entry["sha256"]:
            raise SystemExit(f"{entry['name']}: SHA-256 differs from the manifest")
        open(path, "wb").write(data)
    ok += 1
print(ok, "files")
