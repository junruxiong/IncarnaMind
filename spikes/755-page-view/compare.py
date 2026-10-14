"""Spike for #755: compares each approach's page breaks with LibreOffice's.

python3 compare.py <lo-pages.json> <results-dir> [names...]
lo-pages.json: {name: [[line, ...] per page]} from pdftotext -layout of LibreOffice's PDF.
"""
import json
import os
import re
import sys
import unicodedata

NUMBERING = re.compile(r"^(?:[\(（\[]?(?:\d{1,3}|[a-zA-Z]|[ivxIVX]{1,4}|[一二三四五六七八九十]{1,3})[\.\)）、．](?:\d{1,3}[\.\)]?)*)+")
N = 18


def norm(text):
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    text = re.sub(r"\s+", "", text).lower()
    text = NUMBERING.sub("", text)
    return text


def lo_candidates(pages):
    """Each page's first body lines: the line every page starts with (the header) is skipped."""
    firsts = {}
    for page in pages:
        for line in page[:2]:
            firsts[norm(line)] = firsts.get(norm(line), 0) + 1
    header = {line for line, count in firsts.items() if count > max(2, len(pages) * 0.5) and line}
    out = []
    for page in pages:
        lines = [norm(line) for line in page if norm(line) and norm(line) not in header]
        out.append(lines[:3])
    return out


def match(ours, candidates):
    if not ours:
        return False
    for line in candidates[:2]:
        n = min(N, len(ours), len(line))
        if n >= 4 and (line.startswith(ours[:n]) or ours.startswith(line[:n])):
            return True
    return False


FOOTER = re.compile(r"^(page\d+(of\d+)?|\d+|—\d+—|-\d+-|第\d+页(，?共\d+页?)?)$")


def offsets(lo_pages, ours):
    """For each of our pages, how many body lines from LibreOffice's page start ours starts (None: not found)."""
    lo = lo_candidates(lo_pages)
    header = set()
    lines, starts = [], []
    for page, cands in zip(lo_pages, lo):
        body = [norm(line) for line in page if norm(line)]
        first = cands[0] if cands else None
        if first in body:
            body = body[body.index(first):]
        body = [line for line in body if not FOOTER.match(line)]
        starts.append(len(lines))
        lines.extend(body)
    out = []
    for i, text in enumerate(ours):
        if i >= len(starts) or not text:
            out.append(None)
            continue
        best = None
        for j in range(max(0, starts[i] - 12), min(len(lines), starts[i] + 13)):
            n = min(N, len(text), len(lines[j]))
            if n >= 4 and (lines[j].startswith(text[:n]) or text.startswith(lines[j][:n])):
                if best is None or abs(j - starts[i]) < abs(best):
                    best = j - starts[i]
        out.append(best)
    return out


def compare(lo_pages, entry):
    lo = lo_candidates(lo_pages)
    ours = [norm(line["text"]) for line in entry.get("firstLines", [])]
    same_index = sum(1 for i, text in enumerate(ours) if i < len(lo) and match(text, lo[i]))
    first_diff = next((i + 1 for i, text in enumerate(ours) if i >= len(lo) or not match(text, lo[i])), None)
    # Breaks LibreOffice makes that ours makes too, wherever they fall.
    reproduced = sum(1 for cand in lo[1:] if any(match(text, cand) for text in ours[1:]))
    offs = offsets(lo_pages, ours)
    found = [o for o in offs if o is not None]
    return {
        "line_offset_0": f"{sum(1 for o in found if o == 0)}/{len(offs)}",
        "within_1": f"{sum(1 for o in found if abs(o) <= 1)}/{len(offs)}",
        "within_3": f"{sum(1 for o in found if abs(o) <= 3)}/{len(offs)}",
        "pages_lo": len(lo_pages),
        "pages_ours": entry.get("pages"),
        "same_index": f"{same_index}/{len(lo)}",
        "breaks_reproduced": f"{reproduced}/{max(0, len(lo) - 1)}",
        "first_differing_page": first_diff,
    }


def main():
    lo_all = json.load(open(sys.argv[1], encoding="utf-8"))
    names = sys.argv[3:] or [f[:-5] for f in os.listdir(sys.argv[2]) if f.endswith(".json")]
    for name in sorted(names):
        path = os.path.join(sys.argv[2], name + ".json")
        if not os.path.exists(path) or name not in lo_all:
            continue
        result = json.load(open(path, encoding="utf-8"))
        for mode in ("plus", "custom"):
            if mode in result:
                print(name, mode, json.dumps(compare(lo_all[name], result[mode]), ensure_ascii=False))


if __name__ == "__main__":
    main()
