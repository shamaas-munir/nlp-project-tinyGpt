"""Stream a balanced 15k-story sample from roneneldan/TinyStoriesInstruct.

The corpus is one big text file. Stories are
blocks of header lines (Features / Words / Summary / Random sentence, in random
order) followed by "Story:" and the body, terminated by "<|endoftext|>".

We keep only stories whose Features tag set contains exactly ONE of the target
labels (Dialogue, MoralValue, Conflict), so Phase 3 has a single, unambiguous
label per story. With --strict, the story must carry no other tags at all.

Output: data/raw_tinystories.csv  (raw text kept intact for Phase 1 EDA)

Usage:
    python data/download_tinystories.py [--per-class 5000] [--strict] [--seed 42]
"""

import argparse
import io
import random
import sys
import urllib.request
from collections import Counter
from pathlib import Path

import pandas as pd

REPO = "roneneldan/TinyStoriesInstruct"
TARGETS = ("Dialogue", "MoralValue", "Conflict")
EOT = "<|endoftext|>"
OUT = Path(__file__).resolve().parent / "raw_tinystories.csv"


def iter_story_blocks(fname="TinyStories-Instruct-train.txt"):
    """Stream the raw .txt over HTTP (no 2.6 GB download) and yield story blocks.

    We read the file directly instead of via `datasets` because its streaming
    text loader decodes with the OS locale (cp1252 on Windows) and crashes on
    the UTF-8 curly quotes in the corpus.
    """
    url = f"https://huggingface.co/datasets/{REPO}/resolve/main/{fname}"
    buf = []
    with urllib.request.urlopen(url) as resp:
        for raw in io.TextIOWrapper(resp, encoding="utf-8", errors="replace"):
            line = raw.rstrip("\r\n")
            if line.strip() == EOT:
                if buf:
                    yield buf
                buf = []
            else:
                buf.append(line)


def parse_block(lines):
    rec = {"features": "", "words": "", "summary": "", "random_sentence": ""}
    for i, line in enumerate(lines):
        if line.startswith("Story:"):
            # the story may start on the same line as "Story:" or on the next one
            body_lines = [line[len("Story:"):]] + lines[i + 1:]
            # in some records the Summary line comes AFTER the story
            if body_lines[-1].startswith("Summary:"):
                rec["summary"] = body_lines.pop()[len("Summary:"):].strip()
            body = "\n".join(body_lines).strip()
            break
        key, _, val = line.partition(":")
        key = key.strip().lower().replace(" ", "_")
        if key in rec:
            rec[key] = val.strip()
    else:
        return None
    rec["story"] = body
    rec["raw_text"] = "\n".join(lines) + "\n" + EOT
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=5000)
    ap.add_argument("--strict", action="store_true",
                    help="require the target tag to be the story's ONLY tag")
    ap.add_argument("--max-scan", type=int, default=2_000_000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    buckets = {t: [] for t in TARGETS}
    tag_counts = Counter()
    scanned = 0
    for block in iter_story_blocks():
        scanned += 1
        rec = parse_block(block)
        if rec is None or not rec["story"]:
            continue
        tags = {t.strip() for t in rec["features"].split(",") if t.strip()}
        tag_counts.update(tags or {"<none>"})
        hits = tags & set(TARGETS)
        if len(hits) != 1 or (args.strict and len(tags) != 1):
            continue
        label = hits.pop()
        if len(buckets[label]) < args.per_class:
            rec["narrative_type"] = label
            buckets[label].append(rec)
        if scanned % 20000 == 0:
            print(f"scanned {scanned:,} | " +
                  " ".join(f"{k}={len(v)}" for k, v in buckets.items()), flush=True)
        if all(len(v) >= args.per_class for v in buckets.values()) or scanned >= args.max_scan:
            break

    print(f"\nScanned {scanned:,} stories. Tag frequencies:")
    for tag, n in tag_counts.most_common():
        print(f"  {tag:15s} {n:,}")

    rows = [r for v in buckets.values() for r in v]
    random.Random(args.seed).shuffle(rows)
    df = pd.DataFrame(rows, columns=["narrative_type", "features", "words", "summary",
                                     "random_sentence", "story", "raw_text"])
    df.insert(0, "id", range(len(df)))
    df.to_csv(OUT, index=False, encoding="utf-8")
    print(f"\nSaved {len(df):,} stories -> {OUT}")
    print(df["narrative_type"].value_counts().to_string())
    if any(len(v) < args.per_class for v in buckets.values()):
        print("WARNING: some classes under-filled; raise --max-scan or drop --strict", file=sys.stderr)


if __name__ == "__main__":
    main()
