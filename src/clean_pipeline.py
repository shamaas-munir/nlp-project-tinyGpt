"""Phase 1 cleaning pipeline: raw TinyStories CSV -> cleaned CSV for later phases.

Run from the repo root:  python src/clean_pipeline.py
"""

import re
import time
import unicodedata
from pathlib import Path

import pandas as pd
import spacy

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw_tinystories.csv"
OUT_PATH = ROOT / "data" / "cleaned_tinystories.csv"
TARGETS = ("Dialogue", "MoralValue", "Conflict")

# Regex 1: metadata tag parser
FEATURES_RE = re.compile(r"^Features:\s*([^\n]+)", re.MULTILINE)
# Regex 2: header block, trailing summary and end-of-text marker
HEADER_RE = re.compile(r"\A.*?^Story:[ \t]*", re.DOTALL | re.MULTILINE)
SUMMARY_LINE_RE = re.compile(r"^Summary:.*$", re.MULTILINE)
EOT_RE = re.compile(r"<\|endoftext\|>")
# Regex 3: dialogue normalizer
DOUBLE_QUOTES_RE = re.compile(r"[“”]")
SINGLE_QUOTES_RE = re.compile(r"[‘’]")
DIALOGUE_RE = re.compile(r'"([^"]*)"')
# Regex 4: punctuation and whitespace noise
ELLIPSIS_RE = re.compile(r"…|\.{3,}")
DASH_RE = re.compile(r"\s*(?:—|–|-{2,})\s*")
SPACES_RE = re.compile(r"[ \t]+")


def extract_label(raw_text):
    """Return the single target tag found on the Features: line, else None."""
    match = FEATURES_RE.search(raw_text)
    if match is None:
        return None
    tags = [tag.strip() for tag in match.group(1).split(",")]
    hits = [tag for tag in tags if tag in TARGETS]
    return hits[0] if len(hits) == 1 else None


def clean_text(raw_text):
    """Raw record -> clean story text. Keeps case and punctuation (needed by the LM)."""
    text = unicodedata.normalize("NFC", raw_text)
    text = HEADER_RE.sub("", text)
    text = SUMMARY_LINE_RE.sub("", text)
    text = EOT_RE.sub("", text)
    text = DOUBLE_QUOTES_RE.sub('"', text)
    text = SINGLE_QUOTES_RE.sub("'", text)
    text = ELLIPSIS_RE.sub("...", text)
    text = DASH_RE.sub(" - ", text)
    text = SPACES_RE.sub(" ", text)
    lines = [line.strip() for line in text.split("\n")]
    return "\n".join(line for line in lines if line)


def load_spacy():
    return spacy.load("en_core_web_sm", disable=["parser", "ner"])


def lemmatize_texts(texts, nlp, batch_size=500):
    """Clean story texts -> lowercase lemmas, punctuation dropped, stop words kept."""
    results = []
    for doc in nlp.pipe(texts, batch_size=batch_size):
        lemmas = [tok.lemma_.lower() for tok in doc]
        words = [lemma for lemma in lemmas if lemma.isalpha()]
        results.append(" ".join(words))
    return results


def run_pipeline(raw_path=RAW_PATH, out_path=OUT_PATH):
    df = pd.read_csv(raw_path)
    print(f"Loaded {len(df):,} raw stories")

    df["label"] = df["raw_text"].apply(extract_label)
    mismatches = (df["label"] != df["narrative_type"]).sum()
    print(f"Label check: {mismatches} mismatches")

    df["story_clean"] = df["raw_text"].apply(clean_text)

    nlp = load_spacy()
    start = time.time()
    df["text_processed"] = lemmatize_texts(df["story_clean"].tolist(), nlp)
    print(f"Lemmatized in {time.time() - start:.0f}s")

    df["n_words"] = df["story_clean"].str.split().str.len()
    out = df[["id", "label", "story_clean", "text_processed", "n_words"]]
    out.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Saved {len(out):,} stories -> {out_path}")
    return out


if __name__ == "__main__":
    run_pipeline()
