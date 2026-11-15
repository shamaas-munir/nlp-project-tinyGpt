"""Phase 2, Part A: fit the count-based vectorizers once and save them.

Run from the repo root:  python src/build_count_models.py
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
CLEAN_PATH = ROOT / "data" / "cleaned_tinystories.csv"
MODEL_DIR = ROOT / "models"

# token_pattern=r"\S+" means "a token is anything between spaces".
# Our text is already tokenized and lemmatized by Phase 1, so we only split on spaces.
TOKEN_PATTERN = r"\S+"


def load_corpus():
    df = pd.read_csv(CLEAN_PATH)
    return df["text_processed"].fillna("").tolist()


def main():
    docs = load_corpus()
    MODEL_DIR.mkdir(exist_ok=True)

    bow = CountVectorizer(token_pattern=TOKEN_PATTERN)
    bow.fit(docs)
    joblib.dump(bow, MODEL_DIR / "bow_vectorizer.joblib")

    tfidf = TfidfVectorizer(token_pattern=TOKEN_PATTERN)
    tfidf.fit(docs)
    joblib.dump(tfidf, MODEL_DIR / "tfidf_vectorizer.joblib")

    print(f"Saved BoW and TF-IDF vectorizers ({len(tfidf.vocabulary_):,} words) -> {MODEL_DIR}")


if __name__ == "__main__":
    main()
