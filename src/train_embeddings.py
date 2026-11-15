
"""Train CBOW, Skip-gram and fastText word embedding models."""

import time
from pathlib import Path

import pandas as pd
from gensim.models import FastText, Word2Vec

ROOT = Path(__file__).resolve().parents[1]
CLEAN_PATH = ROOT / "data" / "cleaned_tinystories.csv"
MODEL_DIR = ROOT / "models"

PARAMS = {
    "vector_size": 100,
    "window": 5,
    "min_count": 5,
    "epochs": 10,
    "workers": 4,
    "seed": 42,
}


def load_sentences():
    df = pd.read_csv(CLEAN_PATH)
    return [
        text.split()
        for text in df["text_processed"].fillna("")
        if text.strip()
    ]


def train(name, model_class, sentences, **extra):
    start = time.time()

    model = model_class(
        sentences=sentences,
        **PARAMS,
        **extra
    )

    model.save(str(MODEL_DIR / f"{name}.model"))

    elapsed = time.time() - start

    print(
        f"{name:9s} vocab={len(model.wv):,} "
        f"trained in {elapsed:.2f}s"
    )

    return model


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    sentences = load_sentences()

    print(
        f"Training on {len(sentences):,} stories, "
        f"{sum(map(len, sentences)):,} tokens"
    )

    train("cbow", Word2Vec, sentences, sg=0)
    train("skipgram", Word2Vec, sentences, sg=1)

    train(
        "fasttext",
        FastText,
        sentences,
        sg=1,
        min_n=3,
        max_n=6,
        bucket=200_000
    )

    print("\nAll embedding models trained and saved successfully!")


if __name__ == "__main__":
    main()
