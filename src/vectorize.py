"""Turn a raw story into a vector: vectorize(text, method="tfidf")."""

from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
from gensim.models import FastText, Word2Vec

from src.clean_pipeline import clean_text, lemmatize_texts, load_spacy

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
METHODS = ("bow", "tfidf", "cbow", "skipgram", "fasttext")


@lru_cache(maxsize=None)
def _nlp():
    return load_spacy()


@lru_cache(maxsize=None)
def _load(method):
    if method in ("bow", "tfidf"):
        return joblib.load(MODEL_DIR / f"{method}_vectorizer.joblib")
    if method == "fasttext":
        return FastText.load(str(MODEL_DIR / "fasttext.model"))
    return Word2Vec.load(str(MODEL_DIR / f"{method}.model"))


def preprocess(raw_text):
    return lemmatize_texts([clean_text(raw_text)], _nlp())[0]


def vectorize(raw_text, method="tfidf"):
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    text = preprocess(raw_text)
    model = _load(method)

    if method in ("bow", "tfidf"):
        return model.transform([text]).toarray()[0]

    wv = model.wv
    words = text.split() if method == "fasttext" else [w for w in text.split() if w in wv]
    if not words:
        return np.zeros(wv.vector_size)
    return np.mean([wv[w] for w in words], axis=0)
