import nltk
from nltk.data import find

from .bm25_reranker import BM25Reranker
from .factory import RerankerFactory
from .interface import Reranker


def _ensure_nltk_resources():
    resources = [
        "wordnet",
        "stopwords",
        "punkt",
        "punkt_tab",
    ]
    for resource in resources:
        try:
            find(f"corpora/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)


_ensure_nltk_resources()

__all__ = ["BM25Reranker", "RerankerFactory", "Reranker"]
