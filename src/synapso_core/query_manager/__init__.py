import nltk

from .query_manager import QueryManager

nltk.download("wordnet")  # ignore (used for reranker)
nltk.download("stopwords")  # ignore (used for reranker)
nltk.download("punkt")  # ignore (used for reranker)

__all__ = ["QueryManager"]
