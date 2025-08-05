import math
import time
from typing import List, Tuple

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi

from ..models import Vector
from ..synapso_logger import get_logger
from .interface import Reranker

logger = get_logger(__name__)


class BM25Reranker(Reranker):
    """
    BM25 (Best Matching 25) reranker implementation using the rank_bm25 library.

    BM25 is a ranking function used by search engines to rank documents based on their relevance
    to a given search query. It's an improvement over the TF-IDF algorithm.
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        remove_stopwords: bool = True,
        lemmatize: bool = True,
    ):
        """
        Initialize BM25 reranker.

        Args:
            k1: Controls term frequency scaling (default: 1.5)
            b: Controls length normalization (default: 0.75)
            remove_stopwords: Whether to remove stopwords (default: True)
            lemmatize: Whether to lemmatize tokens (default: True)
        """
        self.k1 = k1
        self.b = b
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self._bm25 = None
        self._documents = []

        # Initialize NLTK components
        self._initialize_nltk()

        # Initialize tokenization components
        if self.remove_stopwords:
            try:
                self._stopwords = set(stopwords.words("english"))
            except LookupError:
                nltk.download("stopwords")
                self._stopwords = set(stopwords.words("english"))
        else:
            self._stopwords = set()

        if self.lemmatize:
            try:
                self._lemmatizer = WordNetLemmatizer()
            except LookupError:
                nltk.download("wordnet")
                self._lemmatizer = WordNetLemmatizer()
        else:
            self._lemmatizer = None

    def _initialize_nltk(self):
        """Initialize NLTK components if needed."""
        try:
            # Test if punkt tokenizer is available
            word_tokenize("test")
        except LookupError:
            nltk.download("punkt")

    def _tokenize(self, text: str) -> List[str]:
        """
        Advanced tokenization using NLTK for BM25.

        Args:
            text: Input text to tokenize

        Returns:
            List of processed tokens
        """
        if not text:
            return []

        # Tokenize using NLTK's word_tokenize
        tokens = word_tokenize(text.lower())

        # Remove punctuation and non-alphabetic tokens
        tokens = [token for token in tokens if token.isalpha()]

        # Remove stopwords if enabled
        if self.remove_stopwords:
            tokens = [token for token in tokens if token not in self._stopwords]

        # Lemmatize if enabled
        if self.lemmatize and self._lemmatizer:
            tokens = [self._lemmatizer.lemmatize(token) for token in tokens]

        return tokens

    def _build_bm25_index(self, texts: List[str]) -> None:
        """
        Build BM25 index from texts.

        Args:
            texts: List of text documents
        """
        self._documents = []

        for text in texts:
            if text:
                tokens = self._tokenize(text)
                self._documents.append(tokens)
            else:
                # Add empty document for empty texts
                self._documents.append([])

        # Create BM25 index
        self._bm25 = BM25Okapi(self._documents, k1=self.k1, b=self.b)

    def rerank(
        self,
        results: List[Tuple[Vector, str, float]],
        query: Vector,
        query_text: str = "",
    ) -> List[Tuple[Vector, str, float]]:
        """
        Rerank results using BM25 algorithm.

        Args:
            results: List of (vector, text, score) tuples from initial search
            query: Query vector
            query_text: Query text for BM25 scoring (optional, will use first result text if not provided)

        Returns:
            Reranked list of (vector, text, score) tuples
        """
        if not results:
            return results

        # Extract texts from results
        texts = [text for _, text, _ in results]

        # Build BM25 index from the result texts
        start_time = time.time()
        self._build_bm25_index(texts)
        logger.info("BM25 index built in %s seconds", time.time() - start_time)

        # Use provided query text or fallback to first result text
        if not query_text and texts:
            query_text = texts[0]  # Fallback

        if not query_text:
            return results

        start_time = time.time()
        query_tokens = self._tokenize(query_text)
        logger.info("Query tokens tokenized in %s seconds", time.time() - start_time)
        if not query_tokens:
            return results

        # Get BM25 scores
        if self._bm25 is not None:
            start_time = time.time()
            bm25_scores = self._bm25.get_scores(query_tokens)
            logger.info(
                "BM25 scores calculated in %s seconds", time.time() - start_time
            )
        else:
            # Fallback if BM25 index couldn't be built
            return results

        # Combine BM25 scores with original scores
        start_time = time.time()
        reranked_results = []
        for i, (vector, text, original_score) in enumerate(results):
            bm25_score = bm25_scores[i] if i < len(bm25_scores) else 0.0

            # Normalize BM25 score to 0-1 range and combine with original score
            # BM25 scores can be negative, so we use a sigmoid-like normalization
            normalized_bm25 = 1.0 / (1.0 + math.exp(-bm25_score))

            # Combine scores (you can adjust this weighting)
            combined_score = 0.7 * normalized_bm25 + 0.3 * original_score
            reranked_results.append((vector, text, combined_score))

        # Sort by combined score in descending order
        reranked_results.sort(key=lambda x: x[2], reverse=True)
        logger.info("Reranked results sorted in %s seconds", time.time() - start_time)
        return reranked_results
