import traceback as tb
from pathlib import Path
from typing import Dict, Tuple

from ..chunking.factory import ChunkerFactory
from ..config_manager import get_config
from ..data_store.factory import DataStoreFactory
from ..synapso_logger import get_logger
from ..vectorizer.factory import VectorizerFactory

logger = get_logger(__name__)


def ingest_file(file_path: Path) -> Tuple[bool, Dict | None]:
    try:
        global_config = get_config()

        chunker_type = global_config.chunker.chunker_type
        chunker = ChunkerFactory.create_chunker(chunker_type)

        vectorizer_type = global_config.vectorizer.vectorizer_type
        vectorizer = VectorizerFactory.create_vectorizer(vectorizer_type)

        vector_store_type = global_config.vector_store.vector_db_type
        vector_store = DataStoreFactory.get_vector_store(vector_store_type)

        private_store_type = global_config.private_store.private_db_type
        private_store = DataStoreFactory.get_private_store(private_store_type)

        logger.info("Ingesting %s", file_path)
        chunks = chunker.chunk_file(str(file_path))

        logger.info("Inserting %d chunks into private store", len(chunks))
        for chunk in chunks:
            private_store.insert(chunk.text)

        logger.info("Vectorizing %d chunks", len(chunks))
        vectors = vectorizer.vectorize_batch(chunks)

        logger.info("Inserting %d vectors into vector store", len(vectors))
        for v in vectors:
            vector_store.insert(v)

        return True, None
    except Exception as e:
        traceback = tb.format_exc()
        error_context = {"error_type": str(e), "traceback": traceback}
        return False, error_context
