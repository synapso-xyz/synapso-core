import traceback as tb
from pathlib import Path
from typing import Dict, Tuple

from ..chunking.factory import ChunkerFactory
from ..config_manager import get_config
from ..data_store.factory import DataStoreFactory
from ..synapso_logger import get_logger
from ..vectorizer.factory import VectorizerFactory

logger = get_logger(__name__)


class DocumentIngestor:
    def __init__(self):
        global_config = get_config()

        chunker_type = global_config.chunker.chunker_type
        self.chunker = ChunkerFactory.create_chunker(chunker_type)

        vectorizer_type = global_config.vectorizer.vectorizer_type
        self.vectorizer = VectorizerFactory.create_vectorizer(vectorizer_type)

        meta_store_type = global_config.meta_store.meta_db_type
        self.meta_store = DataStoreFactory.get_meta_store(meta_store_type)

        vector_store_type = global_config.vector_store.vector_db_type
        self.vector_store = DataStoreFactory.get_vector_store(vector_store_type)

        private_store_type = global_config.private_store.private_db_type
        self.private_store = DataStoreFactory.get_private_store(private_store_type)

    def ingest_file(
        self, file_path: Path, file_version_id: str
    ) -> Tuple[bool, Dict | None]:
        try:
            logger.info("Ingesting %s", file_path)
            chunks = self.chunker.chunk_file(str(file_path))

            chunk_ids = []
            logger.info("Inserting %d chunks into private store", len(chunks))
            for chunk in chunks:
                chunk_id = self.private_store.insert(chunk.text)
                chunk_ids.append(chunk_id)

            self.meta_store.assosiate_chunks(file_version_id, chunk_ids)

            logger.info("Vectorizing %d chunks", len(chunks))
            vectors = self.vectorizer.vectorize_batch(chunks)

            logger.info("Inserting %d vectors into vector store", len(vectors))
            for v in vectors:
                self.vector_store.insert(v)

            return True, None
        except Exception as e:
            traceback = tb.format_exc()
            error_context = {"error_type": str(e), "traceback": traceback}
            logger.error("Error ingesting file %s: %s", file_path, traceback)
            return False, error_context
