import traceback as tb
from pathlib import Path
from typing import Dict, Tuple

from ..chunking.factory import ChunkerFactory
from ..config_manager import get_config
from ..persistence.factory import VectorStoreFactory
from ..vectorizer.factory import VectorizerFactory


async def ingest_file(file_path: Path) -> Tuple[bool, Dict | None]:
    global_config = get_config()

    chunker_type = global_config.chunker.chunker_type
    chunker = ChunkerFactory.create_chunker(chunker_type)

    vectorizer_type = global_config.vectorizer.vectorizer_type
    vectorizer = VectorizerFactory.create_vectorizer(vectorizer_type)

    vector_store_type = global_config.vector_store.vector_db_path
    vector_store = VectorStoreFactory.get_vector_store(vector_store_type)

    try:
        chunks = chunker.chunk_file(str(file_path))
        vectors = vectorizer.vectorize_batch(chunks)

        for v in vectors:
            vector_store.insert(v)

        return True, None
    except Exception as e:
        traceback = tb.extract_stack()
        error_context = {"error_type": str(e), "traceback": traceback}
        return False, error_context
