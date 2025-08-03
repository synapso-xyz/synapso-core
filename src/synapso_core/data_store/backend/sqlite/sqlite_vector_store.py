import json
import logging
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import sqlite_vss

from ....config_manager import get_config
from ....models import Vector, VectorMetadata
from ...interfaces import VectorStore
from .sqlite_backend_identifier import SqliteBackendIdentifierMixin
from .utils import SqliteEngineMixin, create_sqlite_db_if_not_exists

logger = logging.getLogger(__name__)


class SqliteVectorStore(SqliteEngineMixin, VectorStore, SqliteBackendIdentifierMixin):
    def __init__(self):
        # Initialize with a placeholder path, will be set in metastore_setup
        config = get_config()
        if config.vector_store.vector_db_type != "sqlite":
            raise ValueError("Vector store type is not sqlite")
        self.vector_db_path = config.vector_store.vector_db_path
        self.vector_db_path = str(Path(self.vector_db_path).expanduser().resolve())
        SqliteEngineMixin.__init__(self, self.vector_db_path)
        self._conn = None

    def _get_connection(self):
        if self._conn is None:
            conn = sqlite3.connect(self.vector_db_path)
            conn.enable_load_extension(True)
            sqlite_vss.load(conn)
            conn.enable_load_extension(False)
            self._conn = conn
        return self._conn

    def __del__(self):
        """Clean up database connection when the adapter is deleted."""
        self.close()

    def close(self):
        """Explicitly close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def setup(self) -> bool:
        create_sqlite_db_if_not_exists(self.vector_db_path)
        self._setup_tables()
        return True

    def insert(self, vector: Vector) -> bool:
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            vector_id = vector.vector_id
            embedding = vector.vector
            metadata = vector.metadata

            embedding_np = np.array(embedding, dtype=np.float32)
            embedding_bytes = embedding_np.tobytes()
            # Check if the vector's metadata already exists to prevent duplicates
            cursor.execute(
                "SELECT 1 FROM metadata WHERE content_hash = ?",
                (vector_id,),
            )
            cursor.execute(
                "SELECT 1 FROM vss_vectors WHERE embedding = ?",
                (embedding_bytes,),
            )
            if cursor.fetchone() is not None:
                logger.info("Vector with content_hash '%s' already exists.", vector_id)
                return True

            # 1. Validate and prepare vector data
            if len(embedding) != 384:  # Assuming dimension is 384
                raise ValueError(
                    f"Vector dimension is {len(embedding)}, but must be 384."
                )

            # 2. Prepare metadata
            metadata_json = json.dumps(metadata.to_dict()) if metadata else None

            cursor.execute("SELECT max(rowid) FROM vss_vectors")
            max_rowid = cursor.fetchone()[0]
            if max_rowid is None:
                max_rowid = 0
            vector_row_id = max_rowid + 1

            # 3. Perform inserts within a single transaction
            # Insert into the VSS table
            cursor.execute(
                "INSERT INTO vss_vectors(rowid, embedding) VALUES (?, ?)",
                (vector_row_id, embedding_bytes),
            )

            # Correctly get the last row id from the cursor
            vector_row_id = cursor.lastrowid

            cursor.execute(
                "INSERT INTO metadata(vector_row_id, embedding, content_hash, metadata) VALUES (?, ?, ?, ?)",
                (vector_row_id, embedding_bytes, vector_id, metadata_json),
            )

            # Commit the transaction
            conn.commit()
            logger.info(
                "Successfully inserted vector with content_hash '%s'.", vector_id
            )
            return True

        except sqlite3.Error as e:
            logger.error("Database error during insert: %s", e)
            conn.rollback()  # Roll back the transaction on error
            return False

    def get_by_id(self, vector_id: str) -> Vector | None:
        conn = self._get_connection()
        try:
            get_metadata_stmt = """
                SELECT embedding, content_hash, metadata FROM metadata WHERE content_hash = ?
            """
            result_metadata = conn.execute(get_metadata_stmt, (vector_id,)).fetchone()
            if result_metadata is None:
                return None
            embedding = result_metadata[0]
            content_hash = result_metadata[1]
            metadata_json = result_metadata[2]

            retrieved_vector = np.frombuffer(embedding, dtype=np.float32).tolist()

            # Deserialize metadata if it exists
            metadata = None
            if metadata_json:
                metadata_dict = json.loads(metadata_json)
                metadata = VectorMetadata.from_dict(metadata_dict)

            return Vector(content_hash, retrieved_vector, metadata)
        except Exception as e:
            logger.exception(e)
            raise e

    def vector_search(
        self, query_vector: Vector, top_k: int = 5, filters: Dict | None = None
    ) -> List[Tuple[Vector, float]]:
        conn = self._get_connection()
        try:
            query_embeddings = query_vector.vector
            query_embeddings_np = np.array(query_embeddings, dtype=np.float32)
            query_embedding_bytes = query_embeddings_np.tobytes()

            # Check if VSS is available by trying to use it
            search_stmt = """
                select m.content_hash, v.distance
                from vss_vectors v
                join metadata m on v.rowid = m.vector_row_id
                where vss_search(v.embedding, vss_search_params(?, ?))
                order by v.distance
            """
            try:
                results = conn.execute(
                    search_stmt, (query_embedding_bytes, top_k)
                ).fetchall()
            except Exception as e:
                logger.exception(e)
                raise e

            similar_vectors = []
            for vector_id, score in results:
                vector = self.get_by_id(vector_id)
                if vector is not None:
                    similar_vectors.append((vector, score))
            return similar_vectors
        except Exception as e:
            logger.exception(e)
            raise e

    def delete(self, vector_id: str) -> bool:
        raise NotImplementedError("Not implemented")

    def update_metadata(self, vector_id: str, metadata: VectorMetadata) -> bool:
        raise NotImplementedError("Not implemented")

    def count(self) -> int:
        raise NotImplementedError("Not implemented")

    def teardown(self) -> bool:
        raise NotImplementedError("Not implemented")

    def _setup_tables(self) -> None:
        conn = self._get_connection()

        # Try to load VSS extension, but handle cases where it's not available

        create_vector_table_stmt = """
            CREATE VIRTUAL TABLE IF NOT EXISTS vss_vectors USING vss0(
                embedding(384)
            )
        """
        try:
            conn.execute(create_vector_table_stmt)
        except Exception as e:
            logger.exception(e)
            raise e

        create_metadata_table_stmt = """
            CREATE TABLE IF NOT EXISTS metadata (
                vector_row_id INTEGER PRIMARY KEY,
                embedding BLOB NOT NULL,
                content_hash TEXT UNIQUE NOT NULL,
                metadata TEXT
            )
        """
        try:
            conn.execute(create_metadata_table_stmt)
            conn.commit()
        except Exception as e:
            logger.exception(e)
            raise e
