import json
import sqlite3
from typing import Dict, List, Optional, Tuple

import numpy as np
import sqlite_vss

from ....config_manager import get_config
from ....sqlite_utils import SqliteEngineMixin, create_sqlite_db_if_not_exists
from ...interfaces.vector_store import Vector, VectorMetadata, VectorStore


class VectorSqliteAdapter(SqliteEngineMixin, VectorStore):
    def __init__(self):
        # Initialize with a placeholder path, will be set in metastore_setup
        config = get_config()
        if config.vector_store.vector_db_type != "sqlite":
            raise ValueError("Vector store type is not sqlite")
        self.vector_db_path = config.vector_store.vector_db_path
        SqliteEngineMixin.__init__(self, self.vector_db_path)
        self._db: Optional[sqlite3.Connection] = sqlite3.connect(self.vector_db_path)

    def __del__(self):
        """Clean up database connection when the adapter is deleted."""
        self.close()

    def close(self):
        """Explicitly close the database connection."""
        if hasattr(self, "_db") and self._db is not None:
            try:
                self._db.close()
                self._db = None
            except Exception:
                # Ignore errors during cleanup
                pass

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def vectorstore_setup(self) -> bool:
        create_sqlite_db_if_not_exists(self.vector_db_path)
        self._setup_tables()
        return True

    def insert(self, vector: Vector) -> bool:
        if self._db is None:
            raise RuntimeError("Database connection is not available")

        vector_id = vector.vector_id
        embedding = vector.vector
        embedding_bytes = np.array(embedding, dtype=np.float32).tobytes()
        metadata = vector.metadata

        # Handle metadata serialization
        if metadata is not None:
            metadata_str = json.dumps(metadata.to_dict())
        else:
            metadata_str = None

        insert_vector_stmt = """
            INSERT INTO vectors (vector_id, vector) VALUES (?, ?)
        """
        self._db.execute(insert_vector_stmt, (vector_id, embedding_bytes))

        insert_metadata_stmt = """
            INSERT INTO metadata (vector_id, metadata) VALUES (?, ?)
        """
        self._db.execute(insert_metadata_stmt, (vector_id, metadata_str))

        self._db.commit()
        return True

    def get_by_id(self, vector_id: str) -> Vector | None:
        if self._db is None:
            raise RuntimeError("Database connection is not available")

        get_vector_stmt = """
            SELECT vector_id, vector FROM vectors WHERE vector_id = ?
        """
        get_metadata_stmt = """
            SELECT metadata FROM metadata WHERE vector_id = ?
        """
        result_vector = self._db.execute(get_vector_stmt, (vector_id,)).fetchone()
        result_metadata = self._db.execute(get_metadata_stmt, (vector_id,)).fetchone()
        if result_vector:
            vector_id, vector_bytes = result_vector
            vector = np.frombuffer(vector_bytes, dtype=np.float32).tolist()
            if result_metadata is None or result_metadata[0] is None:
                metadata = None
            else:
                metadata_dict = json.loads(result_metadata[0])
                # For now, we'll create a simple metadata object
                # In a real implementation, you'd need to know the concrete class
                metadata = type(
                    "SimpleMetadata",
                    (VectorMetadata,),
                    {
                        "content_hash": metadata_dict.get("content_hash", ""),
                        "additional_data": {
                            k: v
                            for k, v in metadata_dict.items()
                            if k != "content_hash"
                        },
                        "to_dict": lambda self: metadata_dict,
                        "from_dict": classmethod(lambda cls, data: cls()),
                    },
                )()
            return Vector(vector_id, vector, metadata)
        return None

    def vector_search(
        self, query_vector: Vector, top_k: int = 5, filters: Dict | None = None
    ) -> List[Tuple[Vector, float]]:
        if self._db is None:
            raise RuntimeError("Database connection is not available")

        query_embeddings = query_vector.vector
        query_embedding_bytes = np.array(query_embeddings, dtype=np.float32).tobytes()

        # Check if VSS is available by trying to use it
        try:
            search_stmt = """
                SELECT vector_id, vss_score FROM vectors WHERE vss_search(vector, ?) LIMIT ?
            """
            results = self._db.execute(
                search_stmt, (query_embedding_bytes, top_k)
            ).fetchall()

            similar_vectors = []
            for vector_id, score in results:
                vector = self.get_by_id(vector_id)
                if vector is not None:
                    similar_vectors.append((vector, score))
            return similar_vectors
        except Exception:
            # Fallback to simple retrieval if VSS is not available
            # This is a basic implementation that just returns vectors without similarity scoring
            search_stmt = """
                SELECT vector_id FROM vectors LIMIT ?
            """
            results = self._db.execute(search_stmt, (top_k,)).fetchall()

            similar_vectors = []
            for (vector_id,) in results:
                vector = self.get_by_id(vector_id)
                if vector is not None:
                    similar_vectors.append((vector, 1.0))  # Default score of 1.0
            return similar_vectors

    def delete(self, vector_id: str) -> bool:
        raise NotImplementedError("Not implemented")

    def update_metadata(self, vector_id: str, metadata: VectorMetadata) -> bool:
        raise NotImplementedError("Not implemented")

    def count(self) -> int:
        raise NotImplementedError("Not implemented")

    def vectorstore_teardown(self) -> bool:
        raise NotImplementedError("Not implemented")

    def _setup_tables(self) -> None:
        if self._db is None:
            raise RuntimeError("Database connection is not available")

        # Try to load VSS extension, but handle cases where it's not available
        try:
            if hasattr(self._db, "enable_load_extension"):
                self._db.enable_load_extension(True)
                sqlite_vss.load(self._db)
                self._db.enable_load_extension(False)

                # Verify VSS is working
                (version,) = self._db.execute("select vss_version()").fetchone()

                create_vector_table_stmt = """
                    CREATE VIRTUAL TABLE IF NOT EXISTS vectors USING vss0(
                        vector_id TEXT PRIMARY KEY,
                        vector(384)
                    )
                """
                self._db.execute(create_vector_table_stmt)
            else:
                # Fallback to regular table if VSS is not available
                create_vector_table_stmt = """
                    CREATE TABLE IF NOT EXISTS vectors (
                        vector_id TEXT PRIMARY KEY,
                        vector BLOB
                    )
                """
                self._db.execute(create_vector_table_stmt)
        except Exception:
            # Fallback to regular table if VSS extension fails to load
            create_vector_table_stmt = """
                CREATE TABLE IF NOT EXISTS vectors (
                    vector_id TEXT PRIMARY KEY,
                    vector BLOB
                )
            """
            self._db.execute(create_vector_table_stmt)

        create_metadata_table_stmt = """
            CREATE TABLE IF NOT EXISTS metadata (
                vector_id TEXT PRIMARY KEY,
                metadata TEXT
            )
        """
        self._db.execute(create_metadata_table_stmt)
        self._db.commit()
