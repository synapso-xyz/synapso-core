import json
import logging
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import sqlite_vss

from ....chunking.factory import ChunkerFactory
from ....config_manager import get_config
from ....sqlite_utils import SqliteEngineMixin, create_sqlite_db_if_not_exists
from ....vectorizer.factory import VectorizerFactory
from ...interfaces.vector_store import Vector, VectorMetadata, VectorStore

logger = logging.getLogger(__name__)


class SqliteVectorMetadata(VectorMetadata):
    def __init__(self, content_hash: str, additional_data: Dict):
        self.content_hash = content_hash
        self.additional_data = additional_data

    def to_dict(self) -> Dict:
        return {
            "content_hash": self.content_hash,
            "additional_data": self.additional_data,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "VectorMetadata":
        return cls(
            content_hash=data.get("content_hash", ""),
            additional_data=data.get("additional_data", {}),
        )


class VectorSqliteAdapter(SqliteEngineMixin, VectorStore):
    def __init__(self):
        # Initialize with a placeholder path, will be set in metastore_setup
        config = get_config()
        if config.vector_store.vector_db_type != "sqlite":
            raise ValueError("Vector store type is not sqlite")
        self.vector_db_path = config.vector_store.vector_db_path
        self.vector_db_path = str(Path(self.vector_db_path).expanduser().resolve())
        print(f"Vector DB path: {self.vector_db_path}")
        SqliteEngineMixin.__init__(self, self.vector_db_path)
        self.vectorstore_setup()

    def _get_connection(self):
        conn = sqlite3.connect(self.vector_db_path)
        conn.enable_load_extension(True)
        sqlite_vss.load(conn)
        conn.enable_load_extension(False)
        return conn

    def __del__(self):
        """Clean up database connection when the adapter is deleted."""
        self.close()

    def close(self):
        """Explicitly close the database connection."""
        ...

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
        conn = self._get_connection()
        conn.set_trace_callback(print)
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
            print("from metadata", cursor.fetchone())
            cursor.execute(
                "SELECT 1 FROM vss_vectors WHERE embedding = ?",
                (embedding_bytes,),
            )
            print("from vss_vectors", cursor.fetchone())
            if cursor.fetchone() is not None:
                logger.info(f"Vector with content_hash '{vector_id}' already exists.")
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
            print(f"max_rowid: {max_rowid}")
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

            # Insert metadata, linking it via vector_row_id
            # NOTE: Removed redundant storage of the embedding itself
            cursor.execute(
                "INSERT INTO metadata(vector_row_id, embedding, content_hash, metadata) VALUES (?, ?, ?, ?)",
                (vector_row_id, embedding_bytes, vector_id, metadata_json),
            )

            # Commit the transaction
            conn.commit()
            logger.info(
                f"Successfully inserted vector with content_hash '{vector_id}'."
            )
            return True

        except sqlite3.Error as e:
            logger.error(f"Database error during insert: {e}")
            conn.rollback()  # Roll back the transaction on error
            return False
        finally:
            conn.close()

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
            metadata = result_metadata[2]

            retrieved_vector = np.frombuffer(embedding, dtype=np.float32).tolist()
            return Vector(content_hash, retrieved_vector, metadata)

        finally:
            conn.close()

    def vector_search(
        self, query_vector: Vector, top_k: int = 5, filters: Dict | None = None
    ) -> List[Tuple[Vector, float]]:
        conn = self._get_connection()
        try:
            query_embeddings = query_vector.vector
            query_embeddings_np = np.array(query_embeddings, dtype=np.float32)
            query_embedding_json = json.dumps(query_embeddings_np.tolist())

            print(f"top_k: {top_k}")
            print(f"query length: {len(query_embeddings)}")
            print(f"query_np_dim: {query_embeddings_np.shape}")
            # print(f"query_embedding_json: {query_embedding_json}")
            # Check if VSS is available by trying to use it
            search_stmt = """
                select m.content_hash, v.distance
                from vss_vectors v
                join metadata m on v.rowid = m.vector_row_id
                where vss_search(v.embedding, vss_search_params(json(?), ?))
                order by v.distance
            """
            try:
                results = conn.execute(
                    search_stmt, (query_embedding_json, top_k)
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
        finally:
            conn.close()

    def delete(self, vector_id: str) -> bool:
        raise NotImplementedError("Not implemented")

    def update_metadata(self, vector_id: str, metadata: VectorMetadata) -> bool:
        raise NotImplementedError("Not implemented")

    def count(self) -> int:
        raise NotImplementedError("Not implemented")

    def vectorstore_teardown(self) -> bool:
        raise NotImplementedError("Not implemented")

    def _setup_tables(self) -> None:
        conn = self._get_connection()

        # Try to load VSS extension, but handle cases where it's not available
        try:
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
        finally:
            conn.close()

    # Add this new method to your VectorSqliteAdapter class
    def run_core_test(self):
        """A simple, self-contained test to verify core functionality."""
        print("\n--- RUNNING CORE VSS TEST ---")
        conn = self._get_connection()
        conn.set_trace_callback(print)

        cursor = conn.cursor()

        test_vector_table_stmt = """
            CREATE VIRTUAL TABLE IF NOT EXISTS test_vectors USING vss0(
                embedding(384)
            )
        """
        test_metadata_table_stmt = """
            CREATE TABLE IF NOT EXISTS test_metadata (
                vector_row_id INTEGER PRIMARY KEY,
                embedding BLOB NOT NULL,
                content_hash TEXT UNIQUE NOT NULL,
                metadata TEXT
            )
        """
        try:
            cursor.execute(test_vector_table_stmt)
            cursor.execute(test_metadata_table_stmt)
        except Exception as e:
            logger.exception(e)
            raise e

        try:
            chunker = ChunkerFactory.create_chunker("chonkie_recursive")
            chunk_text = "This is a test chunk"
            chunks = chunker.chunk_text(chunk_text)
            for chunk in chunks:
                print(chunk.text)
                print(chunk.metadata)
                print("-" * 100)

            vectorizer = VectorizerFactory.create_vectorizer("sentence_transformer")
            vectors = vectorizer.vectorize_batch(chunks)

            test_vector_obj = vectors[0]
            print(test_vector_obj.vector_id)
            print(test_vector_obj.metadata)
            print(f"vector size: {len(test_vector_obj.vector)}")
            print(f"vector dtype: {type(test_vector_obj.vector[0])}")
            print("-" * 100)

            print("SQLite library version:", sqlite3.sqlite_version)

            # Use your existing insert logic
            vector_id = test_vector_obj.vector_id
            embedding = test_vector_obj.vector
            metadata = test_vector_obj.metadata

            embedding_np = np.array(embedding, dtype=np.float32)
            embedding_bytes = embedding_np.tobytes()

            # 2. Prepare metadata
            metadata_json = json.dumps(metadata.to_dict()) if metadata else None

            conn.commit()
            self.insert(test_vector_obj)
            print(" -> Insert function completed.")

            # 2. Prepare a search vector
            print("\nStep 2: Preparing search vector...")
            query_vec = np.array(test_vector_obj.vector, dtype=np.float32)
            query_bytes = query_vec.tobytes()
            top_k = 1

            # 3. Use the SIMPLEST possible search query (no JOINs)
            print("Step 3: Executing bare-bones search query...")
            simple_search_stmt = """
                SELECT rowid, distance
                FROM vectors
                WHERE vss_search(embedding, vss_search_params(?, ?))
            """
            results = cursor.execute(
                simple_search_stmt, (query_bytes, top_k)
            ).fetchall()

            # 4. Report Result
            print("\n--- CORE TEST COMPLETE ---")
            if len(results) > 0:
                print(
                    f"✅✅✅ SUCCESS: The core VSS search is working. Found {len(results)} result(s)."
                )
            else:
                # This case can happen if there's no result, but it shouldn't crash.
                print("⚠️ SUCCESS: The search ran without crashing but found 0 results.")

            return True

        except Exception as e:
            print("\n--- CORE TEST FAILED ---")
            print(f"❌❌❌ FAILED: The test crashed. Error: {e}")
            logger.exception(e)
            return False
        finally:
            conn.close()
