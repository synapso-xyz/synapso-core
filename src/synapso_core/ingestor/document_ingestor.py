import csv
import json
import os
import traceback as tb
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, Iterator, Tuple

from ..chunking.factory import ChunkerFactory
from ..config_manager import get_config
from ..data_store.data_models import DBFile, DBFileVersion, IndexingJob
from ..data_store.factory import DataStoreFactory
from ..synapso_logger import get_logger
from ..vectorizer.factory import VectorizerFactory

logger = get_logger(__name__)


SUPPORTED_FORMATS = [".md", ".markdown", ".txt"]


class FileState(Enum):
    ELIGIBLE = 0
    HIDDEN_FILE = 1
    HIDDEN_DIRECTORY = 2
    UNSUPPORTED_FORMAT = 3


def _classify(path: Path, root: Path):
    rel_parts = path.relative_to(root).parts
    dir_parts = rel_parts[:-1]
    file_name = rel_parts[-1]

    # 1) Any directory in the chain is hidden → HIDDEN_DIRECTORY
    if any(part.startswith(".") for part in dir_parts):
        return FileState.HIDDEN_DIRECTORY

    # 2) File itself starts with '.' but parent dirs are visible → HIDDEN_FILE
    if file_name.startswith("."):
        return FileState.HIDDEN_FILE

    # 3) Visible file but not markdown → UNSUPPORTED_FORMAT
    if path.suffix.lower() not in SUPPORTED_FORMATS:
        return FileState.UNSUPPORTED_FORMAT

    # 4) Otherwise → ELIGIBLE
    return FileState.ELIGIBLE


@dataclass
class FileRecord:
    path: Path
    state: FileState
    file_name: str
    file_size: int
    file_type: str
    file_created_at: datetime
    file_updated_at: datetime
    db_file: DBFile | None
    db_file_version: DBFileVersion | None

    def __init__(self, path: Path, state: FileState):
        if not path.exists() or not path.is_file():
            raise ValueError(f"Path {path} is not a file")

        self.path = path.expanduser().resolve()
        self.state = state

        stat = path.stat()
        self.file_name = path.name
        self.file_size = stat.st_size  # in bytes
        self.file_type = path.suffix.lower()
        # Note: st_ctime is change time on POSIX. Prefer st_birthtime if present.
        created_ts = getattr(stat, "st_birthtime", stat.st_ctime)
        self.file_created_at = datetime.fromtimestamp(created_ts, timezone.utc)
        self.file_updated_at = datetime.fromtimestamp(stat.st_mtime, timezone.utc)
        self.db_file = None
        self.db_file_version = None


def _get_file_list_path(directory_path: str, ensure_present=True) -> Path:
    file_list_path = Path(directory_path) / ".synapso" / "file_list.csv"
    if ensure_present:
        file_list_path.parent.mkdir(exist_ok=True, parents=True)
        file_list_path.touch()
    return file_list_path


def _get_ingestion_errors_path(directory_path: str, ensure_present=True) -> Path:
    ingestion_errors_path = Path(directory_path) / ".synapso" / "ingestion_errors.jsonl"
    if ensure_present:
        ingestion_errors_path.parent.mkdir(exist_ok=True, parents=True)
        ingestion_errors_path.touch()
    return ingestion_errors_path


def _file_walk(directory_path: str) -> Iterator[FileRecord]:
    root = Path(directory_path).expanduser().resolve()
    for dirpath, _, filenames in os.walk(root):
        base = Path(dirpath)
        for fname in filenames:
            fpath = base / fname
            yield FileRecord(fpath, state=_classify(fpath, root))


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
            file_version = self.meta_store.get_file_version_by_id(file_version_id)  # type: ignore[attr-defined]
            if file_version is None:
                raise ValueError(f"File version {file_version_id} not found")
            cortex_id = file_version.cortex_id
            source_file_id = file_version.file_id

            logger.info("Ingesting %s", file_path)
            chunks = self.chunker.chunk_file(str(file_path))

            chunk_ids = []
            logger.info("Inserting %d chunks into private store", len(chunks))
            for chunk in chunks:
                # Ensure chunk metadata carries linkage for vector metadata
                if isinstance(chunk.metadata, dict):
                    chunk.metadata.setdefault("cortex_id", cortex_id)
                    chunk.metadata.setdefault("source_file_id", source_file_id)
                chunk_id = self.private_store.insert(chunk.text)
                if isinstance(chunk.metadata, dict):
                    chunk.metadata["chunk_id"] = chunk_id
                else:
                    chunk.metadata = {
                        "cortex_id": cortex_id,
                        "source_file_id": source_file_id,
                        "chunk_id": chunk_id,
                    }
                chunk_ids.append(chunk_id)

            # De-duplicate to respect (cortex_id, file_version_id, chunk_id) uniqueness
            unique_chunk_ids = list(dict.fromkeys(chunk_ids))
            self.meta_store.associate_chunks(file_version_id, unique_chunk_ids)

            logger.info("Vectorizing %d chunks", len(chunks))
            vectors = self.vectorizer.vectorize_batch(chunks)

            logger.info("Inserting %d vectors into vector store", len(vectors))
            failed_vectors = []
            for v in vectors:
                ok = self.vector_store.insert(v)
                if not ok:
                    logger.warning("Failed to insert vector for chunk %s", v.vector_id)
                    failed_vectors.append(v)

            if failed_vectors:
                return False, {
                    "error_type": "Failed to insert vector",
                    "failed_vectors": [v.vector_id for v in failed_vectors],
                    "n_failed_vectors": len(failed_vectors),
                    "file_path": str(file_path),
                }

            return True, None
        except Exception as e:
            traceback = tb.format_exc()
            error_context = {
                "error_type": str(e),
                "traceback": traceback,
                "file_path": str(file_path),
                "file_version_id": file_version_id,
            }
            logger.error("Error ingesting file %s: %s", file_path, traceback)
            return False, error_context


class CortexIngestor:
    def __init__(self):
        global_config = get_config()

        meta_store_type = global_config.meta_store.meta_db_type
        self.meta_store = DataStoreFactory.get_meta_store(meta_store_type)

        self.document_ingestor = DocumentIngestor()

    def ingest_cortex(self, cortex_id: str, job_id: str | None = None) -> bool:
        """
        Ingest a cortex.
        """

        cortex = self.meta_store.get_cortex_by_id(cortex_id)
        if cortex is None:
            raise ValueError(f"Cortex with id {cortex_id} not found")

        if not job_id:
            job_id = uuid.uuid4().hex

        db_indexing_job = IndexingJob(
            job_id=job_id,
            cortex_id=cortex_id,
            job_type="ingest_cortex",
            job_status="PREPARING",
            n_total_files=0,
            n_eligible_files=0,
            files_processed=0,
            job_start_time=datetime.now(timezone.utc),
            job_end_time=None,
        )
        db_indexing_job = self.meta_store.create_indexing_job(db_indexing_job)

        try:
            # For now, let is just trigger the indexing.
            cortex_path = cortex.path
            synapso_dir_path = Path(cortex_path) / ".synapso"
            synapso_dir_path.mkdir(exist_ok=True, parents=True)

            file_list_path = _get_file_list_path(directory_path=cortex_path)
            ingestion_errors_path = _get_ingestion_errors_path(
                directory_path=cortex_path
            )

            has_errors = False

            n_total_files = 0
            n_eligible_files = 0

            file_records = []
            for file_record in _file_walk(cortex_path):
                file_path = file_record.path
                file_eligibility = file_record.state

                n_total_files += 1

                if file_eligibility == FileState.ELIGIBLE:
                    n_eligible_files += 1

                db_file = DBFile(
                    cortex_id=cortex.cortex_id,
                    file_id=uuid.uuid4().hex,
                    file_name=file_record.file_name,
                    file_path=str(file_path),
                    file_size=file_record.file_size,
                    file_type=file_record.file_type,
                    file_created_at=file_record.file_created_at,
                    file_updated_at=file_record.file_updated_at,
                    eligibility_status=file_eligibility.name.lower(),
                    last_indexed_at=None,
                )
                db_file = self.meta_store.create_file(db_file)

                db_file_version = DBFileVersion(
                    cortex_id=cortex.cortex_id,
                    file_version_id=uuid.uuid4().hex,
                    file_id=db_file.file_id,
                    file_version_created_at=file_record.file_created_at,
                    file_version_invalid_at=None,
                    file_version_is_valid=True,
                )
                db_file_version = self.meta_store.create_file_version(db_file_version)
                file_record.db_file = db_file
                file_record.db_file_version = db_file_version
                file_records.append(file_record)

            db_indexing_job.n_total_files = n_total_files
            db_indexing_job.n_eligible_files = n_eligible_files
            db_indexing_job.job_status = "IN_PROGRESS"
            db_indexing_job = self.meta_store.update_indexing_job(db_indexing_job)

            with (
                file_list_path.open("w", newline="", encoding="utf-8") as f,
                ingestion_errors_path.open(
                    "w", newline="", encoding="utf-8"
                ) as err_file,
            ):
                writer = csv.writer(f)
                writer.writerow(["path", "eligibility"])
                for file_record in file_records:
                    file_path = file_record.path
                    file_eligibility = file_record.state

                    writer.writerow(
                        [str(file_path), str(file_eligibility.name.lower())]
                    )

                    if file_eligibility == FileState.ELIGIBLE:
                        logger.info("Ingesting %s", file_path)
                        success, error_context = self.document_ingestor.ingest_file(
                            file_path, file_record.db_file_version.file_version_id
                        )
                        if success:
                            db_file = file_record.db_file
                            db_file.last_indexed_at = datetime.now(timezone.utc)
                            self.meta_store.update_file(db_file)
                        else:
                            err_file.write(json.dumps(error_context) + "\n")
                            has_errors = True
                        db_indexing_job.files_processed += 1
                        db_indexing_job = self.meta_store.update_indexing_job(
                            db_indexing_job
                        )
                    else:
                        continue

            db_indexing_job.job_status = (
                "COMPLETED_WITH_ERRORS" if has_errors else "COMPLETED"
            )
            db_indexing_job.job_end_time = datetime.now(timezone.utc)
            db_indexing_job = self.meta_store.update_indexing_job(db_indexing_job)

            cortex.last_indexed_at = datetime.now(timezone.utc)
            self.meta_store.update_cortex(cortex)

            return not has_errors
        except Exception as e:
            db_indexing_job.job_status = "FAILED"
            db_indexing_job.job_end_time = datetime.now(timezone.utc)
            db_indexing_job = self.meta_store.update_indexing_job(db_indexing_job)
            raise e
