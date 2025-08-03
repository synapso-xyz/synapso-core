import csv
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Iterator, List

from .config_manager import GlobalConfig, get_config
from .data_store.data_models import DBCortex
from .data_store.factory import DataStoreFactory
from .data_store.interfaces import MetaStore
from .ingestor.document_ingestor import ingest_file
from .synapso_logger import get_logger

SUPPORTED_FORMATS = [".md", ".markdown"]

logger = get_logger(__name__)


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


@dataclass(frozen=True)
class FileRecord:
    path: Path
    state: FileState


def _get_file_list_path(directory_path: str, ensure_present=True) -> Path:
    file_list_path = Path(directory_path) / ".synapso" / "file_list.csv"
    if ensure_present:
        file_list_path.touch()
    return file_list_path


def _get_ingestion_errors_path(directory_path: str, ensure_present=True) -> Path:
    ingestion_errors_path = Path(directory_path) / ".synapso" / "ingestion_errors.jsonl"
    if ensure_present:
        ingestion_errors_path.touch()
    return ingestion_errors_path


def _file_walk(directory_path: str) -> Iterator[FileRecord]:
    root = Path(directory_path).expanduser().resolve()
    for dirpath, _, filenames in os.walk(root):
        base = Path(dirpath)
        for fname in filenames:
            fpath = base / fname
            yield FileRecord(fpath, state=_classify(fpath, root))


def _validate_cortex_path(cortex_path: str) -> None:
    if not os.path.isdir(cortex_path):
        raise ValueError(f"Path '{cortex_path}' is not a directory.")
    # Check if the directory is visible (does not start with a dot)
    base_name = os.path.basename(os.path.normpath(cortex_path))
    if base_name.startswith("."):
        raise ValueError(f"Directory '{cortex_path}' is hidden (starts with a dot).")


class CortexManager:
    def __init__(self) -> None:
        self.config: GlobalConfig = get_config()
        self.meta_store: MetaStore = DataStoreFactory.get_meta_store(
            self.config.meta_store.meta_db_type
        )
        self.sync_engine = self.meta_store.get_sync_engine()

    def create_cortex(self, cortex_name: str | None, folder_path: str) -> str:
        _validate_cortex_path(folder_path)
        return self.meta_store.create_cortex(cortex_name, folder_path)

    def get_cortex_by_id(self, cortex_id: str) -> DBCortex:
        return self.meta_store.get_cortex_by_id(cortex_id)

    def get_cortex_by_name(self, cortex_name: str) -> DBCortex:
        return self.meta_store.get_cortex_by_name(cortex_name)

    def list_cortices(self) -> List[DBCortex]:
        return self.meta_store.list_cortices()

    def index_cortex(
        self, cortex_id: str | None = None, cortex_name: str | None = None
    ) -> bool:
        if cortex_id is None and cortex_name is None:
            raise ValueError("Either cortex_id or cortex_name must be provided")
        if cortex_id:
            cortex = self.get_cortex_by_id(cortex_id)
        else:
            cortex = self.get_cortex_by_name(cortex_name)

        if cortex is None:
            raise ValueError(
                f"Cortex with id {cortex_id} or name {cortex_name} not found"
            )

        # For now, let is just trigger the indexing.
        cortex_path = cortex.path
        synapso_dir_path = Path(cortex_path) / ".synapso"
        synapso_dir_path.mkdir(exist_ok=True, parents=True)

        # Initialize the file_list.csv file
        _get_file_list_path(cortex_path)

        file_list_path = _get_file_list_path(directory_path=cortex_path)
        ingestion_errors_path = _get_ingestion_errors_path(directory_path=cortex_path)

        has_errors = False
        with (
            file_list_path.open("w", newline="", encoding="utf-8") as f,
            ingestion_errors_path.open("w", newline="", encoding="utf-8") as err_file,
        ):
            writer = csv.writer(f)
            writer.writerow(["path", "eligibility"])
            for file_record in _file_walk(cortex_path):
                file_path = file_record.path
                file_eligibility = file_record.state
                writer.writerow([str(file_path), str(file_eligibility.name.lower())])

                if file_eligibility == FileState.ELIGIBLE:
                    logger.info("Ingesting %s", file_path)
                    success, error_context = ingest_file(file_path)
                    if not success:
                        err_file.write(json.dumps(error_context) + "\n")
                        has_errors = True
                else:
                    # logger.info(
                    #     f"Skipping {file_path} because it is not eligible ({file_eligibility.name.lower()})"
                    # )
                    continue

        cortex.last_indexed_at = datetime.now(timezone.utc)
        self.meta_store.update_cortex(cortex)

        return not has_errors

    async def delete_cortex(self, cortex_id: str) -> bool:
        raise NotImplementedError

    async def purge_cortex(self, cortex_id: str) -> bool:
        raise NotImplementedError
