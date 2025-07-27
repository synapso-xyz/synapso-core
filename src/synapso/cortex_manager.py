import csv
import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Iterator

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .config_manager import GlobalConfig, get_config
from .ingestor.document_ingestor import ingest_file
from .persistence.factory import MetaStoreFactory
from .persistence.interfaces.meta_store import MetaStore
from .persistence.models import Cortex

SUPPORTED_FORMATS = [".md", ".markdown"]


def get_async_engine():
    config: GlobalConfig = get_config()
    meta_store: MetaStore = MetaStoreFactory.get_meta_store(
        config.meta_store.meta_db_type
    )
    async_engine = meta_store.get_async_engine()
    return async_engine


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


async def create_cortex(cortex_name: str | None, folder_path: str) -> Cortex:
    _validate_cortex_path(folder_path)

    cortex_id = uuid.uuid4().hex
    cortex = Cortex(
        cortex_id=cortex_id,
        cortex_name=cortex_name,
        path=folder_path,
        created_at=datetime.now(),
        updated_at=datetime.now(),
        last_indexed_at=None,  # We have not indexed the cortex yet.
    )

    async with AsyncSession(get_async_engine()) as session:
        session.add(cortex)
        await session.commit()

    return cortex


async def get_cortex_by_id(cortex_id: str) -> Cortex:
    stmt = select(Cortex).where(Cortex.cortex_id == cortex_id)
    async with AsyncSession(get_async_engine()) as session:
        result = await session.execute(stmt)
        row = result.first()
        if row is None:
            raise ValueError(f"Cortex with id {cortex_id} not found")
        return row[0]


async def initialize_cortex(cortex_id: str, index_now: bool = True) -> bool:
    # For now, let is just trigger the indexing.
    cortex = await get_cortex_by_id(cortex_id)
    cortex_path = cortex.path
    synapso_dir_path = Path(cortex_path) / ".synapso"
    synapso_dir_path.mkdir(exist_ok=True)

    # Initialize the file_list.csv file
    file_list_path = _get_file_list_path(cortex_path)
    file_list_path.touch(exist_ok=True)

    if index_now:
        indexing_result = await index_cortex(cortex_id=cortex_id)

    return indexing_result


async def index_cortex(cortex_id: str) -> bool:
    cortex = await get_cortex_by_id(cortex_id)
    if cortex is None:
        raise ValueError("Invalid cortex id")
    cortex_path = cortex.path

    file_list_path = _get_file_list_path(directory_path=cortex_path)
    ingestion_errors_path = _get_ingestion_errors_path(directory_path=cortex_path)

    ingestion_status = True
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
                ingestion_status, error_context = await ingest_file(file_path)
                if ingestion_status is False:
                    err_file.write(json.dumps(error_context) + "\n")
                    ingestion_status = False

        return True


async def delete_cortex(cortex_id: str) -> bool: ...


async def purge_cortex(cortex_id: str) -> bool: ...
