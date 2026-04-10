"""
Ingestion Routes
File upload endpoints for geological reports, mineral datasets, and imagery.
Each file type routes to the appropriate processor and collection.
"""

import os
import shutil
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile

from core.config import settings
from core.vector_store_manager import VectorStoreManager
from ingestion.hyperspectral_processor import HyperspectralProcessor
from ingestion.mineral_dataset_processor import MineralDatasetProcessor
from ingestion.pdf_processor import PDFProcessor
from models.schemas import FileCategory, IngestResponse, SystemStatsResponse, CollectionStats
from utils.logger import setup_logger

logger = setup_logger(__name__)
router = APIRouter()

# Instantiate processors (stateless — safe to reuse)
pdf_processor = PDFProcessor()
mineral_processor = MineralDatasetProcessor()
hyperspectral_processor = HyperspectralProcessor()


def get_vector_store(request: Request) -> VectorStoreManager:
    return request.app.state.vector_store_manager


def _collection_for_category(category: FileCategory) -> str:
    mapping = {
        FileCategory.REPORT: settings.COLLECTION_GEOLOGICAL,
        FileCategory.DATASET: settings.COLLECTION_MINERAL,
        FileCategory.IMAGERY: settings.COLLECTION_HYPERSPECTRAL,
    }
    return mapping[category]


def _validate_extension(filename: str, category: FileCategory):
    suffix = Path(filename).suffix.lower()
    allowed = {
        FileCategory.REPORT: {".pdf", ".txt"},
        FileCategory.DATASET: {".csv", ".json"},
        FileCategory.IMAGERY: {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".hdr"},
    }
    if suffix not in allowed[category]:
        raise HTTPException(
            status_code=400,
            detail=f"File type '{suffix}' not supported for category '{category}'. "
                   f"Allowed: {', '.join(allowed[category])}",
        )


@router.post("/upload", response_model=IngestResponse, summary="Upload a document for ingestion")
async def upload_file(
    file: UploadFile = File(...),
    category: FileCategory = Form(...),
    vector_store: VectorStoreManager = Depends(get_vector_store),
):
    """
    Upload a file and ingest it into the appropriate vector store collection.

    **Categories:**
    - `report` — PDF or text geological survey reports
    - `dataset` — CSV or JSON geochemical/mineral datasets
    - `imagery` — GeoTIFF, PNG, JPEG, or ENVI hyperspectral files

    The file is processed, chunked, embedded, and stored for semantic retrieval.
    """
    _validate_extension(file.filename, category)

    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    upload_dir = Path(settings.UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)

    # Save upload to disk
    dest_path = upload_dir / file.filename
    try:
        content = await file.read()
        if len(content) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"File exceeds max size of {settings.MAX_UPLOAD_SIZE_MB} MB",
            )
        with open(dest_path, "wb") as f:
            f.write(content)

        logger.info(f"Saved upload: {dest_path} ({len(content)/1024:.1f} KB)")

        # Process file → chunks
        chunks = _process_file(dest_path, category)

        if not chunks:
            raise HTTPException(
                status_code=422,
                detail="No content could be extracted from the uploaded file.",
            )

        # Ingest into vector store
        collection_name = _collection_for_category(category)
        count = vector_store.add_documents(collection_name, chunks)

        return IngestResponse(
            success=True,
            file_name=file.filename,
            collection=collection_name,
            chunks_added=count,
            message=f"Successfully ingested {count} chunks from '{file.filename}'",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ingestion failed for {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion error: {str(e)}")


def _process_file(file_path: Path, category: FileCategory) -> list[dict]:
    """Route file to the correct processor based on category and extension."""
    suffix = file_path.suffix.lower()

    if category == FileCategory.REPORT:
        if suffix == ".pdf":
            return pdf_processor.process_file(file_path)
        elif suffix == ".txt":
            return _process_text_file(file_path)

    elif category == FileCategory.DATASET:
        return mineral_processor.process_file(file_path)

    elif category == FileCategory.IMAGERY:
        return hyperspectral_processor.process_file(file_path)

    return []


def _process_text_file(file_path: Path) -> list[dict]:
    """Process plain text geological reports."""
    from utils.text_chunker import TextChunker
    import hashlib

    with open(file_path, encoding="utf-8", errors="replace") as f:
        text = f.read()

    chunker = TextChunker()
    chunks_text = chunker.split(text)
    file_hash = hashlib.md5(text.encode()).hexdigest()[:12]

    return [
        {
            "id": f"{file_hash}_c{i}",
            "text": chunk,
            "metadata": {
                "source": file_path.name,
                "doc_type": "geological_report",
                "file_type": "txt",
                "chunk_index": i,
            },
        }
        for i, chunk in enumerate(chunks_text)
        if len(chunk.strip()) >= 50
    ]


@router.get("/stats", response_model=SystemStatsResponse, summary="Get knowledge base statistics")
async def get_stats(vector_store: VectorStoreManager = Depends(get_vector_store)):
    """Return document counts for all collections."""
    raw_stats = vector_store.get_collection_stats()
    collections = [
        CollectionStats(collection_name=name, document_count=info["count"])
        for name, info in raw_stats.items()
    ]
    total = sum(c.document_count for c in collections)
    return SystemStatsResponse(
        status="healthy",
        collections=collections,
        total_documents=total,
        version=settings.VERSION,
    )


@router.delete("/collection/{collection_name}", summary="Clear a collection")
async def clear_collection(
    collection_name: str,
    vector_store: VectorStoreManager = Depends(get_vector_store),
):
    """
    WARNING: Deletes all documents from the specified collection.
    Use for resetting data during development.
    """
    valid_collections = [
        settings.COLLECTION_GEOLOGICAL,
        settings.COLLECTION_MINERAL,
        settings.COLLECTION_HYPERSPECTRAL,
    ]
    if collection_name not in valid_collections:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid collection. Must be one of: {valid_collections}",
        )

    # ChromaDB: delete and recreate the collection
    collection = vector_store.get_collection(collection_name)
    all_ids = collection.get()["ids"]
    if all_ids:
        collection.delete(ids=all_ids)

    return {"message": f"Cleared {len(all_ids)} documents from '{collection_name}'"}
