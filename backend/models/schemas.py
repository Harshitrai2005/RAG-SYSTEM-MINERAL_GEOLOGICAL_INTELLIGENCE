"""
Pydantic Schemas
Request and response models with validation for all API endpoints.
"""

from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field


# ── Enums ─────────────────────────────────────────────────────────────────────

class QueryType(str, Enum):
    ALL = "all"
    GEOLOGICAL = "geological"
    MINERAL = "mineral"
    HYPERSPECTRAL = "hyperspectral"
    DECISION = "decision"


class FileCategory(str, Enum):
    REPORT = "report"          # PDF/text geological reports
    DATASET = "dataset"        # CSV/JSON geochemical data
    IMAGERY = "imagery"        # Satellite/hyperspectral images


# ── Request Schemas ───────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="Natural language geological query",
        examples=["What minerals are associated with the northern fault zone?"],
    )
    query_type: QueryType = Field(
        default=QueryType.ALL,
        description="Restrict search to a specific data collection",
    )
    top_k: Optional[int] = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of context chunks to retrieve",
    )
    stream: bool = Field(
        default=False,
        description="Enable streaming response",
    )


class IngestRequest(BaseModel):
    file_category: FileCategory = Field(
        ...,
        description="Category determines which processor and collection to use",
    )
    metadata_override: Optional[dict] = Field(
        default=None,
        description="Optional metadata to merge with auto-detected metadata",
    )


class MineralZoneAnalysisRequest(BaseModel):
    data_summary: str = Field(
        ...,
        min_length=10,
        description="Text summary of geological data for zone analysis",
    )
    include_report: bool = Field(
        default=False,
        description="Generate a full formatted exploration report",
    )


# ── Response Schemas ──────────────────────────────────────────────────────────

class SourceReference(BaseModel):
    source: str
    doc_type: str
    page: Optional[int] = None
    similarity: float
    collection: str
    snippet: str


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: list[SourceReference]
    chunks_retrieved: int
    model: str
    usage: dict[str, int]


class IngestResponse(BaseModel):
    success: bool
    file_name: str
    collection: str
    chunks_added: int
    message: str


class MineralZoneResponse(BaseModel):
    analysis: str
    report: Optional[str] = None
    model: str


class CollectionStats(BaseModel):
    collection_name: str
    document_count: int


class SystemStatsResponse(BaseModel):
    status: str
    collections: list[CollectionStats]
    total_documents: int
    version: str


class HealthResponse(BaseModel):
    status: str
    message: str
    version: str
