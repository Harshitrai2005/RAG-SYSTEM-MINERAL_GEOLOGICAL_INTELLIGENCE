"""
Query Routes
Handles all RAG query requests — standard and streaming.
"""

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from core.llm_client import LLMClient
from core.rag_engine import RAGEngine
from core.vector_store_manager import VectorStoreManager
from models.schemas import QueryRequest, QueryResponse
from utils.logger import setup_logger

logger = setup_logger(__name__)
router = APIRouter()


def get_rag_engine(request: Request) -> RAGEngine:
    """Dependency injection — resolves RAGEngine from app state."""
    vs: VectorStoreManager = request.app.state.vector_store_manager
    llm = LLMClient()
    return RAGEngine(vector_store=vs, llm_client=llm)


@router.post("/", response_model=QueryResponse, summary="Query the geological knowledge base")
async def query(
    body: QueryRequest,
    rag: RAGEngine = Depends(get_rag_engine),
):
    """
    Run a RAG query against the geological knowledge base.

    The system retrieves relevant context from uploaded documents,
    then asks Claude to answer your question using that context.

    **Query types:**
    - `all` — search across all document collections (default)
    - `geological` — geological survey reports only
    - `mineral` — geochemical datasets only
    - `hyperspectral` — satellite/hyperspectral imagery only
    - `decision` — exploration decision-support mode
    """
    if body.stream:
        # Return a streaming response
        def generate():
            for token in rag.stream_query(body):
                yield token

        return StreamingResponse(generate(), media_type="text/plain")

    try:
        response = rag.query(body)
        return response
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rock-formation", summary="Analyze rock formation from text description")
async def query_rock_formation(
    description: str,
    rag: RAGEngine = Depends(get_rag_engine),
):
    """
    Specialized endpoint for rock formation queries.
    Automatically searches geological reports and returns formation context.
    """
    request = QueryRequest(
        query=f"Describe and interpret the following rock formation: {description}",
        query_type="geological",
        top_k=5,
    )
    try:
        return rag.query(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/mineral-zone", summary="Identify potential mineral zones")
async def query_mineral_zones(
    query_text: str,
    rag: RAGEngine = Depends(get_rag_engine),
):
    """
    Query focused on identifying and characterizing mineral zones
    based on geochemical and geological data in the knowledge base.
    """
    request = QueryRequest(
        query=f"Identify potential mineral zones related to: {query_text}",
        query_type="all",
        top_k=8,
    )
    try:
        return rag.query(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
