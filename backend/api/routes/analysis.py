"""
Analysis Routes
Mineral zone identification and exploration decision support endpoints.
These go beyond simple Q&A — they synthesize data into actionable intelligence.
"""

from fastapi import APIRouter, Depends, HTTPException, Request

from core.llm_client import LLMClient
from core.rag_engine import RAGEngine
from core.vector_store_manager import VectorStoreManager
from models.schemas import (
    MineralZoneAnalysisRequest,
    MineralZoneResponse,
    QueryRequest,
    QueryType,
)
from utils.logger import setup_logger

logger = setup_logger(__name__)
router = APIRouter()


def get_rag_engine(request: Request) -> RAGEngine:
    vs: VectorStoreManager = request.app.state.vector_store_manager
    return RAGEngine(vector_store=vs, llm_client=LLMClient())


@router.post(
    "/mineral-zones",
    response_model=MineralZoneResponse,
    summary="Identify potential mineral zones from provided data",
)
async def analyze_mineral_zones(
    body: MineralZoneAnalysisRequest,
    rag: RAGEngine = Depends(get_rag_engine),
):
    """
    Analyze a data summary (text describing geochemical results, drill data,
    or field observations) to identify potential mineral zones.

    Optionally generates a formatted exploration report.
    """
    try:
        llm = LLMClient()

        # First retrieve relevant context from the knowledge base
        context_request = QueryRequest(
            query=body.data_summary,
            query_type=QueryType.ALL,
            top_k=6,
        )
        context_chunks = rag.vector_store.multi_collection_query(
            query_text=body.data_summary, top_k=6
        )

        # Enrich the summary with retrieved context
        enriched_summary = body.data_summary
        if context_chunks:
            context_texts = "\n---\n".join(
                [f"[From: {c['metadata'].get('source', 'unknown')}]\n{c['text'][:500]}"
                 for c in context_chunks[:3]]
            )
            enriched_summary = (
                f"{body.data_summary}\n\n"
                f"ADDITIONAL CONTEXT FROM KNOWLEDGE BASE:\n{context_texts}"
            )

        result = llm.analyze_mineral_zones(enriched_summary)

        report = None
        if body.include_report:
            report = llm.generate_exploration_report(
                {"analysis": result["answer"], "input_data": body.data_summary}
            )

        return MineralZoneResponse(
            analysis=result["answer"],
            report=report,
            model=result["model"],
        )

    except Exception as e:
        logger.error(f"Mineral zone analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/exploration-decision", summary="Get exploration decision support")
async def exploration_decision(
    context: str,
    target_commodity: str = "Au",
    rag: RAGEngine = Depends(get_rag_engine),
):
    """
    Decision support for mineral exploration programs.

    Synthesizes available data into prioritized drill target recommendations,
    risk factors, and suggested next steps.
    """
    query = (
        f"Based on the following exploration context, provide a decision framework "
        f"for targeting {target_commodity} mineralization. "
        f"Include target prioritization, recommended work program, and key risk factors.\n\n"
        f"Context: {context}"
    )

    request = QueryRequest(
        query=query,
        query_type=QueryType.DECISION,
        top_k=8,
    )

    try:
        return rag.query(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/deposit-models", summary="List supported deposit models")
async def list_deposit_models():
    """Return the deposit models the system understands."""
    from ingestion.mineral_dataset_processor import DEPOSIT_PATHFINDERS
    from ingestion.hyperspectral_processor import MINERAL_SPECTRAL_SIGNATURES

    return {
        "deposit_models": [
            {"name": model, "pathfinders": elements}
            for model, elements in DEPOSIT_PATHFINDERS.items()
        ],
        "spectral_minerals": [
            {
                "mineral": mineral,
                "alteration_type": info["alteration_type"],
                "deposit_association": info["deposit_association"],
            }
            for mineral, info in MINERAL_SPECTRAL_SIGNATURES.items()
        ],
    }
