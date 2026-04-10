from core.config import settings
from utils.logger import setup_logger
from models.schemas import SourceReference

logger = setup_logger(__name__)


class RAGEngine:

    def __init__(self, vector_store, llm_client):
        self.VECTOR_STORE = vector_store
        self.LLM = llm_client

    def _format_sources(self, chunks: list[dict]) -> list[SourceReference]:
        sources = []
        seen = set()

        for chunk in chunks:
            meta = chunk.get("metadata", {})

            # remove duplicates
            key = (meta.get("source"), meta.get("page"))
            if key in seen:
                continue
            seen.add(key)

            sources.append(
                SourceReference(
                    source=meta.get("source", "Unknown"),
                    doc_type=meta.get("doc_type", "Unknown"),
                    page=meta.get("page"),
                    similarity=chunk.get("similarity", 0.0),
                    collection=chunk.get("collection", ""),
                    snippet=(
                        chunk.get("text", "")[:300] + "..."
                        if len(chunk.get("text", "")) > 300
                        else chunk.get("text", "")
                    ),
                )
            )

        return sources  # ✅ CORRECT POSITION

    def query(self, request):

        logger.info(f"Query: {request.query}")

        # normalize
        request.query = request.query.replace("Au", "gold")

        # retrieve context
        context_chunks = self.VECTOR_STORE.multi_collection_query(
            query_text=request.query,
            top_k=request.top_k or settings.TOP_K_RESULTS
        )

        logger.info(f"Retrieved {len(context_chunks)} chunks")

        # LLM call
        llm_result = self.LLM.query(
            user_query=request.query,
            context_chunks=context_chunks,
            system_prompt="You are a geological AI assistant. Use context if available.",
        )

        # format sources properly
        sources = self._format_sources(context_chunks)

        # SAFE response (no crash even if LLM fails)
        return {
            "query": request.query,
            "answer": llm_result.get("answer", "No response generated."),
            "sources": sources,
            "chunks_retrieved": len(context_chunks),
            "model": llm_result.get("model", "fallback"),
            "usage": llm_result.get("usage", {}),
        }