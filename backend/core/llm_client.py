# core/llm_client.py

import time
from dotenv import load_dotenv
from groq import Groq

from core.config import settings
from utils.logger import setup_logger

load_dotenv()
logger = setup_logger(__name__)


GEOLOGICAL_SYSTEM_PROMPT = """You are a senior geological AI assistant specializing 
in mineral exploration. Be technically precise, cite sources when available, and 
provide actionable recommendations for exploration teams."""

DECISION_SUPPORT_PROMPT = """You are a mineral exploration strategist. Provide 
structured, data-driven recommendations that exploration teams can act on immediately."""


class LLMClient:

    def __init__(self):
        try:
            self.client = Groq(api_key=settings.GROQ_API_KEY)

            # ✅ Fast + cheap + strong reasoning
            self.MODEL = "llama-3.1-8b-instant"

            logger.info(f"Groq LLM initialized with model: {self.MODEL}")

        except Exception as e:
            logger.error(f"Groq initialization failed: {e}")
            raise

    # ─────────────────────────────────────────────
    # Prompt Builder
    # ─────────────────────────────────────────────
    def _build_rag_prompt(self, query, context_chunks):

        context_chunks = context_chunks[:5]

        context = "\n\n---\n\n".join([
            f"[Source: {c.get('metadata', {}).get('source', 'Unknown')}]\n{c.get('text', '')}"
            for c in context_chunks
        ])

        return f"""
Context:
{context}

---

Question: {query}

Instructions:
- Answer ONLY using the context
- If answer not found, say: "Not found in provided data"
- Be clear and technical
"""

    # ─────────────────────────────────────────────
    # Main Query
    # ─────────────────────────────────────────────
    def query(self, user_query, context_chunks=None, system_prompt=None):

        context_chunks = context_chunks or []

        # 🚫 BLOCK if no retrieval
        if len(context_chunks) == 0:
            logger.warning("No context retrieved")

            return {
                "answer": "No relevant information found in the knowledge base.",
                "usage": {"input_tokens": 0, "output_tokens": 0},
                "model": "none"
            }

        prompt = self._build_rag_prompt(user_query, context_chunks)

        try:
            response = self.client.chat.completions.create(
                model=self.MODEL,
                messages=[
                    {"role": "system", "content": system_prompt or GEOLOGICAL_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=1024,
            )

            answer = response.choices[0].message.content

            return {
                "answer": answer,
                "usage": {
                    "input_tokens": response.usage.prompt_tokens,
                    "output_tokens": response.usage.completion_tokens,
                },
                "model": self.MODEL,
            }

        except Exception as e:
            logger.error(f"Groq API error: {e}")

            return {
                "answer": "LLM service temporarily unavailable.",
                "usage": {"input_tokens": 0, "output_tokens": 0},
                "model": "error"
            }

    # ─────────────────────────────────────────────
    # Streaming (optional)
    # ─────────────────────────────────────────────
    def stream_query(self, user_query, context_chunks=None, system_prompt=None):

        context_chunks = context_chunks or []

        if len(context_chunks) == 0:
            yield "No relevant information found in the knowledge base."
            return

        prompt = self._build_rag_prompt(user_query, context_chunks)

        try:
            stream = self.client.chat.completions.create(
                model=self.MODEL,
                messages=[
                    {"role": "system", "content": system_prompt or GEOLOGICAL_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=1024,
                stream=True,
            )

            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error(f"Streaming failed: {e}")
            yield "\n[Streaming failed]"

    # ─────────────────────────────────────────────
    # Extra APIs (unchanged)
    # ─────────────────────────────────────────────
    def analyze_mineral_zones(self, data_summary):
        return self.query(
            f"Identify potential mineral zones from this data:\n\n{data_summary}"
        )

    def generate_exploration_report(self, findings):
        result = self.query(
            f"Generate a professional exploration report from:\n\n{findings}"
        )
        return result.get("answer", "Failed to generate report")