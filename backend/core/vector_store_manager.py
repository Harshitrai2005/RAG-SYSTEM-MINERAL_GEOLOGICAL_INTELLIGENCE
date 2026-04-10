# core/vector_store_manager.py

import os
from typing import Optional

import pyarrow as pa
import lancedb
from sentence_transformers import SentenceTransformer

from core.config import settings
from utils.logger import setup_logger

logger = setup_logger(__name__)

EMBEDDING_DIM = 384


def _make_schema() -> pa.Schema:
    return pa.schema([
        pa.field("id", pa.string()),
        pa.field("text", pa.string()),
        pa.field("source", pa.string()),
        pa.field("doc_type", pa.string()),
        pa.field("page", pa.int32()),
        pa.field("section", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), EMBEDDING_DIM)),
    ])


class VectorStoreManager:

    def __init__(self):
        self.PERSIST_DIR = settings.CHROMA_PERSIST_DIR
        self.DB = None
        self.EMBEDDING_MODEL: Optional[SentenceTransformer] = None
        self._TABLES: dict = {}

    def initialize(self):
        os.makedirs(self.PERSIST_DIR, exist_ok=True)

        logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
        self.EMBEDDING_MODEL = SentenceTransformer(settings.EMBEDDING_MODEL)

        logger.info(f"Connecting to LanceDB at: {self.PERSIST_DIR}")
        self.DB = lancedb.connect(self.PERSIST_DIR)

        schema = _make_schema()

        for name in [
            settings.COLLECTION_GEOLOGICAL,
            settings.COLLECTION_MINERAL,
            settings.COLLECTION_HYPERSPECTRAL,
        ]:
            if name in self.DB.table_names():
                self._TABLES[name] = self.DB.open_table(name)
            else:
                self._TABLES[name] = self.DB.create_table(name, schema=schema)

            logger.info(f"{name} count: {self._TABLES[name].count_rows()}")

    # ─────────────────────────────────────

    def embed_texts(self, texts):
        return self.EMBEDDING_MODEL.encode(texts).tolist()

    # ─────────────────────────────────────

    def add_documents(self, collection_name, documents):
        table = self._TABLES[collection_name]

        texts = [d["text"] for d in documents]
        embeddings = self.embed_texts(texts)

        rows = []
        for doc, emb in zip(documents, embeddings):
            meta = doc.get("metadata", {})

            rows.append({
                "id": doc["id"],
                "text": doc["text"],
                "source": str(meta.get("source", "")),
                "doc_type": str(meta.get("doc_type", "")),
                "page": int(meta.get("page") or 0),
                "section": str(meta.get("section", "")),
                "vector": emb,
            })

        table.add(rows)
        logger.info(f"Added {len(rows)} docs to {collection_name}")

    # ─────────────────────────────────────

    def query(self, collection_name, query_text, top_k=None):
        table = self._TABLES[collection_name]

        if table.count_rows() == 0:
            return []

        # 🔥 normalize
        query_text = query_text.replace("Au", "gold")

        top_k = top_k or settings.TOP_K_RESULTS

        query_embedding = self.embed_texts([query_text])[0]

        results = table.search(query_embedding).limit(top_k).to_pandas()

        matches = []
        for _, row in results.iterrows():
            matches.append({
                "id": row["id"],
                "text": row["text"],
                "metadata": {
                    "source": row["source"],
                    "doc_type": row["doc_type"],
                    "page": row["page"] if row["page"] != 0 else None,
                    "section": row["section"],
                },
                "similarity": float(row.get("_distance", 0.0)),
            })

        return matches

    # ─────────────────────────────────────

    def multi_collection_query(self, query_text, top_k=None):
        all_results = []

        for name in self._TABLES:
            results = self.query(name, query_text, top_k)
            for r in results:
                r["collection"] = name
            all_results.extend(results)

        # 🔥 smaller distance = better
        all_results.sort(key=lambda x: x["similarity"])

        return all_results[:top_k or settings.TOP_K_RESULTS]
    
    def get_collection_stats(self):
       stats = {}

       for name, table in self._TABLES.items():
        try:
            count = table.count_rows()
        except:
            count = 0

        stats[name] = {
            "count": count
        }

       return stats