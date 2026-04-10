"""
Mineral Exploration Intelligence System
Main FastAPI Application Entry Point
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from api.routes import query, ingest, analysis, health
from core.config import settings
from core.vector_store_manager import VectorStoreManager
from utils.logger import setup_logger

logger = setup_logger(__name__)

# Build absolute paths so they work regardless of where uvicorn is launched from
BASE_DIR = Path(__file__).resolve().parent.parent   # project root
FRONTEND_DIR = BASE_DIR / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Mineral Exploration Intelligence System...")
    vs_manager = VectorStoreManager()
    vs_manager.initialize()
    app.state.vector_store_manager = vs_manager
    logger.info("System ready. Vector store initialized.")
    yield
    logger.info("Shutting down system...")


app = FastAPI(
    title="Mineral Exploration Intelligence System",
    description="RAG-powered system for mining engineers.",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/api", tags=["Health"])
app.include_router(ingest.router, prefix="/api/ingest", tags=["Data Ingestion"])
app.include_router(query.router, prefix="/api/query", tags=["Query & RAG"])
app.include_router(analysis.router, prefix="/api/analysis", tags=["Mineral Analysis"])

# Use absolute path so Windows always finds the frontend folder
static_dir = FRONTEND_DIR / "static"
static_dir.mkdir(parents=True, exist_ok=True)   # create if missing
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/", include_in_schema=False)
async def serve_frontend():
    return FileResponse(str(FRONTEND_DIR / "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info",
    )