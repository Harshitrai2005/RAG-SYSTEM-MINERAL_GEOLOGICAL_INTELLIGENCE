"""
Health Check Routes
System status and readiness probes.
"""

from fastapi import APIRouter
from models.schemas import HealthResponse
from core.config import settings

router = APIRouter()


@router.get("/health", response_model=HealthResponse, summary="System health check")
async def health_check():
    return HealthResponse(
        status="healthy",
        message="Mineral Exploration Intelligence System is running.",
        version=settings.VERSION,
    )
