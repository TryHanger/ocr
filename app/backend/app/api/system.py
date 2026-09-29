"""System health and version endpoints."""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.session import get_db
from app.services.storage.local import default_storage

router = APIRouter(tags=["System"])


@router.get("/health", status_code=status.HTTP_200_OK)
async def health_check() -> dict:
    """Liveness probe."""
    return {
        "status": "healthy",
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "env": settings.APP_ENV,
    }


@router.get("/ready", status_code=status.HTTP_200_OK)
async def readiness_check(db: AsyncSession = Depends(get_db)) -> dict:
    """Readiness probe checking DB connectivity and storage availability."""
    db_ok = False
    try:
        res = await db.execute(text("SELECT 1"))
        db_ok = res.scalar() == 1
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database connectivity check failed: {e}",
        )

    storage_ok = False
    try:
        storage_ok = default_storage.base_dir.exists()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Storage driver check failed: {e}",
        )

    return {
        "status": "ready",
        "database": "connected" if db_ok else "unreachable",
        "storage": "accessible" if storage_ok else "unreachable",
    }


@router.get("/api/version", status_code=status.HTTP_200_OK)
async def get_version() -> dict:
    """Version metadata."""
    return {
        "version": settings.APP_VERSION,
        "app_name": settings.APP_NAME,
        "environment": settings.APP_ENV,
    }
