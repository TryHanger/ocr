"""API v1 master router."""

from fastapi import APIRouter

from app.api.v1.endpoints.analytics import router as analytics_router
from app.api.v1.endpoints.documents import router as documents_router
from app.api.v1.endpoints.operations import router as operations_router
from app.api.v1.endpoints.improvement import router as improvement_router
from app.api.v1.endpoints.quality import router as quality_router
from app.api.v1.endpoints.research import router as research_router
from app.api.v1.endpoints.review import router as review_router

api_v1_router = APIRouter()
api_v1_router.include_router(documents_router)
api_v1_router.include_router(review_router)
api_v1_router.include_router(quality_router)
api_v1_router.include_router(analytics_router)
api_v1_router.include_router(research_router)
api_v1_router.include_router(operations_router)
api_v1_router.include_router(improvement_router)


