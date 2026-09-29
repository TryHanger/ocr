"""Configuration settings using Pydantic Settings."""

import os
from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment override support."""

    APP_NAME: str = "Document AI Control Center"
    APP_ENV: str = "development"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = True

    API_V1_PREFIX: str = "/api/v1"
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Database
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/document_ai"
    DATABASE_URL_SYNC: str = "postgresql://postgres:postgres@localhost:5432/document_ai"

    # Storage
    STORAGE_BACKEND: str = "local"
    STORAGE_LOCAL_PATH: str = "/data/documents"
    MAX_UPLOAD_SIZE_BYTES: int = 20 * 1024 * 1024  # 20 MB

    # Allowed MIME types
    ALLOWED_MIME_TYPES: List[str] = [
        "image/jpeg",
        "image/png",
        "image/webp",
        "image/tiff",
        "application/pdf",
    ]

    # ML / OCR / KIE Providers
    OCR_PROVIDER: str = "research"
    OCR_EXECUTION_PROVIDER: str = "cpu"
    KIE_PROVIDER: str = "research"
    RESEARCH_ROOT_PATH: str = str(Path(__file__).resolve().parent.parent.parent.parent.parent)

    # Worker
    WORKER_POLL_INTERVAL_SECONDS: float = 1.0
    MAX_JOB_ATTEMPTS: int = 3

    # Automation Policy Thresholds (Receipt)
    POLICY_RECEIPT_COMPANY_THRESHOLD: float = 0.90
    POLICY_RECEIPT_DATE_THRESHOLD: float = 0.90
    POLICY_RECEIPT_ADDRESS_THRESHOLD: float = 0.85
    POLICY_RECEIPT_TOTAL_THRESHOLD: float = 0.95
    POLICY_MIN_IMAGE_QUALITY_SCORE: float = 0.50

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
