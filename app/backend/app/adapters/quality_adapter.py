"""Document Quality Analyzer Adapter."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import math
from typing import Any, Dict, Tuple
import cv2
import numpy as np

from app.adapters.base import QualityAnalyzer
from app.schemas.quality import QualityResultDTO

_quality_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="quality_analyzer")


class QualityAnalysisAdapter(QualityAnalyzer):
    """Computes blur, contrast, noise, and deskew metrics on document images."""

    def _sync_analyze(self, image_bytes: bytes, document_id: str) -> QualityResultDTO:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to decode image for quality analysis: {document_id}")

        height, width = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 1. Blur score: Laplacian variance (higher = sharper, lower = blurrier)
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_32F).var())

        # 2. Contrast score: Standard deviation of pixel intensities
        contrast_rms = float(np.std(gray))

        # 3. Noise score: Difference between image and median filtered image
        denoised = cv2.medianBlur(gray, 3)
        noise_est = float(np.mean(cv2.absdiff(gray, denoised)))

        # 4. Dominant skew angle estimation (Hough Transform)
        rotation_angle = 0.0
        try:
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            lines = cv2.HoughLinesP(
                edges, 1, np.pi / 180, threshold=100, minLineLength=width * 0.1, maxLineGap=10
            )
            if lines is not None and len(lines) > 0:
                angles = []
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    dx = x2 - x1
                    dy = y2 - y1
                    if dx != 0:
                        angle = math.degrees(math.atan2(dy, dx))
                        if -45.0 <= angle <= 45.0:
                            angles.append(angle)
                if angles:
                    rotation_angle = float(np.median(angles))
        except Exception:
            rotation_angle = 0.0

        # Estimated resolution indicator
        resolution_dpi = float(min(width, height) / 3.0)  # Approximation

        # Aggregate normalized quality score (0.0 to 1.0)
        # Blur factor: laplacian > 150 is very sharp (1.0), < 30 is blurry (0.0)
        blur_factor = max(0.0, min(1.0, (laplacian_var - 20.0) / 180.0))
        # Contrast factor: std > 40 is good (1.0), < 15 is washed out (0.0)
        contrast_factor = max(0.0, min(1.0, (contrast_rms - 15.0) / 45.0))
        # Skew factor: |angle| < 1.0 is 1.0, > 10.0 is 0.0
        skew_factor = max(0.0, min(1.0, 1.0 - abs(rotation_angle) / 10.0))
        # Noise factor: noise < 5.0 is clean (1.0), > 25.0 is noisy (0.0)
        noise_factor = max(0.0, min(1.0, 1.0 - (noise_est - 5.0) / 20.0))

        quality_score = round(
            0.4 * blur_factor + 0.3 * contrast_factor + 0.2 * skew_factor + 0.1 * noise_factor, 2
        )
        quality_score = max(0.0, min(1.0, quality_score))

        # Profile summary classification
        if quality_score >= 0.75:
            profile_summary = "High Quality"
        elif quality_score >= 0.50:
            profile_summary = "Acceptable"
        else:
            profile_summary = "Degraded / Review Recommended"

        recommendations = {
            "needs_deskew": abs(rotation_angle) >= 2.0,
            "needs_clahe": contrast_rms < 30.0,
            "needs_denoise": noise_est > 12.0,
        }

        return QualityResultDTO(
            document_id=document_id,
            blur_score=round(laplacian_var, 2),
            contrast_score=round(contrast_rms, 2),
            noise_score=round(noise_est, 2),
            rotation_angle=round(rotation_angle, 2),
            resolution_dpi=round(resolution_dpi, 1),
            quality_score=quality_score,
            profile_summary=profile_summary,
            metrics={
                "blur_factor": round(blur_factor, 2),
                "contrast_factor": round(contrast_factor, 2),
                "skew_factor": round(skew_factor, 2),
                "noise_factor": round(noise_factor, 2),
                "width": width,
                "height": height,
            },
            recommendations=recommendations,
        )

    async def analyze(self, image_bytes: bytes, document_id: str) -> QualityResultDTO:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            _quality_executor, self._sync_analyze, image_bytes, document_id
        )


default_quality_adapter = QualityAnalysisAdapter()
