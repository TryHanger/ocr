"""Smart Preprocessing Decision Layer grounded in B1/B2 research findings."""

import time
from typing import Any, Dict, List, Tuple
import cv2
import numpy as np

from app.core.logging import logger
from app.schemas.quality import QualityResultDTO


class PreprocessingPolicy:
    """Evaluates image quality metrics against empirical research findings to plan and execute targeted preprocessing."""

    # Validated & Provisional Thresholds
    ROTATION_THRESHOLD_DEGREES = 1.5  # Validated by B1 D6 (Geometric Rotation)
    CONTRAST_THRESHOLD_RMS = 32.0  # Provisional (Derived from SROIE validation luminance)
    NOISE_THRESHOLD_ESTIMATE = 10.0  # Provisional (Derived from D3 noise benchmarks)

    @classmethod
    def plan_operations(cls, quality: QualityResultDTO) -> List[Dict[str, Any]]:
        """Determine which preprocessing operations are required based on quality measurements."""
        plan: List[Dict[str, Any]] = []

        # 1. Deskew: Research showed rotation degrades reading order clustering and CER severely
        if abs(quality.rotation_angle) >= cls.ROTATION_THRESHOLD_DEGREES:
            plan.append(
                {
                    "operation": "deskew",
                    "parameters": {
                        "angle": quality.rotation_angle,
                        "max_angle": 15.0,
                        "border_mode": "constant_white",
                        "basis": "B1_D6_rotation_recovery",
                    },
                }
            )

        # 2. CLAHE: Research showed CLAHE restores text readability under shadows and low contrast
        if quality.contrast_score < cls.CONTRAST_THRESHOLD_RMS:
            plan.append(
                {
                    "operation": "clahe",
                    "parameters": {
                        "clip_limit": 2.0,
                        "tile_grid_size": [8, 8],
                        "color_space": "CIE_LAB_L_channel",
                        "basis": "B2_contrast_enhancement",
                    },
                }
            )

        # 3. Denoise: Research showed mild median filter reduces high-frequency noise without edge smearing
        if quality.noise_score > cls.NOISE_THRESHOLD_ESTIMATE:
            plan.append(
                {
                    "operation": "denoise_median",
                    "parameters": {
                        "method": "median",
                        "ksize": 3,
                        "basis": "B2_minimal_cleanup",
                    },
                }
            )

        return plan

    @classmethod
    def execute_plan(
        cls,
        image_bytes: bytes,
        quality: QualityResultDTO,
        initial_step_order: int = 2,
    ) -> Tuple[bytes, List[Dict[str, Any]]]:
        """Execute scheduled preprocessing operations, preserving full execution provenance."""
        plan = cls.plan_operations(quality)

        # If clean document, no-op pass-through
        if not plan:
            return image_bytes, []

        nparr = np.frombuffer(image_bytes, np.uint8)
        current_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if current_img is None:
            return image_bytes, []

        executed_steps: List[Dict[str, Any]] = []
        current_order = initial_step_order

        for op_spec in plan:
            op_name = op_spec["operation"]
            params = op_spec["parameters"]
            t0 = time.perf_counter()

            try:
                if op_name == "deskew":
                    angle = float(params["angle"])
                    h, w = current_img.shape[:2]
                    center = (w // 2, h // 2)
                    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
                    current_img = cv2.warpAffine(
                        current_img,
                        rot_mat,
                        (w, h),
                        flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=(255, 255, 255),
                    )

                elif op_name == "clahe":
                    lab = cv2.cvtColor(current_img, cv2.COLOR_BGR2LAB)
                    l_channel, a_channel, b_channel = cv2.split(lab)
                    clahe_obj = cv2.createCLAHE(
                        clipLimit=float(params["clip_limit"]),
                        tileGridSize=tuple(params["tile_grid_size"]),
                    )
                    cl = clahe_obj.apply(l_channel)
                    merged = cv2.merge((cl, a_channel, b_channel))
                    current_img = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

                elif op_name == "denoise_median":
                    ksize = int(params["ksize"])
                    current_img = cv2.medianBlur(current_img, ksize)

                duration_ms = round((time.perf_counter() - t0) * 1000.0, 2)
                executed_steps.append(
                    {
                        "step_order": current_order,
                        "operation": op_name,
                        "parameters": params,
                        "duration_ms": duration_ms,
                        "status": "APPLIED",
                    }
                )
                current_order += 1

            except Exception as e:
                duration_ms = round((time.perf_counter() - t0) * 1000.0, 2)
                logger.error(f"Failed to apply preprocessing operation '{op_name}': {e}")
                executed_steps.append(
                    {
                        "step_order": current_order,
                        "operation": op_name,
                        "parameters": params,
                        "duration_ms": duration_ms,
                        "status": "FAILED",
                        "error": str(e),
                    }
                )
                current_order += 1

        _, out_buf = cv2.imencode(".jpg", current_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return out_buf.tobytes(), executed_steps
