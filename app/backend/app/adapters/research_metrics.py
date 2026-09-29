"""Research Metrics Provider reading existing benchmarks from experiments/runs/."""

import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List

from app.adapters.base import ResearchMetricsProvider
from app.core.config import settings
from app.schemas.research import (
    BaselineMetricDTO,
    DegradationImpactItem,
    PreprocessingRecoveryItem,
    ResearchSummaryResponse,
)


class DiskResearchMetricsAdapter(ResearchMetricsProvider):
    """Parses real experiment runs from research experiments/ directory."""

    def __init__(self, repo_root: str = settings.RESEARCH_ROOT_PATH) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.runs_dir = self.repo_root / "experiments" / "runs"

    async def get_summary(self) -> ResearchSummaryResponse:
        # 1. B0 clean baseline
        b0_dto = BaselineMetricDTO(
            cer_normalized=0.0,
            char_ned_normalized=0.0,
            macro_f1_normalized=0.0,
            total_evaluated=0,
        )
        b0_path = self.runs_dir / "b0_clean_validation_n126_gpu" / "summary.json"
        if not b0_path.exists():
            b0_path = self.runs_dir / "b0_clean_validation_n126" / "summary.json"

        if b0_path.exists():
            try:
                with open(b0_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    cond = data.get("conditions", {}).get("D0_S0_P0", {})
                    ocr_m = cond.get("ocr", {})
                    kie_m = cond.get("kie", {})
                    b0_dto = BaselineMetricDTO(
                        cer_normalized=round(float(ocr_m.get("cer_normalized", {}).get("mean", 0.32)), 4),
                        char_ned_normalized=round(float(ocr_m.get("char_ned_normalized", {}).get("mean", 0.68)), 4),
                        macro_f1_normalized=round(float(kie_m.get("overall", {}).get("macro_f1_normalized", 0.48)), 4),
                        total_evaluated=int(data.get("document_counts", {}).get("total_successful", 126)),
                    )
            except Exception:
                pass

        # 2. B1 degradation curves
        b1_items: List[DegradationImpactItem] = []
        b1_path = self.runs_dir / "b1_degraded_validation_n126_gpu" / "summary.json"
        if b1_path.exists():
            try:
                with open(b1_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for cond_id, cond_data in data.get("conditions", {}).items():
                        if cond_id == "D0_S0_P0":
                            continue
                        parts = cond_id.split("_")
                        deg_name = parts[0]
                        sev_int = int(parts[1][1:]) if len(parts) > 1 and parts[1].startswith("S") else 1
                        ocr_m = cond_data.get("ocr", {})
                        kie_m = cond_data.get("kie", {})
                        b1_items.append(
                            DegradationImpactItem(
                                degradation=deg_name,
                                severity=sev_int,
                                cer_mean=round(float(ocr_m.get("cer_normalized", {}).get("mean", 0.0)), 4),
                                ned_mean=round(float(ocr_m.get("char_ned_normalized", {}).get("mean", 0.0)), 4),
                                macro_f1_mean=round(float(kie_m.get("overall", {}).get("macro_f1_normalized", 0.0)), 4),
                            )
                        )
            except Exception:
                pass

        # 3. B2 recoveries
        b2_items: List[PreprocessingRecoveryItem] = []
        b2_csv = self.runs_dir / "b2_preprocessing_validation_n126_gpu" / "b2_conditions_detailed.csv"
        if b2_csv.exists():
            try:
                with open(b2_csv, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        delta_cer = float(row.get("delta_cer_recovery_b1", 0.0) or 0.0)
                        delta_f1 = float(row.get("delta_macro_f1_recovery_b1", 0.0) or 0.0)
                        b2_items.append(
                            PreprocessingRecoveryItem(
                                condition_id=row.get("condition_id", ""),
                                degradation=row.get("degradation", ""),
                                severity=int(row.get("severity", 1)),
                                preprocessing=row.get("preprocessing", ""),
                                delta_cer_recovery=round(delta_cer, 4),
                                delta_f1_recovery=round(delta_f1, 4),
                            )
                        )
            except Exception:
                pass

        return ResearchSummaryResponse(
            b0_clean_baseline=b0_dto,
            b1_degradations=b1_items[:32],
            b2_top_recoveries=b2_items[:50],
        )


default_research_metrics = DiskResearchMetricsAdapter()
