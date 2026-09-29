"""Operations ↔ Research Evidence Bridge.

Connects production document image quality analysis to frozen B1 (Degradation Robustness)
and B2 (Smart Preprocessing) research evidence.

Guiding Principles:
1. Strict Semantic Boundaries:
   - NO severity inference (e.g. 4.2° rotation is NOT mapped to S1..S4).
   - NO individual-document predictions (e.g. Expected CER is never claimed).
   - NO preprocessing recommendations (e.g. rotation does NOT prescribe Standard Receipt Enhancement).
2. Existing Production Detection Semantics:
   - Strictly uses detection thresholds already established in app/adapters/quality_adapter.py.
   - Does NOT invent new classification thresholds.
3. Decoupled B2 Degradation Coverage:
   - B2 links to the complete degradation dimension (e.g. D6), where 4 compound policies
     are evaluated side-by-side, rather than prescribing any single policy.
4. Non-Fabrication:
   - Signals without verified research counterparts (e.g. contrast) receive no research reference.
"""

from typing import Any, Dict, List, Optional
from app.schemas.quality import QualityResultDTO
from app.schemas.research import (
    ResearchDegradationRef,
    ResearchEvidenceReference,
    TrackEvidenceRef,
)


class ResearchEvidenceBridge:
    """Service providing verified correspondence between production quality signals and research evidence."""

    # Closed, verified dictionary of production signals that have genuine research counterparts.
    # Note: 'contrast' is intentionally absent as low contrast has no standalone B1 degradation dimension.
    # Dimensions such as D2 (Motion Blur), D4 (JPEG), D7 (Perspective), D8 (Shadow) are evaluated in research
    # but have no corresponding detector in the production Quality Analyzer.
    SIGNAL_REGISTRY: Dict[str, Dict[str, Any]] = {
        "rotation": {
            "research_degradation_code": "D6",
            "research_degradation_name": "Rotation & Skew Robustness",
            "b1_experiment_id": "b1_rotation",
            "b1_available": True,
            "b2_degradation_code": "D6",
            "b2_available": True,
            "b2_evaluated_policies_count": 4,
            "unit": "degrees",
            "b1_description": "Controlled S1–S4 synthetic rotation curves (4 conditions evaluated).",
            "b2_description": "4 compound preprocessing policies evaluated across S1–S4 under D6.",
        },
        "noise": {
            "research_degradation_code": "D3",
            "research_degradation_name": "Gaussian Noise Robustness",
            "b1_experiment_id": "b1_gaussian_noise",
            "b1_available": True,
            "b2_degradation_code": "D3",
            "b2_available": True,
            "b2_evaluated_policies_count": 4,
            "unit": "intensity delta",
            "b1_description": "Controlled S1–S4 additive Gaussian noise curves (4 conditions evaluated).",
            "b2_description": "4 compound preprocessing policies evaluated across S1–S4 under D3.",
        },
        "blur": {
            "research_degradation_code": "D1",
            "research_degradation_name": "Gaussian Blur Robustness",
            "b1_experiment_id": "b1_gaussian_blur",
            "b1_available": True,
            "b2_degradation_code": "D1",
            "b2_available": True,
            "b2_evaluated_policies_count": 4,
            "unit": "laplacian variance",
            "b1_description": "Controlled S1–S4 isotropic Gaussian blur curves (4 conditions evaluated).",
            "b2_description": "4 compound preprocessing policies evaluated across S1–S4 under D1.",
        },
        "resolution": {
            "research_degradation_code": "D5",
            "research_degradation_name": "Downsampling / Resolution Loss",
            "b1_experiment_id": "b1_downsampling",
            "b1_available": True,
            "b2_degradation_code": "D5",
            "b2_available": True,
            "b2_evaluated_policies_count": 4,
            "unit": "DPI",
            "b1_description": "Controlled S1–S4 downsampling and resolution decimation (4 conditions evaluated).",
            "b2_description": "4 compound preprocessing policies evaluated across S1–S4 under D5.",
        },
    }

    def get_mapping(self, signal: str) -> Optional[Dict[str, Any]]:
        """Retrieve mapping configuration for a production signal, or None if unsupported."""
        return self.SIGNAL_REGISTRY.get(signal.lower().strip())

    def create_reference(
        self, signal: str, observed_value: Optional[float] = None
    ) -> Optional[ResearchEvidenceReference]:
        """Create a typed ResearchEvidenceReference for a specific signal and observed value.

        Returns None if the signal has no verified research correspondence.
        """
        mapping = self.get_mapping(signal)
        if not mapping:
            return None

        val = round(observed_value, 2) if observed_value is not None else None

        return ResearchEvidenceReference(
            production_signal=signal.lower().strip(),
            observed_value=val,
            unit=mapping["unit"],
            research_degradation_code=mapping["research_degradation_code"],
            research_degradation_name=mapping["research_degradation_name"],
            b1_experiment_id=mapping["b1_experiment_id"],
            b1_available=mapping["b1_available"],
            b2_degradation_code=mapping["b2_degradation_code"],
            b2_available=mapping["b2_available"],
            b2_evaluated_policies_count=mapping["b2_evaluated_policies_count"],
            research_degradation=ResearchDegradationRef(
                code=mapping["research_degradation_code"],
                name=mapping["research_degradation_name"],
            ),
            b1=TrackEvidenceRef(
                experiment_id=mapping["b1_experiment_id"],
                available=mapping["b1_available"],
                details=mapping["b1_description"],
            ),
            b2=TrackEvidenceRef(
                experiment_id=None,  # Deliberately None: points to complete degradation coverage, not a single policy
                available=mapping["b2_available"],
                details=mapping["b2_description"],
            ),
        )

    def get_evidence_for_quality(self, quality: Any) -> List[ResearchEvidenceReference]:
        """Extract research evidence references from a document's quality analysis.

        Strictly applies detection semantics established in app/adapters/quality_adapter.py:
        - Rotation: detected when recommendations['needs_deskew'] is True or |rotation_angle| >= 2.0.
        - Noise: detected when recommendations['needs_denoise'] is True or noise_score > 12.0.
        - Blur: detected when blur_score < 30.0 (the documented threshold where blur_factor reaches 0.0).
        - Resolution: continuous approximation with NO production detection threshold; not auto-triggered.
        - Contrast: unmapped in B1 research; receives no reference.
        """
        if quality is None:
            return []

        evidence: List[ResearchEvidenceReference] = []

        # Extract values depending on whether input is QualityResultDTO, DocumentQuality model, or dict
        if isinstance(quality, dict):
            rot_angle = float(quality.get("rotation_angle", 0.0) or 0.0)
            blur_sc = float(quality.get("blur_score", 0.0) or 0.0)
            noise_sc = float(quality.get("noise_score", 0.0) or 0.0)
            recs = quality.get("recommendations", {}) or {}
        else:
            rot_angle = float(getattr(quality, "rotation_angle", 0.0) or 0.0)
            blur_sc = float(getattr(quality, "blur_score", 0.0) or 0.0)
            noise_sc = float(getattr(quality, "noise_score", 0.0) or 0.0)
            recs = getattr(quality, "recommendations", None)
            if not recs and hasattr(quality, "metrics_json") and isinstance(quality.metrics_json, dict):
                recs = quality.metrics_json.get("recommendations", {})
            if not recs:
                recs = {}

        # 1. Rotation Characteristic:
        # Existing production detection semantic: needs_deskew (abs(rotation_angle) >= 2.0)
        is_rotation_detected = recs.get("needs_deskew", False) or abs(rot_angle) >= 2.0
        if is_rotation_detected:
            ref = self.create_reference("rotation", observed_value=rot_angle)
            if ref:
                evidence.append(ref)

        # 2. Noise Characteristic:
        # Existing production detection semantic: needs_denoise (noise_score > 12.0)
        is_noise_detected = recs.get("needs_denoise", False) or noise_sc > 12.0
        if is_noise_detected:
            ref = self.create_reference("noise", observed_value=noise_sc)
            if ref:
                evidence.append(ref)

        # 3. Blur Characteristic:
        # Existing production documentation: < 30 is blurry (blur_factor == 0.0)
        is_blur_detected = blur_sc > 0.0 and blur_sc < 30.0
        if is_blur_detected:
            ref = self.create_reference("blur", observed_value=blur_sc)
            if ref:
                evidence.append(ref)

        return evidence


default_evidence_bridge = ResearchEvidenceBridge()
