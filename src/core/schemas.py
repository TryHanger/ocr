"""Core Data Transfer Objects (DTO) and schemas for OCR/KIE robustness research."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set

VALID_SPLITS: Set[str] = {"train", "validation", "test"}


def _validate_numeric(name: str, value: Any) -> float:
    """Validate that value is a real number (not bool, NaN, or inf)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}: {value}")
    if math.isnan(value) or math.isinf(value):
        raise ValueError(f"{name} cannot be NaN or Inf, got {value}")
    return float(value)


@dataclass(frozen=True)
class BoundingBox:
    """Represents axis-aligned 2D bounding box on an image."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def __post_init__(self) -> None:
        x_min = _validate_numeric("x_min", self.x_min)
        y_min = _validate_numeric("y_min", self.y_min)
        x_max = _validate_numeric("x_max", self.x_max)
        y_max = _validate_numeric("y_max", self.y_max)

        if x_min < 0 or y_min < 0:
            raise ValueError(f"Coordinates cannot be negative: ({x_min}, {y_min})")
        if x_max < x_min:
            raise ValueError(f"x_max ({x_max}) must be >= x_min ({x_min})")
        if y_max < y_min:
            raise ValueError(f"y_max ({y_max}) must be >= y_min ({y_min})")

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    @property
    def area(self) -> float:
        return self.width * self.height

    def to_dict(self) -> Dict[str, float]:
        return {
            "x_min": self.x_min,
            "y_min": self.y_min,
            "x_max": self.x_max,
            "y_max": self.y_max,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> BoundingBox:
        required_keys = {"x_min", "y_min", "x_max", "y_max"}
        missing = required_keys - set(data.keys())
        if missing:
            raise KeyError(f"Missing required BoundingBox keys: {missing}")
        return cls(
            x_min=data["x_min"],
            y_min=data["y_min"],
            x_max=data["x_max"],
            y_max=data["y_max"],
        )


@dataclass
class OCRToken:
    """Represents a single recognized token with spatial coordinates and confidence."""

    text: str
    bbox: BoundingBox
    confidence: Optional[float] = None

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError(f"Token text must be str, got {type(self.text).__name__}")

        if isinstance(self.bbox, dict):
            object.__setattr__(self, "bbox", BoundingBox.from_dict(self.bbox))
        elif not isinstance(self.bbox, BoundingBox):
            raise TypeError(f"Token bbox must be BoundingBox, got {type(self.bbox).__name__}")

        if self.confidence is not None:
            conf = _validate_numeric("confidence", self.confidence)
            if not (0.0 <= conf <= 1.0):
                raise ValueError(f"Confidence must be in range [0.0, 1.0], got {conf}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "bbox": self.bbox.to_dict(),
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> OCRToken:
        if "text" not in data or "bbox" not in data:
            raise KeyError("Missing required keys 'text' or 'bbox' for OCRToken")
        bbox_data = data["bbox"]
        bbox = bbox_data if isinstance(bbox_data, BoundingBox) else BoundingBox.from_dict(bbox_data)
        return cls(
            text=data["text"],
            bbox=bbox,
            confidence=data.get("confidence"),
        )


@dataclass
class OCRResult:
    """Represents the complete OCR engine output for a document."""

    document_id: str
    full_text: str
    tokens: List[OCRToken]
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_time_ms: Optional[float] = None
    model_name: Optional[str] = None
    model_version: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.document_id, str) or not self.document_id.strip():
            raise ValueError("document_id must be a non-empty string")
        if not isinstance(self.full_text, str):
            raise TypeError(f"full_text must be str, got {type(self.full_text).__name__}")
        if not isinstance(self.tokens, list):
            raise TypeError(f"tokens must be a list, got {type(self.tokens).__name__}")
        if not isinstance(self.metadata, dict):
            raise TypeError(f"metadata must be dict, got {type(self.metadata).__name__}")

        converted_tokens: List[OCRToken] = []
        for i, token in enumerate(self.tokens):
            if isinstance(token, dict):
                converted_tokens.append(OCRToken.from_dict(token))
            elif isinstance(token, OCRToken):
                converted_tokens.append(token)
            else:
                raise TypeError(f"tokens[{i}] must be OCRToken or dict, got {type(token).__name__}")
        self.tokens = converted_tokens

        if self.processing_time_ms is not None:
            ms = _validate_numeric("processing_time_ms", self.processing_time_ms)
            if ms < 0:
                raise ValueError(f"processing_time_ms cannot be negative, got {ms}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "full_text": self.full_text,
            "tokens": [t.to_dict() for t in self.tokens],
            "metadata": self.metadata,
            "processing_time_ms": self.processing_time_ms,
            "model_name": self.model_name,
            "model_version": self.model_version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> OCRResult:
        required = {"document_id", "full_text", "tokens"}
        missing = required - set(data.keys())
        if missing:
            raise KeyError(f"Missing required keys for OCRResult: {missing}")
        tokens = [
            t if isinstance(t, OCRToken) else OCRToken.from_dict(t)
            for t in data["tokens"]
        ]
        return cls(
            document_id=data["document_id"],
            full_text=data["full_text"],
            tokens=tokens,
            metadata=data.get("metadata", {}),
            processing_time_ms=data.get("processing_time_ms"),
            model_name=data.get("model_name"),
            model_version=data.get("model_version"),
        )


@dataclass
class KIEResult:
    """Represents structured key information extracted from a document."""

    document_id: str
    fields: Dict[str, Any]
    confidences: Optional[Dict[str, float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_time_ms: Optional[float] = None
    model_name: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.document_id, str) or not self.document_id.strip():
            raise ValueError("document_id must be a non-empty string")
        if not isinstance(self.fields, dict):
            raise TypeError(f"fields must be dict, got {type(self.fields).__name__}")
        if not isinstance(self.metadata, dict):
            raise TypeError(f"metadata must be dict, got {type(self.metadata).__name__}")

        if self.confidences is not None:
            if not isinstance(self.confidences, dict):
                raise TypeError(f"confidences must be dict, got {type(self.confidences).__name__}")
            for k, v in self.confidences.items():
                conf = _validate_numeric(f"confidences['{k}']", v)
                if not (0.0 <= conf <= 1.0):
                    raise ValueError(f"Confidence for '{k}' must be in [0.0, 1.0], got {conf}")

        if self.processing_time_ms is not None:
            ms = _validate_numeric("processing_time_ms", self.processing_time_ms)
            if ms < 0:
                raise ValueError(f"processing_time_ms cannot be negative, got {ms}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "fields": self.fields,
            "confidences": self.confidences,
            "metadata": self.metadata,
            "processing_time_ms": self.processing_time_ms,
            "model_name": self.model_name,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> KIEResult:
        if "document_id" not in data or "fields" not in data:
            raise KeyError("Missing required keys 'document_id' or 'fields' for KIEResult")
        return cls(
            document_id=data["document_id"],
            fields=data["fields"],
            confidences=data.get("confidences"),
            metadata=data.get("metadata", {}),
            processing_time_ms=data.get("processing_time_ms"),
            model_name=data.get("model_name"),
        )


@dataclass
class DocumentMetadata:
    """Metadata describing a source document and image attributes."""

    document_id: str
    image_path: str
    width: int
    height: int
    split: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.document_id, str) or not self.document_id.strip():
            raise ValueError("document_id must be a non-empty string")
        if not isinstance(self.image_path, str) or not self.image_path.strip():
            raise ValueError("image_path must be a non-empty string")

        if isinstance(self.width, bool) or not isinstance(self.width, int) or self.width <= 0:
            raise ValueError(f"width must be a positive integer, got {self.width}")
        if isinstance(self.height, bool) or not isinstance(self.height, int) or self.height <= 0:
            raise ValueError(f"height must be a positive integer, got {self.height}")

        if not isinstance(self.split, str) or self.split not in VALID_SPLITS:
            raise ValueError(f"split must be one of {VALID_SPLITS}, got '{self.split}'")

        if not isinstance(self.metadata, dict):
            raise TypeError(f"metadata must be dict, got {type(self.metadata).__name__}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "image_path": self.image_path,
            "width": self.width,
            "height": self.height,
            "split": self.split,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DocumentMetadata:
        required = {"document_id", "image_path", "width", "height", "split"}
        missing = required - set(data.keys())
        if missing:
            raise KeyError(f"Missing required DocumentMetadata keys: {missing}")
        return cls(
            document_id=data["document_id"],
            image_path=data["image_path"],
            width=data["width"],
            height=data["height"],
            split=data["split"],
            metadata=data.get("metadata", {}),
        )


@dataclass
class DegradationSpec:
    """Specification of an image degradation transformation."""

    type: str
    severity: int
    parameters: Dict[str, Any] = field(default_factory=dict)
    seed: int = 42

    def __post_init__(self) -> None:
        if not isinstance(self.type, str) or not self.type.strip():
            raise ValueError("Degradation type must be a non-empty string")

        if isinstance(self.severity, bool) or not isinstance(self.severity, int):
            raise TypeError(f"Severity must be an integer, got {type(self.severity).__name__}")
        if not (0 <= self.severity <= 4):
            raise ValueError(f"Severity must be between 0 (original) and 4 (severe), got {self.severity}")

        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError(f"Seed must be an integer, got {type(self.seed).__name__}")

        if not isinstance(self.parameters, dict):
            raise TypeError(f"parameters must be dict, got {type(self.parameters).__name__}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "severity": self.severity,
            "parameters": self.parameters,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DegradationSpec:
        if "type" not in data or "severity" not in data:
            raise KeyError("Missing required keys 'type' or 'severity' for DegradationSpec")
        return cls(
            type=data["type"],
            severity=data["severity"],
            parameters=data.get("parameters", {}),
            seed=data.get("seed", 42),
        )


@dataclass
class PreprocessingSpec:
    """Specification of preprocessing operations applied prior to OCR."""

    enabled: bool = True
    methods: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError(f"enabled must be bool, got {type(self.enabled).__name__}")
        if not isinstance(self.methods, list):
            raise TypeError(f"methods must be a list, got {type(self.methods).__name__}")
        for i, m in enumerate(self.methods):
            if not isinstance(m, str) or not m.strip():
                raise ValueError(f"methods[{i}] must be a non-empty string, got {m}")
        if not isinstance(self.parameters, dict):
            raise TypeError(f"parameters must be dict, got {type(self.parameters).__name__}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "methods": list(self.methods),
            "parameters": self.parameters,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PreprocessingSpec:
        return cls(
            enabled=data.get("enabled", True),
            methods=data.get("methods", []),
            parameters=data.get("parameters", {}),
        )


@dataclass
class ExperimentResult:
    """Complete experimental run record linking inputs, transformations, and metrics."""

    experiment_id: str
    document_id: str
    degradation: DegradationSpec
    preprocessing: PreprocessingSpec
    ocr_model: str
    kie_model: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.experiment_id, str) or not self.experiment_id.strip():
            raise ValueError("experiment_id must be a non-empty string")
        if not isinstance(self.document_id, str) or not self.document_id.strip():
            raise ValueError("document_id must be a non-empty string")
        if not isinstance(self.ocr_model, str) or not self.ocr_model.strip():
            raise ValueError("ocr_model must be a non-empty string")
        if not isinstance(self.kie_model, str) or not self.kie_model.strip():
            raise ValueError("kie_model must be a non-empty string")

        if isinstance(self.degradation, dict):
            self.degradation = DegradationSpec.from_dict(self.degradation)
        elif not isinstance(self.degradation, DegradationSpec):
            raise TypeError(f"degradation must be DegradationSpec, got {type(self.degradation).__name__}")

        if isinstance(self.preprocessing, dict):
            self.preprocessing = PreprocessingSpec.from_dict(self.preprocessing)
        elif not isinstance(self.preprocessing, PreprocessingSpec):
            raise TypeError(f"preprocessing must be PreprocessingSpec, got {type(self.preprocessing).__name__}")

        if not isinstance(self.metrics, dict):
            raise TypeError(f"metrics must be dict, got {type(self.metrics).__name__}")
        if not isinstance(self.metadata, dict):
            raise TypeError(f"metadata must be dict, got {type(self.metadata).__name__}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "document_id": self.document_id,
            "degradation": self.degradation.to_dict(),
            "preprocessing": self.preprocessing.to_dict(),
            "ocr_model": self.ocr_model,
            "kie_model": self.kie_model,
            "metrics": self.metrics,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ExperimentResult:
        required = {"experiment_id", "document_id", "degradation", "preprocessing", "ocr_model", "kie_model"}
        missing = required - set(data.keys())
        if missing:
            raise KeyError(f"Missing required ExperimentResult keys: {missing}")

        deg_data = data["degradation"]
        deg = deg_data if isinstance(deg_data, DegradationSpec) else DegradationSpec.from_dict(deg_data)

        prep_data = data["preprocessing"]
        prep = prep_data if isinstance(prep_data, PreprocessingSpec) else PreprocessingSpec.from_dict(prep_data)

        return cls(
            experiment_id=data["experiment_id"],
            document_id=data["document_id"],
            degradation=deg,
            preprocessing=prep,
            ocr_model=data["ocr_model"],
            kie_model=data["kie_model"],
            metrics=data.get("metrics", {}),
            metadata=data.get("metadata", {}),
        )


@dataclass
class OCRGroundTruth:
    """Ground truth reference data for OCR evaluation."""

    document_id: str
    text: str
    tokens: Optional[List[OCRToken]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.document_id, str) or not self.document_id.strip():
            raise ValueError("document_id must be a non-empty string")
        if not isinstance(self.text, str):
            raise TypeError(f"text must be str, got {type(self.text).__name__}")
        if not isinstance(self.metadata, dict):
            raise TypeError(f"metadata must be dict, got {type(self.metadata).__name__}")

        if self.tokens is not None:
            if not isinstance(self.tokens, list):
                raise TypeError(f"tokens must be a list, got {type(self.tokens).__name__}")
            converted: List[OCRToken] = []
            for i, t in enumerate(self.tokens):
                if isinstance(t, dict):
                    converted.append(OCRToken.from_dict(t))
                elif isinstance(t, OCRToken):
                    converted.append(t)
                else:
                    raise TypeError(f"tokens[{i}] must be OCRToken or dict, got {type(t).__name__}")
            self.tokens = converted

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "text": self.text,
            "tokens": [t.to_dict() for t in self.tokens] if self.tokens is not None else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> OCRGroundTruth:
        if "document_id" not in data or "text" not in data:
            raise KeyError("Missing required keys 'document_id' or 'text' for OCRGroundTruth")
        raw_tokens = data.get("tokens")
        tokens = None
        if raw_tokens is not None:
            tokens = [
                t if isinstance(t, OCRToken) else OCRToken.from_dict(t)
                for t in raw_tokens
            ]
        return cls(
            document_id=data["document_id"],
            text=data["text"],
            tokens=tokens,
            metadata=data.get("metadata", {}),
        )


@dataclass
class KIEGroundTruth:
    """Ground truth reference data for KIE evaluation."""

    document_id: str
    fields: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.document_id, str) or not self.document_id.strip():
            raise ValueError("document_id must be a non-empty string")
        if not isinstance(self.fields, dict):
            raise TypeError(f"fields must be dict, got {type(self.fields).__name__}")
        if not isinstance(self.metadata, dict):
            raise TypeError(f"metadata must be dict, got {type(self.metadata).__name__}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "fields": self.fields,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> KIEGroundTruth:
        if "document_id" not in data or "fields" not in data:
            raise KeyError("Missing required keys 'document_id' or 'fields' for KIEGroundTruth")
        return cls(
            document_id=data["document_id"],
            fields=data["fields"],
            metadata=data.get("metadata", {}),
        )
