"""RapidOCR engine implementation using PP-OCRv6 ONNX Runtime backend."""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from src.core.schemas import BoundingBox, OCRToken
from src.ocr.base import BaseOCREnginePrimitive


def _setup_cuda_dlls() -> None:
    """Ensure Windows dynamic link loader finds cuDNN and CUDA DLLs from torch or toolkit."""
    if platform.system() == "Windows":
        try:
            import torch
            torch_lib = Path(torch.__file__).parent / "lib"
            if torch_lib.exists():
                os.add_dll_directory(str(torch_lib))
                os.environ["PATH"] = str(torch_lib) + os.pathsep + os.environ.get("PATH", "")
        except ImportError:
            pass



# Expected model filenames and cryptographic SHA256 hashes for reproducibility lock
EXPECTED_MODELS = {
    "detector": {
        "family": "PP-OCRv6",
        "size": "small",
        "filename": "PP-OCRv6_det_small.onnx",
        "sha256": "090f04abcd9d9a7498bc4ebf677e4cb9bdce1fe4197ddb7e529f1ef44e1ff94f",
    },
    "recognizer": {
        "family": "PP-OCRv6",
        "size": "small",
        "filename": "PP-OCRv6_rec_small.onnx",
        "sha256": "6f327246b50388f3c176ae304bd95767ea6dc0c9ae92153ef8cbe210b3c14884",
    },
    "classifier": {
        "family": "PP-OCRv4",
        "size": "mobile",
        "filename": "ch_ppocr_mobile_v2.0_cls_mobile.onnx",
        "sha256": "e47acedf663230f8863ff1ab0e64dd2d82b838fceb5957146dab185a89d6215c",
    },
}


class RapidOCREngine(BaseOCREnginePrimitive):
    """RapidOCR engine baseline utilizing PP-OCRv6 models via ONNX Runtime.

    Engineering properties:
    - 100% local, offline inference via ONNX Runtime (zero external APIs).
    - Unified text detection (PP-OCRv6 small) and recognition (PP-OCRv6 small) pipeline.
    - CPU Execution Provider locked and verified.
    - Preserves oriented 4-point quadrilateral boxes in metadata alongside derived BoundingBoxes.
    - Deterministic execution on CPU without hidden enhancement preprocessing.
    - Dynamic model manifest discovery and SHA256 cryptographic verification.
    """

    def __init__(
        self,
        min_confidence: float = 0.0,
        line_tolerance_factor: float = 0.5,
        text_score: float = 0.5,
        use_cls: bool = True,
        use_preprocess_img: bool = True,
        max_side_len: int = 2000,
        min_side_len: int = 30,
        det_limit_side_len: int = 736,
        det_limit_type: str = "min",
        execution_provider: str = "cpu",
        **kwargs: Any,
    ) -> None:
        super().__init__(line_tolerance_factor=line_tolerance_factor)
        self.min_confidence = float(min_confidence)
        self.text_score = float(text_score)
        self.use_cls = bool(use_cls)
        self.use_preprocess_img = bool(use_preprocess_img)
        self.max_side_len = int(max_side_len)
        self.min_side_len = int(min_side_len)
        self.det_limit_side_len = int(det_limit_side_len)
        self.det_limit_type = str(det_limit_type)
        ep_clean = str(execution_provider).strip().lower()
        if ep_clean not in ("cpu", "cuda"):
            raise ValueError(
                f"Unsupported execution_provider: '{execution_provider}'. "
                f"Supported providers: ['cpu', 'cuda']"
            )
        self.execution_provider = ep_clean
        self.custom_kwargs = kwargs
        self.model_name = "RapidOCR"
        self.model_version = "3.9.2 (PP-OCRv6 small)"

        # Lazy import of RapidOCR
        try:
            from rapidocr import RapidOCR
        except ImportError:
            try:
                from rapidocr_onnxruntime import RapidOCR
            except ImportError as exc:
                raise ImportError(
                    "rapidocr is not installed. Please install it with: pip install rapidocr==3.9.2"
                ) from exc

        # Construct explicit parameters for RapidOCR runtime
        params: Dict[str, Any] = {
            "Global.text_score": self.text_score,
            "Global.use_cls": self.use_cls,
            "Global.use_preprocess_img": self.use_preprocess_img,
            "Global.max_side_len": self.max_side_len,
            "Global.min_side_len": self.min_side_len,
            "Det.limit_side_len": self.det_limit_side_len,
            "Det.limit_type": self.det_limit_type,
        }
        if self.execution_provider == "cuda":
            _setup_cuda_dlls()
            params["EngineConfig.onnxruntime.use_cuda"] = True
        else:
            params["EngineConfig.onnxruntime.use_cuda"] = False

        for k, v in kwargs.items():
            params[k] = v

        self._engine = RapidOCR(params=params)

        # Fail-fast validation: verify ONNX Runtime didn't silently fall back or misconfigure
        det_sess = getattr(self._engine.text_det, "session", None)
        det_ort = getattr(det_sess, "session", None)
        rec_sess = getattr(self._engine.text_rec, "session", None)
        rec_ort = getattr(rec_sess, "session", None)

        det_providers = det_ort.get_providers() if det_ort else []
        rec_providers = rec_ort.get_providers() if rec_ort else []
        cls_providers = []
        if self.use_cls:
            cls_sess = getattr(self._engine.text_cls, "session", None)
            cls_ort = getattr(cls_sess, "session", None)
            cls_providers = cls_ort.get_providers() if cls_ort else []

        if self.execution_provider == "cuda":
            fallback_errors = []
            if not det_providers or det_providers[0] != "CUDAExecutionProvider":
                fallback_errors.append(f"Detector (active: {det_providers})")
            if not rec_providers or rec_providers[0] != "CUDAExecutionProvider":
                fallback_errors.append(f"Recognizer (active: {rec_providers})")
            if self.use_cls and (not cls_providers or cls_providers[0] != "CUDAExecutionProvider"):
                fallback_errors.append(f"Classifier (active: {cls_providers})")
            if fallback_errors:
                raise RuntimeError(
                    f"Requested execution_provider='cuda', but ONNX Runtime failed to activate "
                    f"CUDAExecutionProvider as primary provider (silent fallback detected). "
                    f"Failures: {', '.join(fallback_errors)}"
                )
        elif self.execution_provider == "cpu":
            if det_providers and det_providers[0] != "CPUExecutionProvider":
                raise RuntimeError(
                    f"Requested execution_provider='cpu', but detector primary provider is {det_providers[0]}"
                )


    def _resolve_session_model(self, component: Any) -> Dict[str, Any]:
        """Dynamically inspect an active RapidOCR component to extract actual model path and hash.

        NOTE: Model path resolution currently relies on a private RapidOCR wrapper attribute
        (inner_session._model_path) and is therefore version-sensitive. If this attribute disappears
        in a future version, resolution must fail loudly rather than silently continuing with an
        incomplete manifest.
        """
        if not hasattr(component, "session"):
            raise RuntimeError(
                f"RapidOCR component {component} has no 'session' attribute. "
                "Model path resolution currently relies on a private RapidOCR wrapper attribute "
                "and is therefore version-sensitive."
            )

        ort_session = component.session
        inner_session = getattr(ort_session, "session", None)
        if inner_session is None:
            raise RuntimeError(
                f"OrtInferSession in {component} has no inner 'session' (InferenceSession). "
                "Model path resolution currently relies on a private RapidOCR wrapper attribute "
                "and is therefore version-sensitive."
            )

        model_path_str = getattr(inner_session, "_model_path", None)
        if not model_path_str:
            raise RuntimeError(
                f"ONNX Runtime InferenceSession for {component} has no '_model_path' attribute. "
                "Model path resolution currently relies on a private RapidOCR wrapper attribute "
                "and is therefore version-sensitive."
            )

        p = Path(model_path_str)
        exists = p.exists()
        sha256_hash = hashlib.sha256(p.read_bytes()).hexdigest() if exists else None

        return {
            "model_file": p.name,
            "model_path": str(p).replace("\\", "/"),
            "file_exists": exists,
            "file_size_bytes": p.stat().st_size if exists else None,
            "sha256": sha256_hash,
            "providers": inner_session.get_providers(),
        }

    def get_model_manifest(self) -> Dict[str, Any]:
        """Return comprehensive machine-readable model manifest distinguishing configured, resolved, and environment."""
        try:
            rap_version = importlib.metadata.version("rapidocr")
        except Exception:
            try:
                rap_version = importlib.metadata.version("rapidocr_onnxruntime")
            except Exception:
                rap_version = "unknown"

        try:
            ort_version = importlib.metadata.version("onnxruntime")
        except Exception:
            ort_version = "unknown"

        det_info = self._resolve_session_model(self._engine.text_det)
        rec_info = self._resolve_session_model(self._engine.text_rec)
        cls_info = (
            self._resolve_session_model(self._engine.text_cls)
            if self.use_cls
            else {"enabled": False, "comment": "Classifier disabled via configuration"}
        )

        expected_ep = "CUDAExecutionProvider" if self.execution_provider == "cuda" else "CPUExecutionProvider"
        environment = {
            "platform": sys.platform,
            "python_version": sys.version.split()[0],
            "rapidocr_version": rap_version,
            "onnxruntime_version": ort_version,
            "execution_provider": expected_ep,
            "configured_provider": self.execution_provider,
            "opencv_version": cv2.__version__,
            "numpy_version": np.__version__,
        }

        # Resolve resize policy from active runtime instance
        cfg_global = getattr(getattr(self._engine, "cfg", None), "Global", None)
        resolved_use_pre = getattr(
            cfg_global,
            "use_preprocess_img",
            getattr(self._engine, "use_preprocess_img", self.use_preprocess_img),
        )
        resolved_max_side = getattr(self._engine, "max_side_len", self.max_side_len)
        resolved_min_side = getattr(self._engine, "min_side_len", self.min_side_len)
        resolved_det_limit = getattr(self._engine.text_det, "limit_side_len", self.det_limit_side_len)
        resolved_det_type = getattr(self._engine.text_det, "limit_type", self.det_limit_type)

        resize_configured = {
            "use_preprocess_img": self.use_preprocess_img,
            "max_side_len": self.max_side_len,
            "min_side_len": self.min_side_len,
            "det_limit_side_len": self.det_limit_side_len,
            "det_limit_type": self.det_limit_type,
            "interpolation": "cv2.INTER_LINEAR",
        }

        resize_resolved = {
            "use_preprocess_img": resolved_use_pre,
            "max_side_len": resolved_max_side,
            "min_side_len": resolved_min_side,
            "det_limit_side_len": resolved_det_limit,
            "det_limit_type": resolved_det_type,
            "interpolation": "cv2.INTER_LINEAR",
            "order_of_application": [
                "1. Global preprocess_img (bounds check [min_side_len, max_side_len], cv2.resize bilinear)",
                "2. TextDet DetPreProcess (limit_side_len check, cv2.resize bilinear to multiple of 32)",
            ],
            "coordinate_restoration": "Detections mapped back to input image coordinate space",
            "baseline_consistency_rule": (
                "B0, B1 and B2 use the same OCR-side resize policy and the same OCR configuration. "
                "The resulting resized dimensions may legitimately differ because the input images differ "
                "after degradation/preprocessing."
            ),
        }

        resize_verified = (
            resolved_use_pre == resize_configured["use_preprocess_img"]
            and resolved_max_side == resize_configured["max_side_len"]
            and resolved_min_side == resize_configured["min_side_len"]
            and resolved_det_limit == resize_configured["det_limit_side_len"]
            and resolved_det_type == resize_configured["det_limit_type"]
        )

        resize_policy_block = {
            "configured": resize_configured,
            "resolved": resize_resolved,
            "verified": resize_verified,
        }

        configured = {
            "name": "RapidOCR",
            "version": self.model_version,
            "engine": "rapidocr",
            "backend": "onnxruntime",
            "detector": EXPECTED_MODELS["detector"]["filename"],
            "recognizer": EXPECTED_MODELS["recognizer"]["filename"],
            "classifier": EXPECTED_MODELS["classifier"]["filename"],
            "models": {
                "detector": {
                    "family": EXPECTED_MODELS["detector"]["family"],
                    "size": EXPECTED_MODELS["detector"]["size"],
                    "expected_model_file": EXPECTED_MODELS["detector"]["filename"],
                    "expected_sha256": EXPECTED_MODELS["detector"]["sha256"],
                },
                "recognizer": {
                    "family": EXPECTED_MODELS["recognizer"]["family"],
                    "size": EXPECTED_MODELS["recognizer"]["size"],
                    "expected_model_file": EXPECTED_MODELS["recognizer"]["filename"],
                    "expected_sha256": EXPECTED_MODELS["recognizer"]["sha256"],
                },
                "classifier": {
                    "enabled": self.use_cls,
                    "family": EXPECTED_MODELS["classifier"]["family"],
                    "size": EXPECTED_MODELS["classifier"]["size"],
                    "expected_model_file": EXPECTED_MODELS["classifier"]["filename"],
                    "expected_sha256": EXPECTED_MODELS["classifier"]["sha256"],
                },
            },
            "runtime": {
                "provider": expected_ep,
                "execution_provider": self.execution_provider,
            },
            "resize_policy": resize_configured,
        }

        def _format_resolved_model(model_key: str, info: Dict[str, Any]) -> Dict[str, Any]:
            if not isinstance(info, dict) or not info.get("file_exists", False):
                return {
                    "family": EXPECTED_MODELS[model_key]["family"],
                    "size": EXPECTED_MODELS[model_key]["size"],
                    "filename": info.get("model_file") if isinstance(info, dict) else None,
                    "path": info.get("model_path") if isinstance(info, dict) else None,
                    "exists": False,
                    "file_exists": False,
                    "size_bytes": 0,
                    "file_size_bytes": 0,
                    "sha256": None,
                    "sha256_verified": False,
                    "providers": [],
                }
            sha = info.get("sha256")
            expected_sha = EXPECTED_MODELS[model_key]["sha256"]
            return {
                "family": EXPECTED_MODELS[model_key]["family"],
                "size": EXPECTED_MODELS[model_key]["size"],
                "filename": info.get("model_file"),
                "model_file": info.get("model_file"),
                "path": info.get("model_path"),
                "model_path": info.get("model_path"),
                "exists": True,
                "file_exists": True,
                "size_bytes": info.get("file_size_bytes"),
                "file_size_bytes": info.get("file_size_bytes"),
                "sha256": sha,
                "sha256_verified": (sha == expected_sha),
                "providers": info.get("providers", []),
            }

        resolved_det = _format_resolved_model("detector", det_info)
        resolved_rec = _format_resolved_model("recognizer", rec_info)
        resolved_cls = (
            _format_resolved_model("classifier", cls_info)
            if self.use_cls and isinstance(cls_info, dict)
            else {"enabled": False, "comment": "Classifier disabled via configuration"}
        )

        actual_providers = det_info.get("providers", ["unknown"])

        resolved = {
            "detector": resolved_det,
            "recognizer": resolved_rec,
            "classifier": resolved_cls,
            "runtime": {
                "actual_providers": actual_providers,
                "is_cpu_only": actual_providers == ["CPUExecutionProvider"],
                "configured_provider": self.execution_provider,
            },
            "resize_policy": resize_resolved,
        }

        return {
            "configured": configured,
            "resolved": resolved,
            "environment": environment,
            "resize_policy": resize_policy_block,
        }

    def verify_model_stack(self) -> Tuple[bool, List[str]]:
        """Verify that resolved models and runtime match expected cryptographic hashes and versions.

        Returns:
            Tuple of (is_valid, list_of_error_strings).
        """
        manifest = self.get_model_manifest()
        errors: List[str] = []

        resolved_det = manifest["resolved"]["detector"]
        resolved_rec = manifest["resolved"]["recognizer"]
        resolved_cls = manifest["resolved"]["classifier"]

        # 1. Detector
        if not resolved_det.get("file_exists"):
            errors.append(f"Detector model file does not exist: {resolved_det.get('model_path')}")
        elif resolved_det.get("sha256") != EXPECTED_MODELS["detector"]["sha256"]:
            errors.append(
                f"Detector SHA256 mismatch! Expected {EXPECTED_MODELS['detector']['sha256']}, got {resolved_det.get('sha256')}"
            )

        # 2. Recognizer
        if not resolved_rec.get("file_exists"):
            errors.append(f"Recognizer model file does not exist: {resolved_rec.get('model_path')}")
        elif resolved_rec.get("sha256") != EXPECTED_MODELS["recognizer"]["sha256"]:
            errors.append(
                f"Recognizer SHA256 mismatch! Expected {EXPECTED_MODELS['recognizer']['sha256']}, got {resolved_rec.get('sha256')}"
            )

        # 3. Classifier (if enabled)
        if self.use_cls:
            if not resolved_cls.get("file_exists"):
                errors.append(f"Classifier model file does not exist: {resolved_cls.get('model_path')}")
            elif resolved_cls.get("sha256") != EXPECTED_MODELS["classifier"]["sha256"]:
                errors.append(
                    f"Classifier SHA256 mismatch! Expected {EXPECTED_MODELS['classifier']['sha256']}, got {resolved_cls.get('sha256')}"
                )

        # 4. Providers
        providers = manifest["resolved"]["runtime"]["actual_providers"]
        expected_ep = "CUDAExecutionProvider" if self.execution_provider == "cuda" else "CPUExecutionProvider"
        if not providers or providers[0] != expected_ep:
            errors.append(f"Expected primary provider '{expected_ep}', but active providers are: {providers}")


        # 5. Resize policy verification
        if not manifest.get("resize_policy", {}).get("verified", False):
            errors.append("Resize policy verification failed: resolved runtime parameters differ from configured.")

        return len(errors) == 0, errors

    def _recognize_impl(
        self, image: np.ndarray, document_id: str
    ) -> Tuple[List[OCRToken], Dict[str, Any]]:
        # Color space convention: Convert to BGR format for OpenCV-based RapidOCR
        if image.ndim == 2:
            img_input = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.ndim == 3 and image.shape[2] == 1:
            img_input = cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2BGR)
        elif image.ndim == 3 and image.shape[2] == 4:
            img_input = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        else:
            # Dataset adapter provides RGB. Convert to BGR for RapidOCR.
            img_input = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        t0 = time.perf_counter()
        raw_output = self._engine(img_input)
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        # Handle RapidOCROutput object (rapidocr >= 3.9) as well as legacy tuple output
        if hasattr(raw_output, "boxes") and hasattr(raw_output, "txts") and hasattr(raw_output, "scores"):
            raw_boxes = raw_output.boxes
            raw_txts = raw_output.txts
            raw_scores = raw_output.scores
            elapse_breakdown = getattr(raw_output, "elapse_list", None) or getattr(raw_output, "elapse", None)
            if raw_boxes is not None and raw_txts is not None and raw_scores is not None:
                items = list(zip(raw_boxes, raw_txts, raw_scores))
            else:
                items = []
        elif isinstance(raw_output, tuple) and len(raw_output) == 2:
            items = raw_output[0] if raw_output[0] is not None else []
            elapse_breakdown = raw_output[1]
        else:
            items = []
            elapse_breakdown = None

        tokens: List[OCRToken] = []
        raw_quads: List[List[List[float]]] = []

        for item in items:
            quad_pts, text_val, conf_val = item
            text_clean = str(text_val).strip()
            conf_float = float(conf_val) if conf_val is not None else 0.0
            conf_float = max(0.0, min(1.0, conf_float))

            if conf_float < self.min_confidence:
                continue

            pts = [[float(pt[0]), float(pt[1])] for pt in quad_pts]
            raw_quads.append(pts)

            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]

            x_min, x_max = float(min(xs)), float(max(xs))
            y_min, y_max = float(min(ys)), float(max(ys))

            # Guarantee non-zero dimensions
            if x_max <= x_min:
                x_max = x_min + 1.0
            if y_max <= y_min:
                y_max = y_min + 1.0

            bbox = BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)
            tokens.append(OCRToken(text=text_clean, bbox=bbox, confidence=conf_float))

        meta = {
            "processing_time_ms": elapsed_ms,
            "model_name": "RapidOCR",
            "model_version": "3.9.2 (PP-OCRv6 small)",
            "metadata": {
                "raw_detections_count": len(items),
                "extracted_tokens_count": len(tokens),
                "elapse_breakdown": elapse_breakdown,
                "raw_quadrilaterals": raw_quads,
            },
        }

        return tokens, meta
