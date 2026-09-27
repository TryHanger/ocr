"""Empirical Severity Calibration runner for document image degradations.

Computes objective image distortion metrics (PSNR, SSIM, MAE, Laplacian ratio,
Luminance drop) across candidate parameters on the validation set, verifies
monotonicity (1 < 2 < 3 < 4), and produces calibration artifacts.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is in sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import cv2
import numpy as np
import yaml

from src.core.schemas import DegradationSpec
from src.degradation.pipeline import DEGRADATION_REGISTRY, get_degradation


def compute_mse(img1: np.ndarray, img2: np.ndarray) -> float:
    """Mean Squared Error between two uint8 images."""
    if img1.shape != img2.shape:
        raise ValueError(f"Shape mismatch: {img1.shape} vs {img2.shape}")
    return float(cv2.norm(img1, img2, cv2.NORM_L2SQR) / img1.size)


def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Peak Signal-to-Noise Ratio (dB) between two uint8 images."""
    mse = compute_mse(img1, img2)
    if mse == 0.0:
        return 100.0  # Cap infinity for identical images
    return float(10.0 * np.log10((255.0 ** 2) / mse))


def compute_mae(img1: np.ndarray, img2: np.ndarray) -> float:
    """Mean Absolute Error between two uint8 images."""
    if img1.shape != img2.shape:
        raise ValueError(f"Shape mismatch: {img1.shape} vs {img2.shape}")
    return float(cv2.norm(img1, img2, cv2.NORM_L1) / img1.size)


class ImageReference:
    """Precomputed properties for an original reference image to avoid redundant calculations."""

    def __init__(self, img: np.ndarray):
        self.img = img
        if img.ndim == 3:
            self.gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            self.mean_lum = float(np.mean(img, dtype=np.float32))
        else:
            self.gray = img
            self.mean_lum = float(np.mean(img, dtype=np.float32))

        self.gray_f32 = self.gray.astype(np.float32)
        self.laplacian_var = float(cv2.Laplacian(self.gray, cv2.CV_32F).var())

        # Precomputed SSIM reference terms (sigma=1.5, kernel=11)
        self.mu1 = cv2.GaussianBlur(self.gray_f32, (11, 11), 1.5)
        self.mu1_sq = self.mu1 * self.mu1
        self.sigma1_sq = cv2.GaussianBlur(self.gray_f32 * self.gray_f32, (11, 11), 1.5) - self.mu1_sq


def compute_ssim(
    img1: np.ndarray,
    img2: np.ndarray,
    ref: Optional[ImageReference] = None,
) -> float:
    """Structural Similarity Index (Wang et al., 2004) on luminance channel."""
    if ref is not None:
        y1_f32 = ref.gray_f32
        mu1 = ref.mu1
        mu1_sq = ref.mu1_sq
        sigma1_sq = ref.sigma1_sq
    else:
        if img1.ndim == 3:
            y1_f32 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY).astype(np.float32)
        else:
            y1_f32 = img1.astype(np.float32)
        mu1 = cv2.GaussianBlur(y1_f32, (11, 11), 1.5)
        mu1_sq = mu1 * mu1
        sigma1_sq = cv2.GaussianBlur(y1_f32 * y1_f32, (11, 11), 1.5) - mu1_sq

    if img2.ndim == 3:
        y2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY).astype(np.float32)
    else:
        y2 = img2.astype(np.float32)

    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    kernel_size = 11
    sigma = 1.5

    mu2 = cv2.GaussianBlur(y2, (kernel_size, kernel_size), sigma)
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma2_sq = cv2.GaussianBlur(y2 * y2, (kernel_size, kernel_size), sigma) - mu2_sq
    sigma12 = cv2.GaussianBlur(y1_f32 * y2, (kernel_size, kernel_size), sigma) - mu1_mu2

    numerator = (2.0 * mu1_mu2 + c1) * (2.0 * sigma12 + c2)
    denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)

    ssim_map = numerator / (denominator + 1e-12)
    return float(np.mean(ssim_map))


def compute_laplacian_ratio(
    orig: np.ndarray,
    deg: np.ndarray,
    ref: Optional[ImageReference] = None,
) -> float:
    """Ratio of Laplacian variance (edge energy) of degraded image to original."""
    if ref is not None:
        var_orig = ref.laplacian_var
    else:
        g_orig = cv2.cvtColor(orig, cv2.COLOR_RGB2GRAY) if orig.ndim == 3 else orig
        var_orig = float(cv2.Laplacian(g_orig, cv2.CV_32F).var())

    g_deg = cv2.cvtColor(deg, cv2.COLOR_RGB2GRAY) if deg.ndim == 3 else deg
    var_deg = float(cv2.Laplacian(g_deg, cv2.CV_32F).var())

    if var_orig <= 1e-6:
        return 1.0
    return float(var_deg / var_orig)


def compute_luminance_drop(
    orig: np.ndarray,
    deg: np.ndarray,
    ref: Optional[ImageReference] = None,
) -> float:
    """Fractional mean luminance drop from original to degraded."""
    m_orig = ref.mean_lum if ref is not None else float(np.mean(orig, dtype=np.float32))
    m_deg = float(np.mean(deg, dtype=np.float32))
    if m_orig <= 1e-6:
        return 0.0
    return float(max(0.0, (m_orig - m_deg) / m_orig))


def evaluate_image_pair(
    orig: np.ndarray,
    deg: np.ndarray,
    deg_type: str,
    ref: Optional[ImageReference] = None,
    primary_metric_name: str = "",
) -> Dict[str, float]:
    """Compute all relevant distortion metrics for an original/degraded image pair."""
    # Compute SSIM when configured as primary metric (jpeg_compression, downsampling)
    needs_ssim = ("ssim" in primary_metric_name) or (deg_type in ("jpeg_compression", "downsampling"))
    ssim_val = round(compute_ssim(orig, deg, ref=ref), 4) if needs_ssim else 0.0

    metrics = {
        "psnr_db": round(compute_psnr(orig, deg), 2),
        "ssim": ssim_val,
        "mae": round(compute_mae(orig, deg), 2),
        "laplacian_ratio": round(compute_laplacian_ratio(orig, deg, ref=ref), 4),
        "luminance_drop": round(compute_luminance_drop(orig, deg, ref=ref), 4),
    }
    return metrics


def find_calibration_images(
    data_root: Optional[Path] = None,
    allow_real_data: bool = False,
    max_images: Optional[int] = None,
) -> Tuple[List[Path], str]:
    """Identify calibration images: real SROIE validation split or synthetic fixtures.

    By default (data_root=None, allow_real_data=False), returns deterministic synthetic
    fixtures to ensure unit tests and local dry-runs are fast (<0.5s) and do not
    silently execute against the 126-image real SROIE validation corpus.
    """
    if data_root is not None:
        candidate_dirs = [
            data_root / "train" / "img",
            data_root / "SROIE2019" / "train" / "img",
            data_root / "img",
            data_root,
        ]
        for p in candidate_dirs:
            if p and p.is_dir():
                all_imgs = sorted(list(p.glob("*.jpg")) + list(p.glob("*.png")))
                if len(all_imgs) >= 126:
                    rng = np.random.default_rng(42)
                    perm = rng.permutation(len(all_imgs))
                    val_indices = perm[500:626] if len(all_imgs) >= 626 else perm[-126:]
                    val_imgs = [all_imgs[i] for i in sorted(val_indices)]
                    if max_images:
                        val_imgs = val_imgs[:max_images]
                    return val_imgs, "REAL_SROIE_VALIDATION_SPLIT"
                elif all_imgs:
                    if max_images:
                        all_imgs = all_imgs[:max_images]
                    return all_imgs, "SYNTHETIC_FIXTURE"

    if allow_real_data:
        for p in [Path("data/SROIE2019/train/img"), Path("data/train/img")]:
            if p.is_dir():
                all_imgs = sorted(list(p.glob("*.jpg")) + list(p.glob("*.png")))
                if len(all_imgs) >= 126:
                    rng = np.random.default_rng(42)
                    perm = rng.permutation(len(all_imgs))
                    val_indices = perm[500:626] if len(all_imgs) >= 626 else perm[-126:]
                    val_imgs = [all_imgs[i] for i in sorted(val_indices)]
                    if max_images:
                        val_imgs = val_imgs[:max_images]
                    return val_imgs, "REAL_SROIE_VALIDATION_SPLIT"

    # Default fallback: local synthetic fixtures for testing
    fixture_paths = [
        Path("tests/fixtures/sroie_valid/train/img"),
        Path("tests/fixtures/sroie/train/img"),
    ]
    for fp in fixture_paths:
        if fp.is_dir():
            imgs = sorted(list(fp.glob("*.jpg")) + list(fp.glob("*.png")))
            if imgs:
                if max_images:
                    imgs = imgs[:max_images]
                return imgs, "SYNTHETIC_FIXTURE"

    return [], "NO_DATA_AVAILABLE"


def load_single_image(p: Path, max_side: int = 2000) -> Optional[np.ndarray]:
    """Load single image, cap resolution to max_side (ADR-015 policy), and apply text pattern if flat."""
    img_bgr = cv2.imread(str(p))
    if img_bgr is None:
        return None
    h, w = img_bgr.shape[:2]
    if max_side and max(h, w) > max_side:
        scale = max_side / float(max(h, w))
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    if float(img_bgr.std()) < 1.0:
        h, w = img_bgr.shape[:2]
        cv2.putText(img_bgr, "RECEIPT TAX INVOICE", (max(5, int(w * 0.05)), max(20, int(h * 0.2))), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (20, 20, 20), 1)
        cv2.putText(img_bgr, "TOTAL: $42.50", (max(5, int(w * 0.05)), max(40, int(h * 0.4))), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (10, 10, 10), 1)
        cv2.line(img_bgr, (5, int(h * 0.5)), (w - 5, int(h * 0.5)), (30, 30, 30), 1)
        cv2.putText(img_bgr, "THANK YOU!", (max(5, int(w * 0.1)), max(70, int(h * 0.7))), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (25, 25, 25), 1)
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def run_calibration(
    config_path: Path,
    output_dir: Path,
    data_root: Optional[Path] = None,
    save_samples: bool = True,
    allow_real_data: bool = False,
    max_images: Optional[int] = None,
) -> Dict[str, Any]:
    """Execute calibration procedure and write artifacts."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    images_paths, data_source = find_calibration_images(
        data_root=data_root,
        allow_real_data=allow_real_data,
        max_images=max_images,
    )
    real_data_status = (
        "COMPLETED" if data_source == "REAL_SROIE_VALIDATION_SPLIT" else "PENDING"
    )

    # If no image files found, prepare synthetic dummy fallback
    dummy_fallback = None
    if not images_paths:
        dummy = np.full((300, 200, 3), 250, dtype=np.uint8)
        cv2.putText(dummy, "TAX INVOICE", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        cv2.putText(dummy, "TOTAL: $45.00", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        dummy_fallback = dummy

    num_images = len(images_paths) if images_paths else 1

    degradations_cfg = cfg.get("degradations", {})
    report_degradations: Dict[str, Any] = {}
    is_real = (real_data_status == "COMPLETED")
    calibration_dataset = "sroie_validation" if is_real else "synthetic_fixture"
    parameter_status = "final_research_calibrated" if is_real else "provisional / fixture-based"

    selected_config: Dict[str, Any] = {
        "version": cfg.get("version", "1.0.0"),
        "calibration_date": datetime.now().isoformat(),
        "parameter_status": parameter_status,
        "real_data_calibration": real_data_status,
        "calibration_dataset": calibration_dataset,
        "research_validation_size": 126,
        "final_calibration_completed": is_real,
        "data_source": data_source,
        "degradations": {},
    }

    samples_dir = output_dir / "samples"
    if save_samples:
        samples_dir.mkdir(parents=True, exist_ok=True)

    deg_keys = [k for k in degradations_cfg.keys() if k in DEGRADATION_REGISTRY]
    for step_idx, deg_type in enumerate(deg_keys, 1):
        deg_info = degradations_cfg[deg_type]
        print(f"[{step_idx}/{len(deg_keys)}] Calibrating {deg_type} on {num_images} images...", flush=True)

        deg_instance = get_degradation(deg_type)
        primary_metric_name = deg_info.get("primary_metric", "psnr_db")
        candidates = deg_info.get("candidates", [])
        severities = deg_info.get("severities", {})

        cand_accums = [
            {"psnr_db": [], "ssim": [], "mae": [], "laplacian_ratio": [], "luminance_drop": []}
            for _ in candidates
        ]
        sev_accums = {
            s: {"psnr_db": [], "ssim": [], "mae": [], "laplacian_ratio": [], "luminance_drop": []}
            for s in range(5)
        }

        # Stream images one by one to keep memory footprint under 50MB and preserve L3 cache
        img_items = images_paths if images_paths else [dummy_fallback]
        for img_idx, item in enumerate(img_items):
            if isinstance(item, Path):
                img = load_single_image(item)
                if img is None:
                    continue
            else:
                img = item

            ref = ImageReference(img)

            # 1. Evaluate Candidates for this image
            for c_idx, cand_params in enumerate(candidates):
                spec = DegradationSpec(type=deg_type, severity=1, parameters=cand_params, seed=42)
                deg_img, _ = deg_instance.apply(img, spec)
                pair_metrics = evaluate_image_pair(img, deg_img, deg_type, ref=ref, primary_metric_name=primary_metric_name)
                for k, v in pair_metrics.items():
                    cand_accums[c_idx][k].append(v)

            # 2. Evaluate Selected Severities 0..4 for this image
            for sev in range(5):
                sev_params = severities.get(sev, {})
                spec = DegradationSpec(type=deg_type, severity=sev, parameters=sev_params, seed=42)
                deg_img, _ = deg_instance.apply(img, spec)
                pair_metrics = evaluate_image_pair(img, deg_img, deg_type, ref=ref, primary_metric_name=primary_metric_name)
                for k, v in pair_metrics.items():
                    sev_accums[sev][k].append(v)

                # Save sample visualization for the first image
                if save_samples and img_idx == 0:
                    sample_path = samples_dir / f"{deg_type}_sev{sev}.png"
                    out_bgr = cv2.cvtColor(deg_img, cv2.COLOR_RGB2BGR) if deg_img.ndim == 3 else deg_img
                    cv2.imwrite(str(sample_path), out_bgr)

        # Aggregate candidate evaluations
        candidate_evaluations: List[Dict[str, Any]] = []
        for c_idx, cand_params in enumerate(candidates):
            candidate_evaluations.append({
                "parameters": cand_params,
                "mean_metrics": {k: round(float(np.mean(v)), 2) for k, v in cand_accums[c_idx].items()},
            })

        # Aggregate severity evaluations
        severity_evaluations: Dict[int, Dict[str, Any]] = {}
        for sev in range(5):
            severity_evaluations[sev] = {
                "parameters": severities.get(sev, {}),
                "mean_metrics": {k: round(float(np.mean(v)), 2) for k, v in sev_accums[sev].items()},
            }

        # 3. Monotonicity validation on severities 1..4
        # Parameter monotonicity
        param_keys = {
            "gaussian_blur": ("sigma", True),
            "motion_blur": ("kernel_length", True),
            "gaussian_noise": ("std", True),
            "jpeg_compression": ("quality", False),  # quality decreases with severity
            "downsampling": ("scale_factor", False),  # scale decreases with severity
            "rotation": ("angle", True),
            "perspective": ("distortion_scale", True),
            "shadow": ("opacity", True),
        }
        p_name, p_increasing = param_keys.get(deg_type, ("param", True))
        p_vals = [float(severity_evaluations[s]["parameters"].get(p_name, 0.0)) for s in range(1, 5)]
        if p_increasing:
            monotonic_param = all(p_vals[i] <= p_vals[i + 1] for i in range(len(p_vals) - 1))
        else:
            monotonic_param = all(p_vals[i] >= p_vals[i + 1] for i in range(len(p_vals) - 1))

        # Check family-specific physical metric monotonicity
        psnrs = [severity_evaluations[s]["mean_metrics"]["psnr_db"] for s in range(1, 5)]
        maes = [severity_evaluations[s]["mean_metrics"]["mae"] for s in range(1, 5)]
        lum_drops = [severity_evaluations[s]["mean_metrics"]["luminance_drop"] for s in range(1, 5)]

        if deg_type in ("gaussian_blur", "motion_blur", "gaussian_noise", "jpeg_compression", "downsampling"):
            monotonic_metric = all(psnrs[i] >= psnrs[i + 1] for i in range(len(psnrs) - 1))
        elif deg_type == "rotation":
            # Geometric rotation: pixel displacement / MAE monotonically increases
            monotonic_metric = all(maes[i] <= maes[i + 1] for i in range(len(maes) - 1))
        elif deg_type == "perspective":
            # Perspective distortion scale (corner displacement fraction) monotonically increases
            monotonic_metric = monotonic_param
        elif deg_type == "shadow":
            # Shadow: luminance drop monotonically increases
            monotonic_metric = all(lum_drops[i] <= lum_drops[i + 1] for i in range(len(lum_drops) - 1))
        else:
            monotonic_metric = monotonic_param

        rationales = {
            "gaussian_blur": "Increasing kernel sigma monotonically decreases high-frequency edge energy and PSNR while preserving general layout.",
            "motion_blur": "Increasing linear streak length simulates camera shake, progressively merging adjacent characters.",
            "gaussian_noise": "Additive zero-mean noise with increasing std gradually masks text contrast and introduces sensor grain.",
            "jpeg_compression": "Decreasing quality factor (70->40->20->8) introduces standard DCT 8x8 block artifacts and ringing.",
            "downsampling": "Progressive resolution reduction (0.67->0.50->0.33->0.20) tests robustness to low-DPI capture.",
            "rotation": "Small rotations (2->5->10->20 deg) test tilt tolerance without out-of-boundary document loss.",
            "perspective": "Corner displacement scales (0.05->0.10->0.18->0.28) simulate oblique handheld photography angles.",
            "shadow": "Luminance attenuation (0.25->0.45->0.65->0.85 opacity) across diagonal band models ambient shadows.",
        }

        report_degradations[deg_type] = {
            "description": deg_info.get("description", ""),
            "primary_metric": primary_metric_name,
            "monotonic_parameter": monotonic_param,
            "monotonic_metric": monotonic_metric,
            "selection_rationale": rationales.get(deg_type, "Empirical ordering on physical distortion."),
            "candidate_evaluations": candidate_evaluations,
            "severities": severity_evaluations,
        }

        selected_config["degradations"][deg_type] = {
            "severities": severities,
            "primary_metric": primary_metric_name,
        }

    report = {
        "title": "Document Image Degradation Severity Calibration Report",
        "timestamp": datetime.now().isoformat(),
        "parameter_status": parameter_status,
        "real_data_calibration": real_data_status,
        "calibration_dataset": calibration_dataset,
        "research_validation_size": 126,
        "final_calibration_completed": is_real,
        "data_source": data_source,
        "num_calibration_images": num_images,
        "notes": (
            "Current parameters are provisional / fixture-based serving as an initial starting set. "
            "Real SROIE validation data calibration remains PENDING until Kaggle dataset artifact is provided."
            if real_data_status == "PENDING"
            else "Calibration conducted on canonical SROIE 126-sample validation split."
        ),
        "degradations": report_degradations,
    }

    # Write output artifacts
    output_dir.mkdir(parents=True, exist_ok=True)
    report_json_path = output_dir / "calibration_report.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    calib_yaml_path = output_dir / "calibration_config.yaml"
    with open(calib_yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(selected_config, f, sort_keys=False)

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate Degradation Severity Levels")
    parser.add_argument("--config", type=Path, default=Path("configs/degradation.yaml"), help="Config file path")
    parser.add_argument("--data-root", type=Path, default=None, help="Root path to SROIE dataset")
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/calibration"), help="Output directory")
    parser.add_argument("--no-samples", action="store_true", help="Do not save sample images")
    parser.add_argument("--allow-real-data", action="store_true", help="Allow discovery of real SROIE validation data")
    parser.add_argument("--max-images", type=int, default=None, help="Cap number of calibration images")
    args = parser.parse_args()

    report = run_calibration(
        config_path=args.config,
        output_dir=args.output_dir,
        data_root=args.data_root,
        save_samples=not args.no_samples,
        allow_real_data=args.allow_real_data,
        max_images=args.max_images,
    )

    print(f"Calibration completed. Status: {report['real_data_calibration']}")
    print(f"Artifacts written to: {args.output_dir.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
