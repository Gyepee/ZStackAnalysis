"""Compare legacy averaging with within-plane rigid XY registration.

This workflow is intentionally limited to framewise, within-plane 2-D rigid
registration followed by quality-controlled aggregation.  It does not estimate
or correct axial motion, and it never modifies the source ScanImage TIFF.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import ndimage
from skimage.registration import phase_cross_correlation
import skimage
import tifffile
from tifffile import FileHandle, read_scanimage_metadata

from .bench2p_zstack import (
    DEFAULT_STYLE,
    REPO_ROOT,
    StackMetadata,
    collect_environment,
    configure_matplotlib,
    discover_stack_tif,
    git_state,
    normalized_correlation,
    parse_stack_metadata,
    read_json,
    repo_relative,
    sha256_file,
    stable_digest,
    write_csv,
    write_json,
    write_ome_volume,
)


MODULE_PATH = Path("src/labgraph_ops/workflows/bench2p_zstack_xy_registration.py")
RUNNER_PATH = Path("scripts/build_bench2p_zstack_xy_registration.py")
HELPER_PATH = Path("src/labgraph_ops/workflows/bench2p_zstack.py")
DEFAULT_CONFIG = REPO_ROOT / "config" / "bench2p_zstack_xy_registration.json"
SCOPE_STATEMENT = "within-plane 2D registration only; no axial-motion correction"

METHOD_A = "A_unregistered_all_mean"
METHOD_B = "B_legacy_correlation_filtered_unregistered_mean"
METHOD_C = "C_xy_registered_all_mean"
METHOD_D = "D_xy_registered_quality_filtered_mean"
METHODS = (METHOD_A, METHOD_B, METHOD_C, METHOD_D)


@dataclass
class StackProcessingResult:
    volumes: dict[str, np.ndarray]
    frame_rows: list[dict[str, Any]]
    plane_rows: list[dict[str, Any]]
    z_decay_rows: list[dict[str, Any]]
    volume_rows: list[dict[str, Any]]
    saturation_rows: list[dict[str, Any]]
    saturation_histogram_rows: list[dict[str, Any]]
    representative_raw_frame: np.ndarray
    page_count: int


def z_geometry_qc(
    z_positions_um: Sequence[float],
    actual_step_um: float,
    *,
    absolute_tolerance_um: float,
    require_increasing: bool,
) -> dict[str, Any]:
    """Validate monotonic, regular ScanImage Z coordinates against metadata."""
    z = np.asarray(z_positions_um, dtype=float)
    if z.ndim != 1 or z.size < 2 or not np.isfinite(z).all():
        raise ValueError("Z positions must be a finite one-dimensional sequence")
    if not np.isfinite(actual_step_um) or actual_step_um <= 0:
        raise ValueError(f"actualStackZStepSize must be positive, got {actual_step_um}")
    differences = np.diff(z)
    increasing = bool(np.all(differences > 0))
    decreasing = bool(np.all(differences < 0))
    monotonic = increasing or decreasing
    signed_expected = actual_step_um if increasing else -actual_step_um
    residuals = differences - signed_expected
    median_spacing = float(np.median(differences))
    maximum_residual = float(np.max(np.abs(residuals)))
    regular = bool(maximum_residual <= absolute_tolerance_um)
    status = "pass"
    reasons: list[str] = []
    if not monotonic or (require_increasing and not increasing):
        status = "fail"
        reasons.append("z_positions_not_strictly_increasing")
    if not regular:
        status = "fail"
        reasons.append("z_spacing_irregular_or_mismatched")
    return {
        "status": status,
        "reasons": reasons,
        "actual_stack_z_step_um": float(actual_step_um),
        "median_diff_zs_um": median_spacing,
        "minimum_diff_zs_um": float(np.min(differences)),
        "maximum_diff_zs_um": float(np.max(differences)),
        "maximum_absolute_spacing_residual_um": maximum_residual,
        "absolute_tolerance_um": float(absolute_tolerance_um),
        "strictly_monotonic": monotonic,
        "strictly_increasing": increasing,
        "regular_spacing": regular,
        "plane_count": int(z.size),
    }


def page_z_grouping_qc(
    raw_zs: Sequence[float],
    *,
    n_slices: int,
    frames_per_slice: int,
    absolute_tolerance_um: float,
) -> dict[str, Any]:
    """Check that ScanImage page-level Z values form plane-major blocks."""
    values = np.asarray(raw_zs, dtype=float).reshape(-1)
    expected = int(n_slices * frames_per_slice)
    if values.size != expected:
        return {
            "status": "fail",
            "reason": "zs_length_does_not_match_expected_pages",
            "observed_zs_count": int(values.size),
            "expected_zs_count": expected,
        }
    blocks = values.reshape(n_slices, frames_per_slice)
    maximum_within_plane_deviation = float(
        np.max(np.abs(blocks - blocks[:, :1]))
    )
    constant = maximum_within_plane_deviation <= absolute_tolerance_um
    return {
        "status": "pass" if constant else "fail",
        "reason": None if constant else "z_changes_within_plane_frame_block",
        "observed_zs_count": int(values.size),
        "expected_zs_count": expected,
        "plane_count": int(n_slices),
        "frames_per_plane": int(frames_per_slice),
        "maximum_within_plane_z_deviation_um": maximum_within_plane_deviation,
        "plane_major_contiguous_frame_blocks": bool(constant),
        "block_z_positions_um": blocks[:, 0].tolist(),
    }


def read_page_z_grouping_qc(
    path: Path,
    metadata: StackMetadata,
    *,
    absolute_tolerance_um: float,
) -> dict[str, Any]:
    """Read ScanImage static metadata without decoding image pages."""
    with FileHandle(path) as handle:
        frame_data, _, version = read_scanimage_metadata(handle)
    result = page_z_grouping_qc(
        frame_data.get("SI.hStackManager.zs", []),
        n_slices=metadata.n_slices,
        frames_per_slice=metadata.frames_per_slice,
        absolute_tolerance_um=absolute_tolerance_um,
    )
    result["scanimage_bigtiff_metadata_version"] = int(version)
    return result


def maximum_contiguous_true(values: Sequence[bool]) -> int:
    longest = 0
    current = 0
    for value in values:
        current = current + 1 if bool(value) else 0
        longest = max(longest, current)
    return longest


def _robust_lower_threshold(values: np.ndarray, mad_k: float) -> float:
    finite = values[np.isfinite(values)]
    if not finite.size:
        return float("inf")
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    return median - float(mad_k) * 1.4826 * mad


def _robust_upper_threshold(values: np.ndarray, mad_k: float) -> float:
    finite = values[np.isfinite(values)]
    if not finite.size:
        return float("-inf")
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    return median + float(mad_k) * 1.4826 * mad


def legacy_correlation_mask(
    correlations: np.ndarray, *, mad_k: float, min_keep_fraction: float
) -> tuple[np.ndarray, float]:
    """Reproduce the existing unregistered correlation-MAD frame filter."""
    correlations = np.asarray(correlations, dtype=float)
    threshold = _robust_lower_threshold(correlations, mad_k)
    keep = np.isfinite(correlations) & (correlations >= threshold)
    minimum = max(1, int(math.ceil(correlations.size * min_keep_fraction)))
    if int(keep.sum()) < minimum:
        order = np.argsort(np.nan_to_num(correlations, nan=-np.inf))[::-1]
        keep[:] = False
        keep[order[:minimum]] = True
    return keep, float(threshold)


def _registration_image(image: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    result = np.asarray(image, dtype=np.float32)
    sigma = float(config["gaussian_sigma_px"])
    if sigma > 0:
        result = ndimage.gaussian_filter(result, sigma=sigma, mode="nearest")
    border = int(config["crop_border_px"])
    if border:
        if min(result.shape) <= 2 * border + 8:
            raise ValueError("registration crop border leaves too little image")
        result = result[border:-border, border:-border]
    if bool(config.get("remove_median_and_scale_mad", True)):
        result = result - float(np.median(result))
        mad = float(np.median(np.abs(result)))
        if mad > 0:
            result = result / (1.4826 * mad)
    if bool(config.get("hann_window", True)):
        window = np.outer(np.hanning(result.shape[0]), np.hanning(result.shape[1]))
        result = result * window.astype(np.float32)
    return np.asarray(result, dtype=np.float32)


def estimate_frame_shifts(
    reference: np.ndarray,
    frames: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Return shifts to apply to frames and phase-correlation error values."""
    reference_for_registration = _registration_image(reference, config)
    shifts = np.empty((frames.shape[0], 2), dtype=np.float32)
    errors = np.empty(frames.shape[0], dtype=np.float32)
    normalization = config.get("normalization")
    for frame_index, frame in enumerate(frames):
        moving = _registration_image(frame, config)
        shift, error, _ = phase_cross_correlation(
            reference_for_registration,
            moving,
            upsample_factor=int(config["upsample_factor"]),
            normalization=normalization,
        )
        shifts[frame_index] = np.asarray(shift[:2], dtype=np.float32)
        errors[frame_index] = float(error)
    return shifts, errors


def apply_frame_shifts(
    frames: np.ndarray, shifts: np.ndarray, *, interpolation_order: int
) -> np.ndarray:
    registered = np.empty(frames.shape, dtype=np.float32)
    for index, (frame, shift) in enumerate(zip(frames, shifts, strict=True)):
        registered[index] = ndimage.shift(
            np.asarray(frame, dtype=np.float32),
            shift=tuple(float(value) for value in shift),
            order=interpolation_order,
            mode="constant",
            cval=np.nan,
            prefilter=interpolation_order > 1,
        )
    return registered


def normalized_tenengrad(image: np.ndarray, spacing: Sequence[float] | None = None) -> float:
    """Scale-normalized gradient energy; useful only for within-input comparison."""
    image = np.asarray(image, dtype=np.float64)
    finite = np.isfinite(image)
    if int(finite.sum()) < 4:
        return float("nan")
    fill = float(np.nanmedian(image))
    image = np.nan_to_num(image, nan=fill)
    gradients = np.gradient(image, *(spacing or [1.0] * image.ndim))
    energy = float(np.mean(sum(component * component for component in gradients)))
    variance = float(np.var(image))
    return energy / variance if variance > 0 else float("nan")


def _safe_nanmean(images: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    selected = images if mask is None else images[mask]
    if selected.shape[0] == 0:
        raise ValueError("No frames remain for aggregation")
    valid_count = np.sum(np.isfinite(selected), axis=0)
    total = np.nansum(selected, axis=0, dtype=np.float64)
    result = np.divide(total, valid_count, where=valid_count > 0)
    result[valid_count == 0] = np.nan
    return result.astype(np.float32)


def process_plane_methods(
    frames: np.ndarray,
    *,
    pixel_size_x_um: float,
    pixel_size_y_um: float,
    config: dict[str, Any],
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], dict[str, Any]]:
    """Construct A-D plane images and framewise registration/QC records."""
    frames = np.asarray(frames, dtype=np.float32)
    if frames.ndim != 3 or frames.shape[0] < 2:
        raise ValueError("Expected frame × Y × X data with at least two frames")
    initial_reference = np.median(frames, axis=0).astype(np.float32)
    correlations_before = np.asarray(
        [normalized_correlation(initial_reference, frame) for frame in frames], dtype=float
    )
    legacy = config["legacy_filter"]
    legacy_keep, legacy_threshold = legacy_correlation_mask(
        correlations_before,
        mad_k=float(legacy["correlation_mad_k"]),
        min_keep_fraction=float(legacy["min_keep_fraction"]),
    )

    registration = config["registration"]
    shifts, errors = estimate_frame_shifts(initial_reference, frames, registration)
    registered = apply_frame_shifts(
        frames, shifts, interpolation_order=int(registration["interpolation_order"])
    )
    reference = _safe_nanmean(registered)
    for _ in range(int(registration["reference_updates"])):
        shifts, errors = estimate_frame_shifts(reference, frames, registration)
        registered = apply_frame_shifts(
            frames, shifts, interpolation_order=int(registration["interpolation_order"])
        )
        reference = _safe_nanmean(registered)

    correlations_after = np.asarray(
        [normalized_correlation(reference, frame) for frame in registered], dtype=float
    )
    correlation_gain = correlations_after - correlations_before
    magnitudes = np.hypot(shifts[:, 0], shifts[:, 1])
    quality = config["post_registration_quality"]
    max_shift = float(registration["max_shift_px"])
    geometric_valid = np.isfinite(magnitudes) & (magnitudes <= max_shift)
    correlation_threshold = (
        float(quality["minimum_correlation"])
        if quality.get("minimum_correlation") is not None
        else _robust_lower_threshold(
            correlations_after[geometric_valid], float(quality["correlation_mad_k"])
        )
    )
    error_threshold = (
        float(quality["maximum_registration_error"])
        if quality.get("maximum_registration_error") is not None
        else _robust_upper_threshold(
            errors[geometric_valid], float(quality["registration_error_mad_k"])
        )
    )
    keep = (
        geometric_valid
        & np.isfinite(correlations_after)
        & (correlations_after >= correlation_threshold)
        & np.isfinite(errors)
        & (errors <= error_threshold)
        & (correlation_gain >= float(quality["minimum_correlation_gain"]))
    )
    minimum = max(1, int(math.ceil(frames.shape[0] * float(quality["min_keep_fraction"]))))
    keep_floor_relaxed = False
    if int(keep.sum()) < minimum:
        candidates = np.flatnonzero(geometric_valid & np.isfinite(correlations_after))
        order = candidates[np.argsort(correlations_after[candidates])[::-1]]
        keep[:] = False
        keep[order[: min(minimum, order.size)]] = True
        keep_floor_relaxed = True
    if not keep.any():
        raise ValueError("No geometrically valid registered frames remain")

    frame_rows: list[dict[str, Any]] = []
    for index in range(frames.shape[0]):
        reasons = []
        if magnitudes[index] > max_shift:
            reasons.append("shift_exceeds_max")
        if correlations_after[index] < correlation_threshold:
            reasons.append("post_correlation_low")
        if errors[index] > error_threshold:
            reasons.append("registration_error_high")
        if correlation_gain[index] < float(quality["minimum_correlation_gain"]):
            reasons.append("correlation_gain_low")
        frame_rows.append(
            {
                "local_frame_index": index,
                "dy_px": float(shifts[index, 0]),
                "dx_px": float(shifts[index, 1]),
                "dy_um": float(shifts[index, 0] * pixel_size_y_um),
                "dx_um": float(shifts[index, 1] * pixel_size_x_um),
                "shift_magnitude_px": float(magnitudes[index]),
                "shift_magnitude_um": float(
                    np.hypot(
                        shifts[index, 0] * pixel_size_y_um,
                        shifts[index, 1] * pixel_size_x_um,
                    )
                ),
                "registration_error": float(errors[index]),
                "reference_correlation_before": float(correlations_before[index]),
                "reference_correlation_after": float(correlations_after[index]),
                "correlation_gain": float(correlation_gain[index]),
                "legacy_retained": bool(legacy_keep[index]),
                "registered_quality_retained": bool(keep[index]),
                "registered_rejection_reasons": ";".join(reasons),
            }
        )

    images = {
        METHOD_A: np.mean(frames, axis=0, dtype=np.float64).astype(np.float32),
        METHOD_B: np.mean(frames[legacy_keep], axis=0, dtype=np.float64).astype(np.float32),
        METHOD_C: _safe_nanmean(registered),
        METHOD_D: _safe_nanmean(registered, keep),
    }
    summary = {
        "frames_total": int(frames.shape[0]),
        "legacy_frames_retained": int(legacy_keep.sum()),
        "legacy_retained_fraction": float(legacy_keep.mean()),
        "legacy_correlation_threshold": legacy_threshold,
        "registered_frames_retained": int(keep.sum()),
        "registered_retained_fraction": float(keep.mean()),
        "registered_keep_floor_relaxed": keep_floor_relaxed,
        "registered_correlation_threshold": float(correlation_threshold),
        "registered_error_threshold": float(error_threshold),
        "shift_magnitude_px_p50": float(np.median(magnitudes)),
        "shift_magnitude_px_p95": float(np.percentile(magnitudes, 95)),
        "shift_magnitude_px_max": float(np.max(magnitudes)),
        "shift_magnitude_um_p95": float(
            np.percentile([row["shift_magnitude_um"] for row in frame_rows], 95)
        ),
        "correlation_before_mean": float(np.mean(correlations_before)),
        "correlation_after_mean": float(np.mean(correlations_after)),
        "correlation_gain_mean": float(np.mean(correlation_gain)),
        "registration_error_mean": float(np.mean(errors)),
        "registration_error_p95": float(np.percentile(errors, 95)),
        "maximum_contiguous_legacy_rejected_frames": maximum_contiguous_true(~legacy_keep),
        "maximum_contiguous_registered_rejected_frames": maximum_contiguous_true(~keep),
    }
    for method, image in images.items():
        summary[f"sharpness_{method}"] = normalized_tenengrad(
            image, (pixel_size_y_um, pixel_size_x_um)
        )
    return images, frame_rows, summary


def _histogram_quantile(histogram: np.ndarray, percentile: float) -> int:
    cumulative = np.cumsum(histogram)
    if not cumulative.size or cumulative[-1] <= 0:
        raise ValueError("Empty intensity histogram")
    target = float(percentile) * float(cumulative[-1]) / 100.0
    return int(np.searchsorted(cumulative, target, side="left") - 32768)


def _normalized_saturation_offsets(fractions: Sequence[float], frame_count: int) -> list[int]:
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    offsets = []
    for fraction in fractions:
        value = float(fraction)
        if not 0 <= value <= 1:
            raise ValueError("normalized saturation frame positions must be in [0, 1]")
        offsets.append(int(round(value * (frame_count - 1))))
    return list(dict.fromkeys(offsets))


def _z_decay_rows(
    volumes: dict[str, np.ndarray], *, z_step_um: float, maximum_lag: int
) -> list[dict[str, Any]]:
    rows = []
    for method, volume in volumes.items():
        for lag in range(1, min(maximum_lag, volume.shape[0] - 1) + 1):
            correlations = np.asarray(
                [
                    normalized_correlation(volume[index], volume[index + lag])
                    for index in range(volume.shape[0] - lag)
                ],
                dtype=float,
            )
            rows.append(
                {
                    "method": method,
                    "lag_planes": lag,
                    "lag_um": float(lag * z_step_um),
                    "pair_count": int(correlations.size),
                    "correlation_mean": float(np.nanmean(correlations)),
                    "correlation_median": float(np.nanmedian(correlations)),
                    "correlation_p05": float(np.nanpercentile(correlations, 5)),
                    "correlation_p95": float(np.nanpercentile(correlations, 95)),
                }
            )
    return rows


def _volume_rows(
    volumes: dict[str, np.ndarray], metadata: StackMetadata, config: dict[str, Any]
) -> list[dict[str, Any]]:
    metric_config = config["volume_metrics"]
    y_index = int(round(float(metric_config["representative_y_fraction"]) * (metadata.height_px - 1)))
    x_index = int(round(float(metric_config["representative_x_fraction"]) * (metadata.width_px - 1)))
    rows = []
    for method, volume in volumes.items():
        adjacent = np.asarray(
            [normalized_correlation(volume[index], volume[index + 1]) for index in range(volume.shape[0] - 1)]
        )
        xz = volume[:, y_index, :]
        yz = volume[:, :, x_index]
        rows.append(
            {
                "method": method,
                "adjacent_plane_correlation_mean": float(np.nanmean(adjacent)),
                "adjacent_plane_correlation_median": float(np.nanmedian(adjacent)),
                "xz_single_slice_normalized_tenengrad": normalized_tenengrad(
                    xz, (metadata.z_step_um, metadata.pixel_size_x_um)
                ),
                "yz_single_slice_normalized_tenengrad": normalized_tenengrad(
                    yz, (metadata.z_step_um, metadata.pixel_size_y_um)
                ),
                "axial_mip_normalized_tenengrad": normalized_tenengrad(
                    np.nanmax(volume, axis=0),
                    (metadata.pixel_size_y_um, metadata.pixel_size_x_um),
                ),
                "mean_intensity": float(np.nanmean(volume)),
                "std_intensity": float(np.nanstd(volume)),
                "finite_fraction": float(np.isfinite(volume).mean()),
            }
        )
    return rows


def process_stack(
    metadata: StackMetadata,
    read_path: Path,
    config: dict[str, Any],
) -> StackProcessingResult:
    """Read one stack once and build A-D reconstructions plus QC."""
    discard = int(config["discard_initial_frames_per_slice"])
    usable_count = metadata.frames_per_slice - discard
    if usable_count < 2:
        raise ValueError("At least two usable frames per plane are required")
    volumes = {
        method: np.empty(
            (metadata.n_slices, metadata.height_px, metadata.width_px), dtype=np.float32
        )
        for method in METHODS
    }
    frame_rows: list[dict[str, Any]] = []
    plane_rows: list[dict[str, Any]] = []
    histogram = np.zeros(65536, dtype=np.int64)
    saturation_config = config["saturation_qc"]
    saturation_offsets = _normalized_saturation_offsets(
        saturation_config["normalized_frame_positions_per_slice"], usable_count
    )
    representative_plane = int(
        round(
            float(config["volume_metrics"]["representative_z_fraction"])
            * (metadata.n_slices - 1)
        )
    )
    representative_raw: np.ndarray | None = None

    with tifffile.TiffFile(read_path) as tif:
        page_count = len(tif.pages)
        if page_count != metadata.expected_pages:
            raise ValueError(
                f"TIFF has {page_count} pages; metadata expects {metadata.expected_pages}"
            )
        frame_buffer = np.empty(
            (metadata.frames_per_slice, metadata.height_px, metadata.width_px),
            dtype=tif.pages[0].dtype,
        )
        for plane_index in range(metadata.n_slices):
            first_page = plane_index * metadata.frames_per_slice
            for local_index in range(metadata.frames_per_slice):
                frame_buffer[local_index] = tif.pages[first_page + local_index].asarray()
            raw_usable = frame_buffer[discard:].copy()
            if representative_raw is None and plane_index == representative_plane:
                representative_raw = raw_usable[raw_usable.shape[0] // 2].astype(np.float32)
            for local_index in saturation_offsets:
                values = raw_usable[local_index].astype(np.int32).ravel() + 32768
                histogram += np.bincount(values, minlength=65536)

            images, local_frame_rows, plane_summary = process_plane_methods(
                raw_usable,
                pixel_size_x_um=metadata.pixel_size_x_um,
                pixel_size_y_um=metadata.pixel_size_y_um,
                config=config,
            )
            for method, image in images.items():
                volumes[method][plane_index] = image
            z_scanimage = float(metadata.z_positions_um[plane_index])
            z_relative = z_scanimage - float(metadata.z_positions_um[0])
            for row in local_frame_rows:
                local_index = int(row["local_frame_index"]) + discard
                page_index = first_page + local_index
                row.update(
                    {
                        "animal_id": metadata.animal_id,
                        "date": metadata.date,
                        "scan_id": metadata.scan_id,
                        "session_id": metadata.session_id,
                        "plane_index": plane_index,
                        "z_relative_um": z_relative,
                        "z_scanimage_um": z_scanimage,
                        "source_page_index": page_index,
                        "local_frame_index": local_index,
                        "frame_time_s": float(page_index / metadata.frame_rate_hz),
                    }
                )
                frame_rows.append(row)
            plane_summary.update(
                {
                    "animal_id": metadata.animal_id,
                    "date": metadata.date,
                    "scan_id": metadata.scan_id,
                    "session_id": metadata.session_id,
                    "plane_index": plane_index,
                    "z_relative_um": z_relative,
                    "z_scanimage_um": z_scanimage,
                    "plane_start_time_s": float(first_page / metadata.frame_rate_hz),
                    "plane_end_time_s": float(
                        (first_page + metadata.frames_per_slice - 1) / metadata.frame_rate_hz
                    ),
                    "plane_acquisition_duration_s": float(
                        metadata.frames_per_slice / metadata.frame_rate_hz
                    ),
                    "raw_plane_p99_9_intensity": float(np.percentile(raw_usable, 99.9)),
                    "raw_plane_max_intensity": int(np.max(raw_usable)),
                }
            )
            plane_rows.append(plane_summary)
            if plane_index == 0 or (plane_index + 1) % 5 == 0 or plane_index + 1 == metadata.n_slices:
                print(
                    f"[{metadata.scan_id}] registered plane {plane_index + 1}/{metadata.n_slices}",
                    flush=True,
                )

    if representative_raw is None:
        raise RuntimeError("Representative raw frame was not captured")
    pixel_count = int(histogram.sum())
    nonzero = np.flatnonzero(histogram)
    upper_percentile = float(saturation_config["upper_tail_percentile"])
    upper_value = _histogram_quantile(histogram, upper_percentile)
    upper_index = upper_value + 32768
    upper_count = int(histogram[upper_index:].sum())
    upper_mode_count = int(histogram[upper_index:].max())
    upper_mode_index = int(np.argmax(histogram[upper_index:]) + upper_index)
    upper_mode_fraction = float(upper_mode_count / upper_count) if upper_count else 0.0
    saturation_rows = [
        {
            "scan_id": metadata.scan_id,
            "sampling_rule": "normalized_frame_positions_per_slice",
            "local_frame_indices_after_discard": ";".join(map(str, saturation_offsets)),
            "sampled_frames": int(metadata.n_slices * len(saturation_offsets)),
            "sampled_pixels": pixel_count,
            "observed_min": int(nonzero[0] - 32768),
            "observed_max": int(nonzero[-1] - 32768),
            "p99": _histogram_quantile(histogram, 99.0),
            "p99_9": _histogram_quantile(histogram, 99.9),
            "p99_99": _histogram_quantile(histogram, 99.99),
            "upper_tail_percentile": upper_percentile,
            "upper_tail_threshold": upper_value,
            "upper_tail_pixel_fraction": float(upper_count / pixel_count),
            "upper_tail_mode_value": int(upper_mode_index - 32768),
            "upper_tail_mode_fraction": upper_mode_fraction,
            "fraction_at_observed_max": float(histogram[nonzero[-1]] / pixel_count),
            "fraction_at_negative_int16_rail": float(histogram[0] / pixel_count),
            "fraction_at_positive_int16_rail": float(histogram[-1] / pixel_count),
            "plateau_review_threshold": float(
                saturation_config["plateau_mode_fraction_review_threshold"]
            ),
            "plateau_like_upper_tail": bool(
                upper_mode_fraction
                >= float(saturation_config["plateau_mode_fraction_review_threshold"])
            ),
        }
    ]
    saturation_histogram_rows = [
        {"intensity": int(index - 32768), "count": int(histogram[index])}
        for index in nonzero
    ]
    return StackProcessingResult(
        volumes=volumes,
        frame_rows=frame_rows,
        plane_rows=plane_rows,
        z_decay_rows=_z_decay_rows(
            volumes,
            z_step_um=metadata.z_step_um,
            maximum_lag=int(config["volume_metrics"]["z_correlation_max_lag_planes"]),
        ),
        volume_rows=_volume_rows(volumes, metadata, config),
        saturation_rows=saturation_rows,
        saturation_histogram_rows=saturation_histogram_rows,
        representative_raw_frame=representative_raw,
        page_count=page_count,
    )


def _display_transform(
    image: np.ndarray, low: float, high: float, gamma: float
) -> np.ndarray:
    normalized = np.nan_to_num((np.asarray(image, dtype=np.float32) - low) / (high - low), nan=0.0)
    return np.power(np.clip(normalized, 0.0, 1.0), gamma, dtype=np.float32)


def _thin_half_width(thickness_um: float, spacing_um: float) -> int:
    count = max(1, int(round(thickness_um / spacing_um)))
    if count % 2 == 0:
        count += 1
    return count // 2


def save_comparison_figure(
    run_dir: Path,
    result: StackProcessingResult,
    metadata: StackMetadata,
    config: dict[str, Any],
    style: dict[str, Any],
) -> tuple[dict[str, Path], list[dict[str, Any]], dict[str, float]]:
    figure_id = "fig_bench2p_xy_registration_method_comparison"
    figure_dir = run_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    configure_matplotlib(style, str(config["figure_profile"]))
    display = config["display"]
    reference = result.volumes[str(display["reference_method"])]
    finite = reference[np.isfinite(reference)]
    low, high = np.percentile(
        finite, [float(display["percentile_low"]), float(display["percentile_high"])]
    )
    gamma = float(display["gamma"])
    display_volumes = {
        method: _display_transform(volume, float(low), float(high), gamma)
        for method, volume in result.volumes.items()
    }
    raw_display = _display_transform(result.representative_raw_frame, float(low), float(high), gamma)
    metrics = config["volume_metrics"]
    z_index = int(round(float(metrics["representative_z_fraction"]) * (metadata.n_slices - 1)))
    y_index = int(round(float(metrics["representative_y_fraction"]) * (metadata.height_px - 1)))
    x_index = int(round(float(metrics["representative_x_fraction"]) * (metadata.width_px - 1)))
    half_y = _thin_half_width(float(metrics["thin_slab_thickness_um"]), metadata.pixel_size_y_um)
    half_x = _thin_half_width(float(metrics["thin_slab_thickness_um"]), metadata.pixel_size_x_um)
    y_slice = slice(max(0, y_index - half_y), min(metadata.height_px, y_index + half_y + 1))
    x_slice = slice(max(0, x_index - half_x), min(metadata.width_px, x_index + half_x + 1))
    column_names = ["representative raw frame", *METHODS]
    row_names = [
        "representative XY plane",
        "single-voxel XZ slice",
        "single-voxel YZ slice",
        "thin-slab XZ MIP",
        "thin-slab YZ MIP",
        "full-volume axial MIP",
    ]
    arrays: dict[str, np.ndarray] = {"raw_representative_xy": raw_display}
    panel_rows: list[dict[str, Any]] = []
    fig, axes = plt.subplots(6, 5, figsize=(13.0, 13.5), constrained_layout=True)
    for row_index, row_name in enumerate(row_names):
        axes[row_index, 0].set_axis_off()
        if row_index == 0:
            axes[row_index, 0].imshow(raw_display, cmap="gray", vmin=0, vmax=1)
            axes[row_index, 0].set_title(column_names[0])
            panel_rows.append(
                {
                    "panel_row": row_name,
                    "method": "raw_single_frame",
                    "computation": "one raw frame from the representative Z plane",
                }
            )
        for column_index, method in enumerate(METHODS, start=1):
            volume = display_volumes[method]
            if row_index == 0:
                image = volume[z_index]
                extent = None
                computation = "registered or unregistered plane aggregation"
            elif row_index == 1:
                image = volume[:, y_index, :]
                extent = (0, metadata.fov_x_um, metadata.n_slices * metadata.z_step_um, 0)
                computation = "single-voxel-thick XZ orthogonal slice"
            elif row_index == 2:
                image = volume[:, :, x_index]
                extent = (0, metadata.fov_y_um, metadata.n_slices * metadata.z_step_um, 0)
                computation = "single-voxel-thick YZ orthogonal slice"
            elif row_index == 3:
                image = np.max(volume[:, y_slice, :], axis=1)
                extent = (0, metadata.fov_x_um, metadata.n_slices * metadata.z_step_um, 0)
                computation = f"XZ maximum over {y_slice.stop-y_slice.start} Y voxels"
            elif row_index == 4:
                image = np.max(volume[:, :, x_slice], axis=2)
                extent = (0, metadata.fov_y_um, metadata.n_slices * metadata.z_step_um, 0)
                computation = f"YZ maximum over {x_slice.stop-x_slice.start} X voxels"
            else:
                image = np.max(volume, axis=0)
                extent = None
                computation = "maximum intensity over the full Z volume"
            key = f"{method}__{row_name.replace(' ', '_').replace('-', '_')}"
            arrays[key] = np.asarray(image, dtype=np.float32)
            axis = axes[row_index, column_index]
            axis.imshow(
                image,
                cmap="gray",
                vmin=0,
                vmax=1,
                extent=extent,
                aspect="equal" if extent is not None else "auto",
                interpolation="nearest",
            )
            if row_index == 0:
                axis.set_title(method.split("_", 1)[0])
            if column_index == 1:
                axis.set_ylabel(row_name)
            axis.set_xticks([])
            axis.set_yticks([])
            panel_rows.append(
                {
                    "panel_row": row_name,
                    "method": method,
                    "computation": computation,
                    "z_plane_index": z_index,
                    "z_relative_um": float(z_index * metadata.z_step_um),
                    "y_index": y_index,
                    "x_index": x_index,
                    "display_low_intensity": float(low),
                    "display_high_intensity": float(high),
                    "display_gamma": gamma,
                }
            )
    fig.suptitle(f"{metadata.scan_id}: A-D reconstruction with one shared display transform")
    svg = figure_dir / f"{figure_id}.svg"
    png = figure_dir / f"{figure_id}.png"
    arrays_path = figure_dir / f"{figure_id}_panel_arrays.npz"
    fig.savefig(svg, bbox_inches="tight", facecolor="white")
    fig.savefig(png, dpi=style["output"]["png_dpi"], bbox_inches="tight", facecolor="white")
    plt.close(fig)
    np.savez_compressed(arrays_path, **arrays)
    return (
        {"svg": svg, "png": png, "panel_arrays": arrays_path},
        panel_rows,
        {"low_intensity": float(low), "high_intensity": float(high), "gamma": gamma},
    )


def save_trace_figure(
    run_dir: Path,
    frame_rows: Sequence[dict[str, Any]],
    metadata: StackMetadata,
    style: dict[str, Any],
    profile_name: str,
) -> dict[str, Path]:
    figure_id = "fig_bench2p_xy_registration_trace"
    configure_matplotlib(style, profile_name)
    time = np.asarray([row["frame_time_s"] for row in frame_rows], dtype=float)
    dx = np.asarray([row["dx_um"] for row in frame_rows], dtype=float)
    dy = np.asarray([row["dy_um"] for row in frame_rows], dtype=float)
    magnitude = np.asarray([row["shift_magnitude_um"] for row in frame_rows], dtype=float)
    before = np.asarray([row["reference_correlation_before"] for row in frame_rows], dtype=float)
    after = np.asarray([row["reference_correlation_after"] for row in frame_rows], dtype=float)
    error = np.asarray([row["registration_error"] for row in frame_rows], dtype=float)
    retained = np.asarray([row["registered_quality_retained"] for row in frame_rows], dtype=bool)
    fig, axes = plt.subplots(4, 1, figsize=(7.0, 7.5), sharex=True, constrained_layout=True)
    axes[0].plot(time, dx, linewidth=0.45, label="dx")
    axes[0].plot(time, dy, linewidth=0.45, label="dy")
    axes[0].set_ylabel("shift (µm)")
    axes[0].legend(frameon=False, ncol=2)
    axes[1].plot(time, magnitude, color="#0072B2", linewidth=0.45)
    axes[1].scatter(time[~retained], magnitude[~retained], s=5, color="#D55E00", label="rejected")
    axes[1].set_ylabel("|shift| (µm)")
    axes[1].legend(frameon=False)
    axes[2].plot(time, before, color="0.55", linewidth=0.4, label="before")
    axes[2].plot(time, after, color="#009E73", linewidth=0.4, label="after")
    axes[2].set_ylabel("reference corr.")
    axes[2].legend(frameon=False, ncol=2)
    axes[3].plot(time, error, color="#CC79A7", linewidth=0.45)
    axes[3].set_ylabel("registration error")
    axes[3].set_xlabel("time from stack start (s)")
    fig.suptitle(f"{metadata.scan_id}: {SCOPE_STATEMENT}")
    figure_dir = run_dir / "figures"
    svg = figure_dir / f"{figure_id}.svg"
    png = figure_dir / f"{figure_id}.png"
    fig.savefig(svg, bbox_inches="tight", facecolor="white")
    fig.savefig(png, dpi=style["output"]["png_dpi"], bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {"svg": svg, "png": png}


def _write_rows(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError(f"Cannot write an empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _code_records() -> list[dict[str, str]]:
    return [
        {"path": str(path), "sha256": sha256_file(REPO_ROOT / path)}
        for path in (RUNNER_PATH, MODULE_PATH, HELPER_PATH)
    ]


def build_registration_run(
    *,
    data_root: Path,
    animal_id: str,
    date: str,
    scan_id: str,
    config_path: Path,
    style_path: Path,
    output_root: Path,
    read_copy: Path | None = None,
) -> Path:
    config = read_json(config_path)
    if config.get("scope_statement") != SCOPE_STATEMENT:
        raise ValueError(f"Config must declare: {SCOPE_STATEMENT}")
    style = read_json(style_path)
    source_path = discover_stack_tif(data_root, animal_id, date, scan_id)
    metadata = parse_stack_metadata(source_path)
    if metadata.setup != "bench2p" or not metadata.stack_enabled or metadata.stack_mode.lower() != "slow":
        raise ValueError("Source is not an enabled bench2p slow Z stack")
    metadata_qc = z_geometry_qc(
        metadata.z_positions_um,
        metadata.z_step_um,
        absolute_tolerance_um=float(config["metadata_qc"]["z_spacing_absolute_tolerance_um"]),
        require_increasing=bool(config["metadata_qc"]["require_increasing_z"]),
    )
    if metadata_qc["status"] != "pass":
        raise ValueError(f"Z geometry gate failed: {metadata_qc['reasons']}")
    page_grouping_qc = read_page_z_grouping_qc(
        source_path,
        metadata,
        absolute_tolerance_um=float(config["metadata_qc"]["z_spacing_absolute_tolerance_um"]),
    )
    if page_grouping_qc["status"] != "pass":
        raise ValueError(f"Page/Z grouping gate failed: {page_grouping_qc['reason']}")
    read_path = read_copy.resolve() if read_copy else source_path
    if read_copy:
        if read_path.stat().st_size != source_path.stat().st_size:
            raise ValueError("Local read copy size does not match the authoritative source")
        if read_path.name != source_path.name:
            raise ValueError("Local read copy filename does not match the source TIFF")

    identity = {
        "config": config,
        "source_path": str(source_path.resolve()),
        "source_size": source_path.stat().st_size,
        "source_mtime_ns": source_path.stat().st_mtime_ns,
    }
    created_at = datetime.now(timezone.utc)
    analysis_id = (
        f"bench2p_zstack_xyreg__{created_at.strftime('%Y%m%dT%H%M%SZ')}__"
        f"{stable_digest(identity)}"
    )
    final_dir = output_root.resolve() / analysis_id
    incomplete_dir = output_root.resolve() / f".{analysis_id}.incomplete"
    if final_dir.exists() or incomplete_dir.exists():
        raise FileExistsError(f"Run target already exists for {analysis_id}")
    for child in ("tables", "volumes", "figures", "logs"):
        (incomplete_dir / child).mkdir(parents=True, exist_ok=child != "tables")
    try:
        result = process_stack(metadata, read_path, config)
        volume_paths: dict[str, Path] = {}
        for method, volume in result.volumes.items():
            path = incomplete_dir / "volumes" / f"{metadata.scan_id}__{method}.ome.tif"
            write_ome_volume(
                path,
                volume,
                pixel_size_x_um=metadata.pixel_size_x_um,
                pixel_size_y_um=metadata.pixel_size_y_um,
                z_step_um=metadata.z_step_um,
            )
            volume_paths[method] = path
        _write_rows(incomplete_dir / "tables" / "frame_registration_qc.csv", result.frame_rows)
        _write_rows(incomplete_dir / "tables" / "plane_reconstruction_qc.csv", result.plane_rows)
        _write_rows(incomplete_dir / "tables" / "z_correlation_decay.csv", result.z_decay_rows)
        _write_rows(incomplete_dir / "tables" / "volume_method_comparison.csv", result.volume_rows)
        _write_rows(incomplete_dir / "tables" / "raw_saturation_summary.csv", result.saturation_rows)
        _write_rows(
            incomplete_dir / "tables" / "raw_saturation_histogram.csv",
            result.saturation_histogram_rows,
        )
        comparison_paths, comparison_rows, display_metadata = save_comparison_figure(
            incomplete_dir, result, metadata, config, style
        )
        _write_rows(
            incomplete_dir / "figures" / "fig_bench2p_xy_registration_method_comparison_plot_data.csv",
            comparison_rows,
        )
        trace_paths = save_trace_figure(
            incomplete_dir,
            result.frame_rows,
            metadata,
            style,
            str(config["figure_profile"]),
        )
        resolved_config = {
            **config,
            "analysis_id": analysis_id,
            "created_at_utc": created_at.isoformat(),
            "animal_id": animal_id,
            "date": date,
            "scan_id": scan_id,
            "authoritative_source_path": str(source_path.resolve()),
            "local_read_copy_used": bool(read_copy),
            "physical_spacing_zyx_um": [
                metadata.z_step_um,
                metadata.pixel_size_y_um,
                metadata.pixel_size_x_um,
            ],
            "display_resolved": display_metadata,
            "config_source": repo_relative(config_path),
            "figure_style_source": repo_relative(style_path),
        }
        write_json(incomplete_dir / "config.snapshot.json", resolved_config)
        write_json(incomplete_dir / "environment.json", collect_environment())
        shutil.move(str(incomplete_dir), str(final_dir))
        comparison_paths = {
            key: final_dir / path.relative_to(incomplete_dir)
            for key, path in comparison_paths.items()
        }
        trace_paths = {
            key: final_dir / path.relative_to(incomplete_dir)
            for key, path in trace_paths.items()
        }
    except BaseException:
        if incomplete_dir.exists():
            write_json(
                incomplete_dir / "logs" / "INCOMPLETE.json",
                {"analysis_id": analysis_id, "status": "incomplete", "scope": SCOPE_STATEMENT},
            )
        raise

    source_hash = sha256_file(final_dir / "volumes" / f"{metadata.scan_id}__{METHOD_A}.ome.tif")
    read_copy_hash = sha256_file(read_path)
    sources_payload = {
        "schema_version": "labgraph.bench2p_zstack_xyreg.sources.v1",
        "read_only": True,
        "scope_statement": SCOPE_STATEMENT,
        "authoritative_source": {
            **asdict(metadata),
            "z_positions_um": list(metadata.z_positions_um),
            "sha256": read_copy_hash,
            "sha256_computed_from_identical_local_read_copy": bool(read_copy),
        },
        "method_a_volume_sha256": source_hash,
    }
    write_json(final_dir / "sources.json", sources_payload)
    retained = np.asarray(
        [row["registered_retained_fraction"] for row in result.plane_rows], dtype=float
    )
    relaxed_planes = int(
        sum(bool(row["registered_keep_floor_relaxed"]) for row in result.plane_rows)
    )
    saturation = result.saturation_rows[0]
    qc_status = "pass"
    if relaxed_planes or bool(saturation["plateau_like_upper_tail"]):
        qc_status = "needs_review"
    qc = {
        "schema_version": "labgraph.bench2p_zstack_xyreg.qc.v1",
        "overall_status": qc_status,
        "scope_statement": SCOPE_STATEMENT,
        "gates": {
            "z_geometry": metadata_qc,
            "page_z_grouping": page_grouping_qc,
            "tiff_page_count": {
                "status": "pass",
                "observed": result.page_count,
                "expected": metadata.expected_pages,
            },
            "registered_quality_retention": {
                "status": "pass" if not relaxed_planes else "needs_review",
                "mean_retained_fraction": float(np.mean(retained)),
                "minimum_plane_retained_fraction": float(np.min(retained)),
                "planes_with_relaxed_keep_floor": relaxed_planes,
            },
            "sampled_int16_saturation": {
                "status": "needs_review" if bool(saturation["plateau_like_upper_tail"]) else "pass",
                **saturation,
                "limitation": (
                    "TIFF digital values cannot by themselves exclude optical or analog PMT "
                    "saturation. Exact int16 rail hits and upper-tail concentration are descriptive."
                ),
            },
        },
    }
    write_json(final_dir / "qc.json", qc)
    code = _code_records()
    figure_specs = {
        "fig_bench2p_xy_registration_method_comparison": {
            "question": "How do A-D within-plane aggregation methods change matched XY/XZ/YZ views?",
            "paths": comparison_paths,
            "plot_data": final_dir
            / "figures"
            / "fig_bench2p_xy_registration_method_comparison_plot_data.csv",
            "limits": [
                SCOPE_STATEMENT,
                "Apparent axial elongation mixes optical PSF, sampling, activity, saturation, and motion.",
                "All methods use one display transform derived from method A; source float32 volumes are unchanged.",
            ],
        },
        "fig_bench2p_xy_registration_trace": {
            "question": "What framewise XY shifts and registration quality were measured over stack time?",
            "paths": trace_paths,
            "plot_data": final_dir / "tables" / "frame_registration_qc.csv",
            "limits": [SCOPE_STATEMENT, "The trace does not estimate Z displacement."],
        },
    }
    for figure_id, spec in figure_specs.items():
        json_path = final_dir / "figures" / f"{figure_id}.json"
        md_path = final_dir / "figures" / f"{figure_id}.md"
        write_json(
            json_path,
            {
                "schema_version": "labgraph.figure.v1",
                "figure_id": figure_id,
                "analysis_id": analysis_id,
                "created_at_utc": created_at.isoformat(),
                "question": spec["question"],
                "scope_statement": SCOPE_STATEMENT,
                "input_source_manifest": {
                    "path": repo_relative(final_dir / "sources.json"),
                    "sha256": sha256_file(final_dir / "sources.json"),
                },
                "config": {
                    "path": repo_relative(final_dir / "config.snapshot.json"),
                    "sha256": sha256_file(final_dir / "config.snapshot.json"),
                },
                "code": code,
                "plot_data": {
                    "path": repo_relative(spec["plot_data"]),
                    "sha256": sha256_file(spec["plot_data"]),
                },
                "outputs": {
                    key: {"path": repo_relative(path), "sha256": sha256_file(path)}
                    for key, path in spec["paths"].items()
                },
                "limits": spec["limits"],
            },
        )
        md_path.write_text(
            "\n".join(
                [
                    f"# {figure_id}",
                    "",
                    "## Question",
                    "",
                    str(spec["question"]),
                    "",
                    "## Scope and limits",
                    "",
                    *[f"- {item}" for item in spec["limits"]],
                    "",
                    "## Reproduction",
                    "",
                    f"- Plot data: `{repo_relative(spec['plot_data'])}`",
                    f"- Config: `{repo_relative(final_dir / 'config.snapshot.json')}`",
                    f"- Source manifest: `{repo_relative(final_dir / 'sources.json')}`",
                    "",
                ]
            ),
            encoding="utf-8",
        )

    (final_dir / "README.md").write_text(
        "\n".join(
            [
                f"# {analysis_id}",
                "",
                f"Exploratory A-D reconstruction comparison for `{metadata.scan_id}`.",
                "",
                f"**Scope:** {SCOPE_STATEMENT}.",
                "",
                "- A: all usable raw frames, unregistered mean",
                "- B: legacy correlation-filtered, unregistered mean",
                "- C: all usable frames after within-plane rigid XY registration, mean",
                "- D: registered frames after post-registration quality rejection, mean",
                f"- QC status: `{qc_status}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    output_paths = sorted(
        path for path in final_dir.rglob("*") if path.is_file() and path.name != "analysis_manifest.json"
    )
    manifest = {
        "schema_version": "labgraph.analysis_manifest.v1",
        "analysis_id": analysis_id,
        "title": "bench2p A-D within-plane rigid XY registration benchmark",
        "question": "Does framewise rigid XY registration improve sharpness and repeatability over filter-only averaging?",
        "created_at_utc": created_at.isoformat(),
        "analysis_state": config["analysis_state"],
        "scope_statement": SCOPE_STATEMENT,
        "analysis_unit": ["raw frame", "averaged Z plane", "single sequential stack"],
        "n": {"frames": len(result.frame_rows), "planes": metadata.n_slices},
        "methods": list(METHODS),
        "inputs": {
            "sources_manifest": {
                "path": repo_relative(final_dir / "sources.json"),
                "sha256": sha256_file(final_dir / "sources.json"),
            }
        },
        "configuration": {
            "path": repo_relative(final_dir / "config.snapshot.json"),
            "sha256": sha256_file(final_dir / "config.snapshot.json"),
        },
        "code": code,
        "git": git_state(),
        "qc": {
            "path": repo_relative(final_dir / "qc.json"),
            "sha256": sha256_file(final_dir / "qc.json"),
            "status": qc_status,
        },
        "statistical_method": "descriptive paired within-stack method comparison; no inferential test",
        "supported_interpretations": [
            "A-D differ only in within-plane frame registration/rejection and aggregation as declared."
        ],
        "unsupported_interpretations": [
            "This analysis is not RT-3DMC, 3-D motion correction, axial-motion correction, or a corrected anatomical volume.",
            "Apparent axial FWHM or elongation is not soma morphology.",
        ],
        "outputs": [
            {
                "path": repo_relative(path),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in output_paths
        ],
        "supersedes": None,
        "superseded_by": None,
    }
    write_json(final_dir / "analysis_manifest.json", manifest)
    return final_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ["LABGRAPH_DATA_ROOT"])
        if os.environ.get("LABGRAPH_DATA_ROOT")
        else None,
    )
    parser.add_argument("--animal", required=True, dest="animal_id")
    parser.add_argument("--date", required=True)
    parser.add_argument("--scan", required=True, dest="scan_id")
    parser.add_argument("--read-copy", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--style", type=Path, default=DEFAULT_STYLE)
    parser.add_argument(
        "--output-root", type=Path, default=REPO_ROOT / "datasets" / "analysis_runs"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.data_root is None:
        raise SystemExit("--data-root or LABGRAPH_DATA_ROOT is required")
    run_dir = build_registration_run(
        data_root=args.data_root.expanduser(),
        animal_id=args.animal_id,
        date=args.date,
        scan_id=args.scan_id,
        config_path=args.config.expanduser(),
        style_path=args.style.expanduser(),
        output_root=args.output_root.expanduser(),
        read_copy=args.read_copy.expanduser() if args.read_copy else None,
    )
    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
