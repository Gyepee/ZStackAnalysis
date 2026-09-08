"""Minimal median reconstruction and mean-projection workflow for slow Z-stacks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
import tifffile

PROJECT_ROOT = Path(__file__).resolve().parents[2]
from .scanimage_io import (
    normalized_correlation,
    parse_stack_metadata,
    write_ome_volume,
)


DEFAULT_CONFIG = PROJECT_ROOT / "config" / "median_projection.json"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "analysis_runs"
DEFAULT_FAILED_ROOT = PROJECT_ROOT / "failed_runs"
DEFAULT_ANALYSIS_TITLE = "median-meanproj"
METHOD_STATEMENT = (
    "per-plane pixelwise median across stored TIFF samples; stored samples preserve "
    "ScanImage file aggregation; no post hoc frame registration or rejection"
)
MOTION_STATEMENT = "no axial-motion correction"


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty table: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def stable_digest(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:8]


def _slug(value: str) -> str:
    cleaned = "".join(
        character if character.isalnum() or character in {"-", "_"} else "-"
        for character in value.strip()
    )
    return cleaned.strip("-") or "analysis"


def analysis_run_id(
    *, metadata: Any, analysis_title: str, created_at: datetime, identity: Any
) -> str:
    scope = f"{metadata.animal_id}_{metadata.date}_{metadata.scan_id}"
    digest = stable_digest(
        {"identity": identity, "created_at_utc": created_at.isoformat()}
    )
    return (
        f"{_slug(analysis_title)}__{_slug(scope)}__"
        f"analyzed-{created_at.strftime('%Y%m%d')}__{digest}"
    )


def validate_z_geometry(
    z_positions_um: Sequence[float],
    actual_step_um: float,
    *,
    tolerance_um: float,
    require_increasing: bool,
) -> dict[str, Any]:
    z = np.asarray(z_positions_um, dtype=float)
    if z.ndim != 1 or z.size < 2 or not np.isfinite(z).all():
        raise ValueError("Z positions must be a finite one-dimensional vector")
    differences = np.diff(z)
    increasing = bool(np.all(differences > 0))
    decreasing = bool(np.all(differences < 0))
    monotonic = increasing or decreasing
    signed_step = float(actual_step_um if increasing else -actual_step_um)
    maximum_error = float(np.max(np.abs(differences - signed_step)))
    regular = bool(maximum_error <= tolerance_um)
    passed = monotonic and regular and (increasing or not require_increasing)
    reasons: list[str] = []
    if not monotonic or (require_increasing and not increasing):
        reasons.append("z_positions_not_strictly_increasing")
    if not regular:
        reasons.append("actualStackZStepSize_disagrees_with_diff_zs")
    return {
        "status": "pass" if passed else "fail",
        "reasons": reasons,
        "actual_stack_z_step_um": float(actual_step_um),
        "median_diff_zs_um": float(np.median(differences)),
        "minimum_diff_zs_um": float(np.min(differences)),
        "maximum_diff_zs_um": float(np.max(differences)),
        "maximum_absolute_spacing_error_um": maximum_error,
        "strictly_increasing": increasing,
        "regular_spacing": regular,
        "plane_count": int(z.size),
    }


def page_grouping_qc(
    page_zs: Sequence[float],
    *,
    planes: int,
    frames_per_plane: int,
    tolerance_um: float,
) -> dict[str, Any]:
    values = np.asarray(page_zs, dtype=float).reshape(-1)
    expected = int(planes * frames_per_plane)
    if values.size != expected:
        return {
            "status": "fail",
            "reason": "page_z_count_mismatch",
            "observed": int(values.size),
            "expected": expected,
        }
    blocks = values.reshape(planes, frames_per_plane)
    deviation = float(np.max(np.abs(blocks - blocks[:, :1])))
    passed = deviation <= tolerance_um
    return {
        "status": "pass" if passed else "fail",
        "reason": None if passed else "z_changes_inside_plane_frame_block",
        "observed": int(values.size),
        "expected": expected,
        "maximum_within_plane_z_deviation_um": deviation,
        "plane_major_contiguous_frame_blocks": bool(passed),
    }


def resolve_source_channel(
    saved_channels: Sequence[int], requested_channel: int | None
) -> tuple[int, int]:
    """Return selected channel ID and its zero-based position in each frame."""
    channels = tuple(int(channel) for channel in saved_channels)
    if not channels:
        raise ValueError("No saved ScanImage channels were found")
    if requested_channel is None:
        if len(channels) != 1:
            raise ValueError(
                f"TIFF contains saved channels {channels}; specify --channel explicitly"
            )
        requested_channel = channels[0]
    if requested_channel not in channels:
        raise ValueError(
            f"Requested channel {requested_channel} is not in saved channels {channels}"
        )
    return requested_channel, channels.index(requested_channel)


def source_page_index(
    stored_frame_index: int, channel_position: int, channel_count: int
) -> int:
    """Map ScanImage's ZTC ordering to the physical TIFF page index."""
    if stored_frame_index < 0:
        raise ValueError("stored_frame_index must be non-negative")
    if channel_count < 1 or not 0 <= channel_position < channel_count:
        raise ValueError("Invalid channel position or channel count")
    return stored_frame_index * channel_count + channel_position


def median_plane(frames: np.ndarray) -> np.ndarray:
    """Return the pixelwise median of all supplied frames in float32."""
    values = np.asarray(frames)
    if values.ndim != 3 or values.shape[0] < 1:
        raise ValueError("Expected frame × Y × X array")
    return np.median(values, axis=0).astype(np.float32)


def maximum_contiguous(values: Sequence[bool]) -> int:
    longest = 0
    current = 0
    for value in values:
        current = current + 1 if bool(value) else 0
        longest = max(longest, current)
    return longest


def _frame_correlations(
    frames: np.ndarray, reference: np.ndarray, stride_px: int
) -> np.ndarray:
    if stride_px < 1:
        raise ValueError("correlation_stride_px must be positive")
    sampled_reference = reference[::stride_px, ::stride_px]
    return np.asarray(
        [
            normalized_correlation(
                sampled_reference, np.asarray(frame)[::stride_px, ::stride_px]
            )
            for frame in frames
        ],
        dtype=float,
    )


def _page_level_z_values(path: Path) -> np.ndarray:
    with tifffile.TiffFile(path) as tif:
        metadata = tif.scanimage_metadata or {}
    frame_data = metadata.get("FrameData", {}) if isinstance(metadata, dict) else {}
    return np.asarray(frame_data.get("SI.hStackManager.zs", []), dtype=float).reshape(-1)


def reconstruct_median_volume(
    metadata: Any,
    read_path: Path,
    config: dict[str, Any],
    *,
    source_channel: int,
    channel_position: int,
) -> tuple[np.ndarray, list[dict[str, Any]], list[dict[str, Any]], int]:
    discard = int(config["aggregation"]["discard_initial_stored_images_per_plane"])
    if not 0 <= discard < metadata.stored_frames_per_slice:
        raise ValueError("discard_initial_stored_images_per_plane is outside the plane")
    stride = int(config["within_plane_qc"]["correlation_stride_px"])
    mad_k = float(config["within_plane_qc"]["descriptive_low_correlation_mad_k"])
    volume = np.empty(
        (metadata.n_slices, metadata.height_px, metadata.width_px), dtype=np.float32
    )
    frame_rows: list[dict[str, Any]] = []
    plane_rows: list[dict[str, Any]] = []
    channel_count = len(metadata.saved_channels)
    with tifffile.TiffFile(read_path) as tif:
        page_count = len(tif.pages)
        if page_count != metadata.expected_tiff_pages:
            raise ValueError(
                f"TIFF contains {page_count} pages; metadata and saved channels "
                f"expect {metadata.expected_tiff_pages}"
            )
        buffer = np.empty(
            (metadata.stored_frames_per_slice, metadata.height_px, metadata.width_px),
            dtype=tif.pages[channel_position].dtype,
        )
        for plane_index in range(metadata.n_slices):
            first_stored_frame = plane_index * metadata.stored_frames_per_slice
            first_acquisition_frame = plane_index * metadata.frames_per_slice
            for local_index in range(metadata.stored_frames_per_slice):
                stored_frame_index = first_stored_frame + local_index
                page_index = source_page_index(
                    stored_frame_index, channel_position, channel_count
                )
                buffer[local_index] = tif.pages[page_index].asarray()
            usable = buffer[discard:]
            reference = median_plane(usable)
            volume[plane_index] = reference
            correlations = _frame_correlations(usable, reference, stride)
            median_corr = float(np.nanmedian(correlations))
            mad = float(np.nanmedian(np.abs(correlations - median_corr)))
            descriptive_threshold = median_corr - mad_k * 1.4826 * mad
            descriptive_low = np.isfinite(correlations) & (
                correlations < descriptive_threshold
            )
            frame_means = usable.mean(axis=(1, 2), dtype=np.float64)
            z_scanimage = float(metadata.z_positions_um[plane_index])
            z_relative = z_scanimage - float(metadata.z_positions_um[0])
            for usable_index, correlation in enumerate(correlations):
                local_index = usable_index + discard
                stored_frame_index = first_stored_frame + local_index
                acquisition_frame_start_index = (
                    first_acquisition_frame + local_index * metadata.log_average_factor
                )
                acquisition_frame_end_index = (
                    acquisition_frame_start_index + metadata.log_average_factor - 1
                )
                page_index = source_page_index(
                    stored_frame_index, channel_position, channel_count
                )
                frame_rows.append(
                    {
                        "plane_index": plane_index,
                        "z_relative_um": z_relative,
                        "z_scanimage_um": z_scanimage,
                        "local_stored_image_index": local_index,
                        "acquisition_frame_start_index": acquisition_frame_start_index,
                        "acquisition_frame_end_index": acquisition_frame_end_index,
                        "acquired_frames_represented": metadata.log_average_factor,
                        "source_storage_aggregation": metadata.storage_aggregation,
                        "source_channel": source_channel,
                        "source_page_index": page_index,
                        "frame_time_s": float(
                            acquisition_frame_start_index / metadata.frame_rate_hz
                        ),
                        "frame_to_plane_median_correlation": float(correlation),
                        "frame_mean_intensity": float(frame_means[usable_index]),
                        "descriptive_low_correlation": bool(
                            descriptive_low[usable_index]
                        ),
                        "included_in_median": True,
                    }
                )
            plane_rows.append(
                {
                    "plane_index": plane_index,
                    "z_relative_um": z_relative,
                    "z_scanimage_um": z_scanimage,
                    "acquired_frames_total": int(metadata.frames_per_slice),
                    "stored_images_total": int(metadata.stored_frames_per_slice),
                    "stored_images_used_in_median": int(usable.shape[0]),
                    "stored_images_rejected": 0,
                    "log_average_factor": int(metadata.log_average_factor),
                    "source_storage_aggregation": metadata.storage_aggregation,
                    "frame_to_median_correlation_min": float(
                        np.nanmin(correlations)
                    ),
                    "frame_to_median_correlation_p05": float(
                        np.nanpercentile(correlations, 5)
                    ),
                    "frame_to_median_correlation_median": median_corr,
                    "frame_to_median_correlation_p95": float(
                        np.nanpercentile(correlations, 95)
                    ),
                    "descriptive_low_correlation_threshold": float(
                        descriptive_threshold
                    ),
                    "descriptive_low_correlation_frame_count": int(
                        descriptive_low.sum()
                    ),
                    "maximum_contiguous_descriptive_low_correlation_frames": (
                        maximum_contiguous(descriptive_low)
                    ),
                    "frame_mean_intensity_cv": float(
                        np.std(frame_means) / np.mean(frame_means)
                    ),
                    "plane_acquisition_duration_s": float(
                        metadata.frames_per_slice / metadata.frame_rate_hz
                    ),
                    "median_image_mean_intensity": float(np.mean(reference)),
                    "median_image_p99_intensity": float(
                        np.percentile(reference, 99)
                    ),
                }
            )
            if (plane_index + 1) % 10 == 0 or plane_index == 0:
                print(
                    f"[{metadata.scan_id}] reconstructed plane {plane_index + 1}/{metadata.n_slices}",
                    flush=True,
                )
    return volume, frame_rows, plane_rows, page_count


def physical_projections(volume: np.ndarray) -> dict[str, np.ndarray]:
    """Return raw-float mean and maximum projections along each physical axis."""
    values = np.asarray(volume, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError("Expected a Z × Y × X volume")
    return {
        "mean_xy_over_z": np.nanmean(values, axis=0),
        "mean_xz_over_y": np.nanmean(values, axis=1),
        "mean_yz_over_x": np.nanmean(values, axis=2),
        "max_xy_over_z": np.nanmax(values, axis=0),
        "max_xz_over_y": np.nanmax(values, axis=1),
        "max_yz_over_x": np.nanmax(values, axis=2),
    }


def _display_transform(
    values: np.ndarray, low: float, high: float, gamma: float
) -> np.ndarray:
    scaled = np.nan_to_num((np.asarray(values, dtype=np.float32) - low) / (high - low))
    return np.clip(scaled, 0, 1) ** gamma


def save_projection_outputs(
    run_dir: Path,
    volume: np.ndarray,
    metadata: Any,
    config: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    display = config["display"]
    finite = volume[np.isfinite(volume)]
    low, high = np.percentile(
        finite,
        [float(display["low_percentile"]), float(display["high_percentile"])],
    )
    gamma = float(display["gamma"])
    if high <= low or gamma <= 0:
        raise ValueError("Invalid display range or gamma")
    projections = physical_projections(volume)
    np.savez_compressed(
        run_dir / "figures" / "projection_source_arrays_float32.npz",
        **{key: np.asarray(value, dtype=np.float32) for key, value in projections.items()},
    )
    figure, axes = plt.subplots(2, 3, figsize=(11.2, 5.8), constrained_layout=True)
    row_specs = (("mean", "mean projection"), ("max", "maximum projection"))
    col_specs = (
        ("xy_over_z", "XY; projected across Z"),
        ("xz_over_y", "XZ; projected across Y"),
        ("yz_over_x", "YZ; projected across X"),
    )
    extent_by_column = {
        "xy_over_z": [
            0,
            metadata.width_px * metadata.pixel_size_x_um,
            metadata.height_px * metadata.pixel_size_y_um,
            0,
        ],
        "xz_over_y": [
            0,
            metadata.width_px * metadata.pixel_size_x_um,
            (metadata.z_positions_um[-1] - metadata.z_positions_um[0])
            + metadata.z_step_um,
            0,
        ],
        "yz_over_x": [
            0,
            metadata.height_px * metadata.pixel_size_y_um,
            (metadata.z_positions_um[-1] - metadata.z_positions_um[0])
            + metadata.z_step_um,
            0,
        ],
    }
    for row_index, (prefix, row_label) in enumerate(row_specs):
        for col_index, (suffix, col_label) in enumerate(col_specs):
            key = f"{prefix}_{suffix}"
            axes[row_index, col_index].imshow(
                _display_transform(projections[key], float(low), float(high), gamma),
                cmap=str(config["figure"]["cmap"]),
                vmin=0,
                vmax=1,
                extent=extent_by_column[suffix],
                aspect="equal",
            )
            axes[row_index, col_index].set_title(f"{row_label}: {col_label}")
            axes[row_index, col_index].set_xlabel("µm")
            axes[row_index, col_index].set_ylabel("µm")
    figure.suptitle(
        f"{metadata.scan_id}: stored-sample-median volume; mean vs maximum projections"
    )
    projection_path = run_dir / "figures" / "fig_mean_vs_max_projections.png"
    figure.savefig(
        projection_path,
        dpi=int(config["figure"]["dpi"]),
        bbox_inches="tight",
    )
    plt.close(figure)

    mean_keys = ("mean_xy_over_z", "mean_xz_over_y", "mean_yz_over_x")
    mean_values = np.concatenate([projections[key].ravel() for key in mean_keys])
    mean_low, mean_high = np.percentile(
        mean_values,
        [float(display["low_percentile"]), float(display["high_percentile"])],
    )
    figure, axes = plt.subplots(1, 3, figsize=(11.2, 3.2), constrained_layout=True)
    for axis, (suffix, title) in zip(axes, col_specs, strict=True):
        key = f"mean_{suffix}"
        axis.imshow(
            _display_transform(
                projections[key], float(mean_low), float(mean_high), gamma
            ),
            cmap=str(config["figure"]["cmap"]),
            vmin=0,
            vmax=1,
            extent=extent_by_column[suffix],
            aspect="equal",
        )
        axis.set_title(title)
        axis.set_xlabel("µm")
        axis.set_ylabel("µm")
    figure.suptitle(
        f"{metadata.scan_id}: mean projections of the stored-sample-median volume"
    )
    figure.savefig(
        run_dir / "figures" / "fig_mean_projections.png",
        dpi=int(config["figure"]["dpi"]),
        bbox_inches="tight",
    )
    plt.close(figure)

    representative_index = int(
        round(
            float(config["projection"]["representative_z_fraction"])
            * (volume.shape[0] - 1)
        )
    )
    figure, axis = plt.subplots(figsize=(5.3, 5.0), constrained_layout=True)
    axis.imshow(
        _display_transform(volume[representative_index], float(low), float(high), gamma),
        cmap=str(config["figure"]["cmap"]),
        vmin=0,
        vmax=1,
        extent=extent_by_column["xy_over_z"],
        aspect="equal",
    )
    axis.set(
        title=(
            f"pixelwise-median XY plane {representative_index}; "
            f"z={metadata.z_positions_um[representative_index]:.3f} µm"
        ),
        xlabel="X (µm)",
        ylabel="Y (µm)",
    )
    plane_path = run_dir / "figures" / "fig_representative_median_plane.png"
    figure.savefig(plane_path, dpi=int(config["figure"]["dpi"]))
    plt.close(figure)

    rows = []
    for name, array in projections.items():
        rows.append(
            {
                "projection": name,
                "aggregation_axis": name.rsplit("_", 1)[-1],
                "source_dtype": "float32 median volume",
                "minimum": float(np.nanmin(array)),
                "mean": float(np.nanmean(array)),
                "maximum": float(np.nanmax(array)),
            }
        )
    return (
        {
            "low_intensity": float(low),
            "high_intensity": float(high),
            "low_percentile": float(display["low_percentile"]),
            "high_percentile": float(display["high_percentile"]),
            "gamma": gamma,
            "same_transform_for_all_panels": True,
            "mean_only_figure_low_intensity": float(mean_low),
            "mean_only_figure_high_intensity": float(mean_high),
            "mean_only_figure_uses_one_transform_for_all_three_panels": True,
            "representative_plane_index": representative_index,
        },
        rows,
    )


def adjacent_plane_rows(volume: np.ndarray, metadata: Any) -> list[dict[str, Any]]:
    rows = []
    for index in range(volume.shape[0] - 1):
        rows.append(
            {
                "plane_index_a": index,
                "plane_index_b": index + 1,
                "z_a_scanimage_um": float(metadata.z_positions_um[index]),
                "z_b_scanimage_um": float(metadata.z_positions_um[index + 1]),
                "delta_z_um": float(
                    metadata.z_positions_um[index + 1]
                    - metadata.z_positions_um[index]
                ),
                "normalized_correlation": normalized_correlation(
                    volume[index], volume[index + 1]
                ),
            }
        )
    return rows


def collect_environment() -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "tifffile": tifffile.__version__,
        "matplotlib": matplotlib.__version__,
    }


def build_run(
    *,
    source_tiff: Path,
    read_copy: Path | None,
    config_path: Path,
    output_root: Path,
    failed_root: Path,
    source_channel: int | None,
) -> Path:
    source_tiff = source_tiff.resolve()
    config_path = config_path.resolve()
    config = read_json(config_path)
    metadata = parse_stack_metadata(source_tiff)
    if metadata.setup != "bench2p" or not metadata.stack_enabled:
        raise ValueError("Input is not an enabled Bench2p ScanImage stack")
    source_channel, channel_position = resolve_source_channel(
        metadata.saved_channels, source_channel
    )
    tolerance = float(config["metadata_qc"]["z_spacing_absolute_tolerance_um"])
    z_qc = validate_z_geometry(
        metadata.z_positions_um,
        metadata.z_step_um,
        tolerance_um=tolerance,
        require_increasing=bool(config["metadata_qc"]["require_increasing_z"]),
    )
    grouping_qc = page_grouping_qc(
        _page_level_z_values(source_tiff),
        planes=metadata.n_slices,
        frames_per_plane=metadata.frames_per_slice,
        tolerance_um=tolerance,
    )
    if z_qc["status"] != "pass" or grouping_qc["status"] != "pass":
        raise ValueError(f"Metadata admission failed: z={z_qc}, grouping={grouping_qc}")
    read_path = source_tiff if read_copy is None else read_copy.resolve()
    if read_copy is not None:
        if read_path.name != source_tiff.name:
            raise ValueError("read copy filename differs from the authoritative TIFF")
        if read_path.stat().st_size != source_tiff.stat().st_size:
            raise ValueError("read copy size differs from the authoritative TIFF")
    identity = {
        "source": str(source_tiff),
        "source_size": source_tiff.stat().st_size,
        "source_mtime_ns": source_tiff.stat().st_mtime_ns,
        "source_channel": source_channel,
        "config": config,
    }
    created_at = datetime.now(timezone.utc)
    analysis_title = str(config.get("analysis_title", DEFAULT_ANALYSIS_TITLE))
    run_id = analysis_run_id(
        metadata=metadata,
        analysis_title=analysis_title,
        created_at=created_at,
        identity=identity,
    )
    final_dir = output_root.resolve() / run_id
    incomplete_dir = failed_root.resolve() / f".{run_id}.incomplete"
    if final_dir.exists() or incomplete_dir.exists():
        raise FileExistsError(f"Run already exists: {run_id}")
    for child in ("volumes", "tables", "figures"):
        (incomplete_dir / child).mkdir(parents=True, exist_ok=child != "volumes")
    try:
        volume, frame_rows, plane_rows, page_count = reconstruct_median_volume(
            metadata,
            read_path,
            config,
            source_channel=source_channel,
            channel_position=channel_position,
        )
        channel_suffix = (
            f"__channel-{source_channel}" if len(metadata.saved_channels) > 1 else ""
        )
        volume_path = (
            incomplete_dir
            / "volumes"
            / f"{metadata.scan_id}{channel_suffix}__median.ome.tif"
        )
        write_ome_volume(
            volume_path,
            volume,
            pixel_size_x_um=metadata.pixel_size_x_um,
            pixel_size_y_um=metadata.pixel_size_y_um,
            z_step_um=metadata.z_step_um,
        )
        write_rows(incomplete_dir / "tables" / "frame_to_median_qc.csv", frame_rows)
        write_rows(incomplete_dir / "tables" / "plane_median_qc.csv", plane_rows)
        write_rows(
            incomplete_dir / "tables" / "adjacent_plane_correlation.csv",
            adjacent_plane_rows(volume, metadata),
        )
        display_metadata, projection_rows = save_projection_outputs(
            incomplete_dir, volume, metadata, config
        )
        write_rows(
            incomplete_dir / "tables" / "projection_summary.csv", projection_rows
        )
        resolved = {
            **config,
            "analysis_id": run_id,
            "created_at_utc": created_at.isoformat(),
            "authoritative_source_tiff": str(source_tiff),
            "local_read_copy_used": read_copy is not None,
            "selected_source_channel": source_channel,
            "source_channel_position": channel_position,
            "metadata": asdict(metadata),
            "display_resolved": display_metadata,
            "method_statement": METHOD_STATEMENT,
            "motion_statement": MOTION_STATEMENT,
        }
        write_json(incomplete_dir / "config.snapshot.json", resolved)
        write_json(incomplete_dir / "environment.json", collect_environment())
        write_json(
            incomplete_dir / "qc.json",
            {
                "overall_status": "pass",
                "method_statement": METHOD_STATEMENT,
                "motion_statement": MOTION_STATEMENT,
                "z_geometry": z_qc,
                "page_grouping": grouping_qc,
                "page_count": {
                    "status": "pass",
                    "observed": page_count,
                    "expected": metadata.expected_tiff_pages,
                },
                "source_channel": {
                    "saved_channels": list(metadata.saved_channels),
                    "selected": source_channel,
                    "position_in_each_frame": channel_position,
                    "page_order": "ZTC (channel varies fastest)",
                },
                "interpretation_limit": (
                    "Median aggregation is robust to a minority of transient outlier "
                    "frames but is not motion correction and cannot establish axial stability."
                ),
            },
        )
        output_root.resolve().mkdir(parents=True, exist_ok=True)
        shutil.move(str(incomplete_dir), str(final_dir))
    except BaseException:
        if incomplete_dir.exists():
            write_json(
                incomplete_dir / "INCOMPLETE.json",
                {"analysis_id": run_id, "status": "incomplete"},
            )
        raise
    files = sorted(path for path in final_dir.rglob("*") if path.is_file())
    source_sha256 = sha256_file(read_path)
    code_paths = (
        PROJECT_ROOT / "src" / "zstack_analysis" / "pipeline.py",
        PROJECT_ROOT / "src" / "zstack_analysis" / "scanimage_io.py",
        PROJECT_ROOT / "scripts" / "reconstruct_median_stack.py",
    )
    write_json(
        final_dir / "manifest.json",
        {
            "schema_version": "zstack_analysis.median_projection.run.v1",
            "analysis_id": run_id,
            "created_at_utc": created_at.isoformat(),
            "method_statement": METHOD_STATEMENT,
            "motion_statement": MOTION_STATEMENT,
            "source": {
                "path": str(source_tiff),
                "saved_channels": list(metadata.saved_channels),
                "selected_channel": source_channel,
                "size_bytes": source_tiff.stat().st_size,
                "mtime_ns": source_tiff.stat().st_mtime_ns,
                "read_only": True,
                "sha256": source_sha256,
                "sha256_computed_from_identical_local_read_copy": read_copy is not None,
            },
            "code": [
                {
                    "path": str(path.relative_to(PROJECT_ROOT)),
                    "sha256": sha256_file(path),
                }
                for path in code_paths
            ],
            "outputs": [
                {
                    "path": str(path.relative_to(final_dir)),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in files
            ],
        },
    )
    return final_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-tiff", type=Path, required=True)
    parser.add_argument("--read-copy", type=Path)
    parser.add_argument(
        "--channel",
        type=int,
        help="Saved ScanImage channel to reconstruct; required for multichannel TIFFs",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--failed-root", type=Path, default=DEFAULT_FAILED_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_dir = build_run(
        source_tiff=args.source_tiff,
        read_copy=args.read_copy,
        config_path=args.config,
        output_root=args.output_root,
        failed_root=args.failed_root,
        source_channel=args.channel,
    )
    print(json.dumps({"analysis_run": str(run_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
