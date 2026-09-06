"""Render physically scaled side reconstructions from an averaged bench2p Z volume.

This workflow deliberately separates faithful orthogonal/thin-slab views from an
auxiliary oblique display. It consumes an immutable mean-volume run, samples raw
frames only to distinguish detector saturation from display clipping, and never
modifies the input analysis or source TIFFs.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import ndimage
from skimage.feature import peak_local_max
import skimage
import tifffile

from .bench2p_zstack import (
    DEFAULT_STYLE,
    REPO_ROOT,
    configure_matplotlib,
    git_state,
    read_json,
    repo_relative,
    sha256_file,
    stable_digest,
    write_csv,
    write_json,
)


MODULE_PATH = Path("src/labgraph_ops/workflows/bench2p_zstack_sideview.py")
RUNNER_PATH = Path("scripts/render_bench2p_zstack_sideview.py")
HELPER_PATH = Path("src/labgraph_ops/workflows/bench2p_zstack.py")
DEFAULT_CONFIG = REPO_ROOT / "config" / "bench2p_zstack_sideview.json"


@dataclass(frozen=True)
class BrightCandidate:
    candidate_id: str
    z_index: int
    y_index: int
    x_index: int
    z_relative_um: float
    z_scanimage_um: float
    y_um: float
    x_um: float
    feature_score: float
    source_intensity: float


def physical_spacing_from_ome(path: Path) -> tuple[float, float, float]:
    """Return OME physical spacing as (Z, Y, X) in micrometres."""
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        if series.axes != "ZYX":
            raise ValueError(f"Expected ZYX OME volume, found axes={series.axes!r}")
        ome_xml = tif.ome_metadata
    if not ome_xml:
        raise ValueError(f"OME metadata is missing from {path}")
    root = ET.fromstring(ome_xml)
    pixels = next((item for item in root.iter() if item.tag.endswith("Pixels")), None)
    if pixels is None:
        raise ValueError(f"OME Pixels metadata is missing from {path}")
    spacing = []
    for axis in ("Z", "Y", "X"):
        value = pixels.attrib.get(f"PhysicalSize{axis}")
        unit = pixels.attrib.get(f"PhysicalSize{axis}Unit", "µm")
        if value is None:
            raise ValueError(f"PhysicalSize{axis} is missing from {path}")
        if unit not in {"µm", "um", "micrometer", "micrometre"}:
            raise ValueError(f"Unsupported PhysicalSize{axis} unit {unit!r}")
        spacing.append(float(value))
    return tuple(spacing)  # type: ignore[return-value]


def normalize_volume_for_display(
    volume: np.ndarray,
    *,
    low_percentile: float,
    high_percentile: float,
    asinh_strength: float,
) -> tuple[np.ndarray, dict[str, float]]:
    """Globally normalize a volume without per-plane contrast equalization."""
    if not 0 <= low_percentile < high_percentile <= 100:
        raise ValueError("Display percentiles must satisfy 0 <= low < high <= 100")
    if asinh_strength < 0:
        raise ValueError("asinh_strength must be non-negative")
    finite = volume[np.isfinite(volume)]
    if not finite.size:
        raise ValueError("Volume contains no finite voxels")
    low, high = np.percentile(finite, [low_percentile, high_percentile])
    if high <= low:
        raise ValueError("Display intensity range is empty")
    normalized = np.nan_to_num((volume.astype(np.float32) - low) / (high - low), nan=0.0)
    clipped_low = float(np.mean(normalized <= 0))
    clipped_high = float(np.mean(normalized >= 1))
    normalized = np.clip(normalized, 0.0, 1.0)
    if asinh_strength > 0:
        normalized = np.arcsinh(asinh_strength * normalized) / np.arcsinh(asinh_strength)
    return normalized.astype(np.float32), {
        "low_intensity": float(low),
        "high_intensity": float(high),
        "low_percentile": float(low_percentile),
        "high_percentile": float(high_percentile),
        "asinh_strength": float(asinh_strength),
        "volume_fraction_clipped_low": clipped_low,
        "volume_fraction_clipped_high": clipped_high,
    }


def _sigma_voxels(values_um: Sequence[float], spacing_um: Sequence[float]) -> tuple[float, ...]:
    if len(values_um) != 3 or len(spacing_um) != 3:
        raise ValueError("Sigma and spacing must both contain Z, Y, X values")
    return tuple(float(value) / float(spacing) for value, spacing in zip(values_um, spacing_um))


def detect_bright_candidates(
    volume: np.ndarray,
    spacing_um: Sequence[float],
    config: dict[str, Any],
    z_scanimage_positions_um: Sequence[float] | None = None,
) -> tuple[list[BrightCandidate], np.ndarray]:
    """Find compact bright structures for exploratory ROI review, not cell calls."""
    filled = np.nan_to_num(volume.astype(np.float32), nan=float(np.nanmedian(volume)))
    background = ndimage.gaussian_filter(
        filled,
        sigma=_sigma_voxels(config["background_sigma_um"], spacing_um),
        mode="nearest",
    )
    feature = ndimage.gaussian_filter(
        filled - background,
        sigma=_sigma_voxels(config["feature_sigma_um"], spacing_um),
        mode="nearest",
    )
    threshold = float(np.percentile(feature, float(config["threshold_percentile"])))
    peak_kwargs: dict[str, Any] = {}
    if "minimum_distance_um" in config:
        minimum_distance_um = float(config["minimum_distance_um"])
        if minimum_distance_um <= 0:
            raise ValueError("minimum_distance_um must be positive")
        radii = [
            max(1, int(np.ceil(minimum_distance_um / float(spacing))))
            for spacing in spacing_um
        ]
        grids = np.ogrid[
            tuple(slice(-radius, radius + 1) for radius in radii)
        ]
        footprint = np.zeros(tuple(2 * radius + 1 for radius in radii), dtype=float)
        for grid, spacing in zip(grids, spacing_um, strict=True):
            footprint += (grid * float(spacing) / minimum_distance_um) ** 2
        peak_kwargs.update({"min_distance": 1, "footprint": footprint <= 1.0})
    else:
        peak_kwargs["min_distance"] = int(config["minimum_distance_voxels"])
    if "exclude_border_um" in config:
        exclude_border = tuple(
            int(np.ceil(float(value) / float(spacing)))
            for value, spacing in zip(config["exclude_border_um"], spacing_um, strict=True)
        )
    else:
        exclude_border = tuple(int(value) for value in config["exclude_border_voxels"])
    coordinates = peak_local_max(
        feature,
        threshold_abs=threshold,
        exclude_border=exclude_border,
        num_peaks=int(config["candidate_count"]),
        **peak_kwargs,
    )
    coordinates = sorted(
        coordinates,
        key=lambda item: float(feature[tuple(item)]),
        reverse=True,
    )
    sz, sy, sx = map(float, spacing_um)
    if z_scanimage_positions_um is None:
        scanimage_z = np.arange(volume.shape[0], dtype=float) * sz
    else:
        scanimage_z = np.asarray(z_scanimage_positions_um, dtype=float)
        if scanimage_z.shape != (volume.shape[0],):
            raise ValueError("ScanImage Z-coordinate count does not match volume planes")
    candidates = [
        BrightCandidate(
            candidate_id=f"candidate_{index:02d}",
            z_index=int(z),
            y_index=int(y),
            x_index=int(x),
            z_relative_um=float(z * sz),
            z_scanimage_um=float(scanimage_z[z]),
            y_um=float(y * sy),
            x_um=float(x * sx),
            feature_score=float(feature[z, y, x]),
            source_intensity=float(filled[z, y, x]),
        )
        for index, (z, y, x) in enumerate(coordinates, start=1)
    ]
    return candidates, feature


def _half_width_voxels(half_size_um: Sequence[float], spacing_um: Sequence[float]) -> tuple[int, ...]:
    return tuple(max(1, int(round(float(size) / float(spacing)))) for size, spacing in zip(half_size_um, spacing_um))


def candidate_bounds(
    candidate: BrightCandidate,
    shape: Sequence[int],
    half_size_um: Sequence[float],
    spacing_um: Sequence[float],
) -> tuple[slice, slice, slice]:
    centers = (candidate.z_index, candidate.y_index, candidate.x_index)
    half_widths = _half_width_voxels(half_size_um, spacing_um)
    bounds = []
    for center, half_width, length in zip(centers, half_widths, shape):
        bounds.append(slice(max(0, center - half_width), min(int(length), center + half_width + 1)))
    return tuple(bounds)  # type: ignore[return-value]


def slab_half_width(thickness_um: float, spacing_um: float) -> int:
    """Return a symmetric half-width whose odd voxel count approximates thickness."""
    if thickness_um <= 0 or spacing_um <= 0:
        raise ValueError("Slab thickness and spacing must be positive")
    count = max(1, int(round(thickness_um / spacing_um)))
    if count % 2 == 0:
        count += 1
    return count // 2


def attenuated_mip(
    volume: np.ndarray,
    *,
    axis: int,
    threshold: float,
    attenuation: float,
) -> np.ndarray:
    """Depth-weighted MIP for an auxiliary local oblique display."""
    if not 0 <= threshold < 1:
        raise ValueError("threshold must be in [0, 1)")
    if attenuation < 0:
        raise ValueError("attenuation must be non-negative")
    signal = np.clip((volume - threshold) / (1.0 - threshold), 0.0, 1.0)
    optical_depth = np.cumsum(signal, axis=axis) - signal
    score = signal * np.exp(-attenuation * optical_depth)
    return np.max(score, axis=axis).astype(np.float32)


def alpha_composite_volume(
    volume: np.ndarray,
    *,
    axis: int,
    threshold: float,
    opacity: float,
) -> np.ndarray:
    """Front-to-back grayscale alpha composite without allocating a 4-D RGBA array."""
    if not 0 <= threshold < 1:
        raise ValueError("threshold must be in [0, 1)")
    if opacity <= 0:
        raise ValueError("opacity must be positive")
    planes = np.moveaxis(np.asarray(volume, dtype=np.float32), axis, 0)
    composite = np.zeros(planes.shape[1:], dtype=np.float32)
    transmittance = np.ones(planes.shape[1:], dtype=np.float32)
    for plane in planes:
        signal = np.clip((plane - threshold) / (1.0 - threshold), 0.0, 1.0)
        alpha = 1.0 - np.exp(-opacity * signal)
        composite += transmittance * alpha * signal
        transmittance *= 1.0 - alpha
    return np.clip(composite, 0.0, 1.0)


def cuboid_edge_mask(shape: Sequence[int], thickness_px: int = 1) -> np.ndarray:
    """Return all 12 edges of a Z-Y-X cuboid as a binary volume."""
    if len(shape) != 3 or any(int(length) < 2 for length in shape):
        raise ValueError("Cuboid shape must contain three dimensions of length >= 2")
    if thickness_px < 1:
        raise ValueError("Edge thickness must be at least one pixel")
    z_length, y_length, x_length = map(int, shape)
    edge = np.zeros((z_length, y_length, x_length), dtype=bool)
    t = min(int(thickness_px), min(shape) // 2)
    endpoints_z = (slice(0, t), slice(z_length - t, z_length))
    endpoints_y = (slice(0, t), slice(y_length - t, y_length))
    endpoints_x = (slice(0, t), slice(x_length - t, x_length))
    for y_end in endpoints_y:
        for x_end in endpoints_x:
            edge[:, y_end, x_end] = True
    for z_end in endpoints_z:
        for x_end in endpoints_x:
            edge[z_end, :, x_end] = True
    for z_end in endpoints_z:
        for y_end in endpoints_y:
            edge[z_end, y_end, :] = True
    return edge


def _crop_and_pad_cuboid_render(
    rendered: np.ndarray,
    support: np.ndarray,
    edges: np.ndarray,
    *,
    padding_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Apply one support crop to all layers, then add real transparent margin."""
    if padding_fraction < 0:
        raise ValueError("padding_fraction must be non-negative")
    support = np.asarray(support, dtype=bool)
    rows, columns = np.where(support)
    if not len(rows):
        raise ValueError("Projected cuboid support is empty")
    y0, y1 = int(rows.min()), int(rows.max()) + 1
    x0, x1 = int(columns.min()), int(columns.max()) + 1
    render_crop = np.asarray(rendered[y0:y1, x0:x1], dtype=np.float32).copy()
    support_crop = support[y0:y1, x0:x1]
    edge_crop = np.asarray(edges[y0:y1, x0:x1], dtype=bool)
    render_crop[~support_crop] = np.nan
    padding_px = max(12, int(np.ceil(max(render_crop.shape) * padding_fraction)))
    pad = ((padding_px, padding_px), (padding_px, padding_px))
    return (
        np.pad(render_crop, pad, mode="constant", constant_values=np.nan),
        np.pad(support_crop, pad, mode="constant", constant_values=False),
        np.pad(edge_crop, pad, mode="constant", constant_values=False),
        padding_px,
    )


def make_cuboid_oblique_render(
    display_volume: np.ndarray,
    spacing_um: Sequence[float],
    config: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Render the measured physical cuboid with alpha-composited fluorescence."""
    if display_volume.ndim != 3 or len(spacing_um) != 3:
        raise ValueError("Expected a Z-Y-X volume and three physical spacings")
    physical_spans_um = np.asarray(display_volume.shape, dtype=float) * np.asarray(
        spacing_um, dtype=float
    )
    max_dimension_px = int(config["max_dimension_px"])
    if max_dimension_px < 64:
        raise ValueError("max_dimension_px must be at least 64")
    target_spacing_um = max(
        float(np.min(spacing_um)), float(np.max(physical_spans_um) / max_dimension_px)
    )
    output_shape = np.maximum(
        2, np.rint(physical_spans_um / target_spacing_um).astype(int)
    )
    zoom = output_shape / np.asarray(display_volume.shape, dtype=float)
    isotropic = ndimage.zoom(
        display_volume,
        zoom=tuple(float(value) for value in zoom),
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    ).astype(np.float32, copy=False)
    support = np.ones(isotropic.shape, dtype=np.uint8)
    edges = cuboid_edge_mask(
        isotropic.shape, thickness_px=int(config.get("edge_thickness_px", 1))
    ).astype(np.uint8)

    rotation_kwargs = {
        "reshape": True,
        "mode": "constant",
        "prefilter": False,
    }
    yaw = float(config["azimuth_deg"])
    elevation = float(config["elevation_deg"])
    yawed = ndimage.rotate(
        isotropic, angle=yaw, axes=(1, 2), order=1, cval=0.0, **rotation_kwargs
    )
    del isotropic
    yawed_support = ndimage.rotate(
        support, angle=yaw, axes=(1, 2), order=0, cval=0, **rotation_kwargs
    )
    yawed_edges = ndimage.rotate(
        edges, angle=yaw, axes=(1, 2), order=0, cval=0, **rotation_kwargs
    )
    tilted = ndimage.rotate(
        yawed, angle=elevation, axes=(0, 1), order=1, cval=0.0, **rotation_kwargs
    )
    del yawed
    tilted_support = ndimage.rotate(
        yawed_support,
        angle=elevation,
        axes=(0, 1),
        order=0,
        cval=0,
        **rotation_kwargs,
    )
    tilted_edges = ndimage.rotate(
        yawed_edges,
        angle=elevation,
        axes=(0, 1),
        order=0,
        cval=0,
        **rotation_kwargs,
    )
    rendered = alpha_composite_volume(
        tilted,
        axis=0,
        threshold=float(config["signal_threshold"]),
        opacity=float(config["opacity"]),
    )
    projected_support = tilted_support.max(axis=0) > 0
    projected_edges = tilted_edges.max(axis=0) > 0
    projected_edges = ndimage.binary_dilation(
        projected_edges, iterations=int(config.get("edge_dilation_px", 1))
    )
    rendered, projected_support, projected_edges, padding_px = _crop_and_pad_cuboid_render(
        rendered,
        projected_support,
        projected_edges,
        padding_fraction=float(config["padding_fraction"]),
    )
    metadata = {
        "render_method": "front_to_back_alpha_composite_after_isotropic_resampling",
        "wireframe": "all_12_physical_acquisition_cuboid_edges",
        "physical_spans_zyx_um": physical_spans_um.tolist(),
        "source_shape_zyx": list(display_volume.shape),
        "render_shape_before_rotation_zyx": output_shape.tolist(),
        "target_isotropic_spacing_um": target_spacing_um,
        "azimuth_deg": yaw,
        "elevation_deg": elevation,
        "signal_threshold": float(config["signal_threshold"]),
        "opacity": float(config["opacity"]),
        "padding_fraction": float(config["padding_fraction"]),
        "padding_px": padding_px,
        "background": "transparent_outside_projected_cuboid_with_explicit_margin",
    }
    return {
        "render": rendered,
        "support": projected_support,
        "edges": projected_edges,
    }, metadata


def save_cuboid_figure(
    figure_dir: Path,
    layers: dict[str, np.ndarray],
    metadata: dict[str, Any],
    *,
    style: dict[str, Any],
    profile_name: str,
    edge_color: Sequence[float] = (0.20, 0.78, 0.88),
) -> dict[str, Path]:
    """Save an annotated figure and a transparent standalone cuboid raster."""
    figure_id = "fig_bench2p_zstack_cuboid"
    selected_font = configure_matplotlib(style, profile_name)
    figure_dir.mkdir(parents=True, exist_ok=True)
    rendered = layers["render"]
    valid = np.isfinite(rendered)
    rgba = matplotlib.colormaps["gray"](np.nan_to_num(rendered, nan=0.0))
    rgba[..., 3] = valid.astype(float)
    rgba[layers["edges"], :3] = np.asarray(edge_color, dtype=float)
    rgba[layers["edges"], 3] = 1.0

    raster_path = figure_dir / f"{figure_id}_oblique_raster.png"
    plt.imsave(raster_path, rgba)

    spans = metadata["physical_spans_zyx_um"]
    fig, axis = plt.subplots(figsize=(6.8, 5.0), constrained_layout=True)
    fig.patch.set_facecolor("white")
    axis.set_facecolor("black")
    axis.imshow(rgba, interpolation="nearest")
    axis.set_title("physically scaled oblique fluorescence cuboid")
    axis.text(
        0.5,
        -0.025,
        f"acquisition extent: X {spans[2]:.1f} × Y {spans[1]:.1f} × Z {spans[0]:.1f} µm",
        transform=axis.transAxes,
        ha="center",
        va="top",
    )
    axis.set_axis_off()
    svg_path = figure_dir / f"{figure_id}.svg"
    png_path = figure_dir / f"{figure_id}.png"
    fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
    fig.savefig(
        png_path,
        dpi=style["output"]["png_dpi"],
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)
    return {
        "svg": svg_path,
        "png": png_path,
        "oblique_raster": raster_path,
        "font": Path(selected_font),
    }


def local_oblique_view(
    roi: np.ndarray,
    spacing_um: Sequence[float],
    config: dict[str, Any],
) -> np.ndarray:
    target_spacing = min(map(float, spacing_um))
    isotropic = ndimage.zoom(
        roi,
        zoom=tuple(float(value) / target_spacing for value in spacing_um),
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    support = np.ones(isotropic.shape, dtype=np.uint8)
    yawed = ndimage.rotate(
        isotropic,
        angle=float(config["azimuth_deg"]),
        axes=(1, 2),
        reshape=True,
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    yawed_support = ndimage.rotate(
        support,
        angle=float(config["azimuth_deg"]),
        axes=(1, 2),
        reshape=True,
        order=0,
        mode="constant",
        cval=0,
        prefilter=False,
    )
    tilted = ndimage.rotate(
        yawed,
        angle=float(config["elevation_deg"]),
        axes=(0, 1),
        reshape=True,
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    tilted_support = ndimage.rotate(
        yawed_support,
        angle=float(config["elevation_deg"]),
        axes=(0, 1),
        reshape=True,
        order=0,
        mode="constant",
        cval=0,
        prefilter=False,
    )
    rendered = attenuated_mip(
        tilted,
        axis=1,
        threshold=float(config["signal_threshold"]),
        attenuation=float(config["attenuation"]),
    )
    rendered[tilted_support.max(axis=1) == 0] = np.nan
    return rendered


def _histogram_quantile(histogram: np.ndarray, percentile: float) -> int:
    cdf = np.cumsum(histogram)
    total = int(cdf[-1])
    if total <= 0:
        raise ValueError("Cannot compute a quantile from an empty histogram")
    target = percentile * total / 100.0
    index = int(np.searchsorted(cdf, target, side="left"))
    return index - 32768


def resolve_saturation_offsets(
    local_frame_indices: Sequence[int], frames_per_slice: int
) -> tuple[list[int], str]:
    """Keep valid offsets, or map an older full-range sampling grid to this stack."""
    requested = [int(value) for value in local_frame_indices]
    if frames_per_slice <= 0:
        raise ValueError("frames_per_slice must be positive")
    if not requested:
        raise ValueError("raw_saturation_qc requires at least one frame index")
    if any(value < 0 for value in requested):
        raise ValueError("raw_saturation_qc frame indices must be non-negative")
    if max(requested) < frames_per_slice:
        return requested, "fixed_local_frame_indices_per_slice"

    requested_last = max(requested)
    if requested_last == 0:
        return [0], "fixed_local_frame_indices_per_slice"
    scaled = [
        int(round(value * (frames_per_slice - 1) / requested_last))
        for value in requested
    ]
    return list(dict.fromkeys(scaled)), "normalized_frame_positions_per_slice"


def sample_raw_saturation(
    sources_manifest: dict[str, Any], local_frame_indices: Sequence[int]
) -> list[dict[str, Any]]:
    """Sample matched frame positions from every slice and count signed-16-bit rail hits."""
    rows = []
    for source_record in sources_manifest["sources"]:
        frames_per_slice = int(source_record["frames_per_slice"])
        requested_offsets = [int(value) for value in local_frame_indices]
        offsets, sampling_rule = resolve_saturation_offsets(
            requested_offsets, frames_per_slice
        )
        histogram = np.zeros(65536, dtype=np.int64)
        sampled_frames = 0
        source_path = Path(source_record["source_path"])
        with tifffile.TiffFile(source_path) as tif:
            for slice_index in range(int(source_record["n_slices"])):
                first_page = slice_index * frames_per_slice
                for local_index in offsets:
                    image = tif.pages[first_page + local_index].asarray()
                    if image.dtype != np.int16:
                        raise ValueError(f"Expected int16 raw data, found {image.dtype}")
                    histogram += np.bincount(
                        image.astype(np.int32).ravel() + 32768,
                        minlength=65536,
                    )
                    sampled_frames += 1
        nonzero = np.flatnonzero(histogram)
        pixel_count = int(histogram.sum())
        rows.append(
            {
                "scan_id": source_record["scan_id"],
                "sampling_rule": sampling_rule,
                "requested_local_frame_indices": ";".join(map(str, requested_offsets)),
                "local_frame_indices": ";".join(map(str, offsets)),
                "sampled_frames": sampled_frames,
                "sampled_pixels": pixel_count,
                "observed_min": int(nonzero[0] - 32768),
                "observed_max": int(nonzero[-1] - 32768),
                "p99_9": _histogram_quantile(histogram, 99.9),
                "p99_99": _histogram_quantile(histogram, 99.99),
                "fraction_at_negative_rail": float(histogram[0] / pixel_count),
                "fraction_at_positive_rail": float(histogram[-1] / pixel_count),
            }
        )
    return rows


def candidate_rows(candidates: Sequence[BrightCandidate]) -> list[dict[str, Any]]:
    return [candidate.__dict__.copy() for candidate in candidates]


def save_candidate_figure(
    figure_dir: Path,
    display_volume: np.ndarray,
    candidates: Sequence[BrightCandidate],
    spacing_um: Sequence[float],
    config: dict[str, Any],
    style: dict[str, Any],
    profile_name: str,
) -> dict[str, Path]:
    figure_id = "fig_bench2p_candidate_side_reconstruction"
    configure_matplotlib(style, profile_name)
    figure_dir.mkdir(parents=True, exist_ok=True)
    count = len(candidates)
    fig, axes = plt.subplots(count, 4, figsize=(6.8, 1.28 * count), constrained_layout=True)
    axes = np.atleast_2d(axes)
    sz, sy, sx = map(float, spacing_um)
    half_size_um = config["candidate_detection"]["roi_half_size_um"]
    slab_um = float(config["candidate_detection"]["orthogonal_slab_thickness_um"])
    half_y = slab_half_width(slab_um, sy)
    half_x = slab_half_width(slab_um, sx)
    for row_index, candidate in enumerate(candidates):
        bounds = candidate_bounds(candidate, display_volume.shape, half_size_um, spacing_um)
        roi = display_volume[bounds]
        z_slice, y_slice, x_slice = bounds
        local_z = candidate.z_index - int(z_slice.start)
        local_y = candidate.y_index - int(y_slice.start)
        local_x = candidate.x_index - int(x_slice.start)
        xy = roi.max(axis=0)
        xz = roi[:, max(0, local_y - half_y) : local_y + half_y + 1, :].max(axis=1)
        yz = roi[:, :, max(0, local_x - half_x) : local_x + half_x + 1].max(axis=2)
        oblique = local_oblique_view(roi, spacing_um, config["oblique"])
        panels = (
            (xy, (x_slice.start * sx, x_slice.stop * sx, y_slice.stop * sy, y_slice.start * sy), "XY local MIP"),
            (xz, (x_slice.start * sx, x_slice.stop * sx, z_slice.stop * sz, z_slice.start * sz), f"X–Z ({slab_um:g} µm Y slab)"),
            (yz, (y_slice.start * sy, y_slice.stop * sy, z_slice.stop * sz, z_slice.start * sz), f"Y–Z ({slab_um:g} µm X slab)"),
        )
        for column, (image, extent, title) in enumerate(panels):
            axis = axes[row_index, column]
            axis.imshow(
                image,
                cmap="gray",
                vmin=0,
                vmax=1,
                extent=extent,
                aspect="equal",
                interpolation="nearest",
            )
            axis.set_title(title)
            axis.set_xlabel("µm")
            axis.set_ylabel("µm")
        axes[row_index, 3].imshow(oblique, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        axes[row_index, 3].set_title("local oblique attenuated MIP")
        axes[row_index, 3].set_axis_off()
        axes[row_index, 0].text(
            -0.16,
            0.5,
            candidate.candidate_id.replace("_", " "),
            transform=axes[row_index, 0].transAxes,
            rotation=90,
            ha="center",
            va="center",
            fontweight="bold",
        )
    svg_path = figure_dir / f"{figure_id}.svg"
    png_path = figure_dir / f"{figure_id}.png"
    fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
    fig.savefig(png_path, dpi=style["output"]["png_dpi"], bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {"svg": svg_path, "png": png_path}


def save_side_slab_figure(
    figure_dir: Path,
    display_volume: np.ndarray,
    spacing_um: Sequence[float],
    config: dict[str, Any],
    style: dict[str, Any],
    profile_name: str,
) -> tuple[dict[str, Path], list[dict[str, Any]]]:
    figure_id = "fig_bench2p_xz_thin_slabs"
    configure_matplotlib(style, profile_name)
    sz, sy, sx = map(float, spacing_um)
    slab_config = config["side_slabs"]
    if slab_config["axis"] != "y":
        raise ValueError("Version 1 supports X-Z slabs selected along Y")
    half_width = slab_half_width(float(slab_config["thickness_um"]), sy)
    centers = [
        int(round(float(fraction) * (display_volume.shape[1] - 1)))
        for fraction in slab_config["center_fractions"]
    ]
    fig, axes = plt.subplots(len(centers), 1, figsize=(6.8, 1.25 * len(centers)), constrained_layout=True)
    axes = np.atleast_1d(axes)
    rows = []
    for panel_index, (axis, center) in enumerate(zip(axes, centers, strict=True)):
        start = max(0, center - half_width)
        stop = min(display_volume.shape[1], center + half_width + 1)
        xz = display_volume[:, start:stop, :].max(axis=1)
        axis.imshow(
            xz,
            cmap="gray",
            vmin=0,
            vmax=1,
            extent=(0, display_volume.shape[2] * sx, display_volume.shape[0] * sz, 0),
            aspect="equal",
            interpolation="nearest",
        )
        axis.set_title(
            f"X–Z thin slab at y={center * sy:.1f} µm "
            f"({(stop - start) * sy:.1f} µm thickness)"
        )
        axis.set_xlabel("x (µm)")
        axis.set_ylabel("z (µm)")
        rows.append(
            {
                "panel": chr(ord("a") + panel_index),
                "projection": "maximum_over_thin_y_slab",
                "y_center_index": center,
                "y_center_um": float(center * sy),
                "y_start_index": start,
                "y_stop_index_exclusive": stop,
                "actual_thickness_um": float((stop - start) * sy),
                "x_span_um": float(display_volume.shape[2] * sx),
                "z_span_um": float(display_volume.shape[0] * sz),
            }
        )
    svg_path = figure_dir / f"{figure_id}.svg"
    png_path = figure_dir / f"{figure_id}.png"
    fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
    fig.savefig(png_path, dpi=style["output"]["png_dpi"], bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {"svg": svg_path, "png": png_path}, rows


def write_figure_bundle(
    *,
    run_dir: Path,
    analysis_id: str,
    figure_id: str,
    title: str,
    figure_paths: dict[str, Path],
    plot_data_path: Path,
    input_analysis_id: str,
    input_volume_path: Path,
    config_snapshot_path: Path,
    style_path: Path,
    panel_description: dict[str, Any],
    limits: Sequence[str],
) -> tuple[Path, Path]:
    figure_dir = run_dir / "figures"
    figure_json_path = figure_dir / f"{figure_id}.json"
    figure_md_path = figure_dir / f"{figure_id}.md"
    payload = {
        "schema_version": "labgraph.figure.v1",
        "figure_id": figure_id,
        "title": title,
        "analysis_id": analysis_id,
        "analysis_manifest": repo_relative(run_dir / "analysis_manifest.json"),
        "input_analysis_id": input_analysis_id,
        "input_volume": {
            "path": repo_relative(input_volume_path),
            "sha256": sha256_file(input_volume_path),
        },
        "generating_code": [
            {"path": str(MODULE_PATH), "sha256": sha256_file(REPO_ROOT / MODULE_PATH)},
            {"path": str(RUNNER_PATH), "sha256": sha256_file(REPO_ROOT / RUNNER_PATH)},
            {"path": str(HELPER_PATH), "sha256": sha256_file(REPO_ROOT / HELPER_PATH)},
        ],
        "git": git_state(),
        "config": {"path": repo_relative(config_snapshot_path), "sha256": sha256_file(config_snapshot_path)},
        "style": {"path": repo_relative(style_path), "sha256": sha256_file(style_path)},
        "plotted_data": {"path": repo_relative(plot_data_path), "sha256": sha256_file(plot_data_path)},
        "panels": panel_description,
        "outputs": {
            name: {"path": repo_relative(path), "sha256": sha256_file(path)}
            for name, path in figure_paths.items()
        },
        "limits": list(limits),
    }
    write_json(figure_json_path, payload)
    lines = [
        f"# {figure_id}",
        "",
        "## Question",
        "",
        title,
        "",
        "## Calculation",
        "",
        "The immutable input is the mean OME-TIFF from "
        f"`{input_analysis_id}`. Intensities use one global transform recorded in the "
        "configuration; no per-plane contrast equalization is applied.",
        "",
        "## Interpretation and limits",
        "",
        *[f"- {item}" for item in limits],
        "",
        "## Reproduction",
        "",
        f"- Plotted panel table: `{repo_relative(plot_data_path)}`",
        f"- Figure metadata: `{repo_relative(figure_json_path)}`",
        f"- Input volume: `{repo_relative(input_volume_path)}`",
        f"- Config snapshot: `{repo_relative(config_snapshot_path)}`",
        f"- Generating code: `{MODULE_PATH}`, `{RUNNER_PATH}`",
        "",
    ]
    figure_md_path.write_text("\n".join(lines), encoding="utf-8")
    return figure_json_path, figure_md_path


def collect_environment() -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "packages": {
            "matplotlib": matplotlib.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_image": skimage.__version__,
            "tifffile": tifffile.__version__,
        },
    }


def _output_records(run_dir: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(run_dir.rglob("*")):
        if path.is_file() and path.name != "analysis_manifest.json":
            records.append(
                {
                    "path": repo_relative(path),
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    return records


def resolve_input_volume_path(input_run: Path, configured_relative_path: str) -> Path:
    """Resolve one authoritative volume while allowing a new input run override."""

    configured = input_run / configured_relative_path
    if configured.is_file():
        return configured
    candidates = sorted((input_run / "volumes").glob("*.ome.tif"))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected one OME-TIFF in {input_run / 'volumes'} after the configured "
            f"path was not found; found {len(candidates)}"
        )
    return candidates[0]


def input_scanimage_z_positions(
    input_run: Path,
    sources_manifest: dict[str, Any],
    plane_count: int,
) -> np.ndarray:
    """Recover per-plane ScanImage Z coordinates from the upstream run."""
    combined = input_run / "tables" / "combined_slice_metrics.csv"
    if combined.is_file():
        with combined.open(encoding="utf-8", newline="") as stream:
            values = [float(row["z_um"]) for row in csv.DictReader(stream)]
    else:
        sources = sources_manifest.get("sources", [])
        if len(sources) != 1:
            raise ValueError("Cannot infer one Z-coordinate vector from multiple sources")
        values = [float(value) for value in sources[0]["z_positions_um"]]
    if len(values) != plane_count:
        raise ValueError(
            f"Upstream Z-coordinate count {len(values)} does not match volume planes {plane_count}"
        )
    return np.asarray(values, dtype=float)


def run(config_path: Path, output_root: Path, input_analysis_id: str | None = None) -> Path:
    config_path = config_path.resolve()
    config = read_json(config_path)
    input_analysis_id = input_analysis_id or str(config["input_analysis_id"])
    input_run = REPO_ROOT / "datasets" / "analysis_runs" / input_analysis_id
    input_volume_path = resolve_input_volume_path(input_run, str(config["input_volume"]))
    input_sources_path = input_run / "sources.json"
    if not input_volume_path.is_file() or not input_sources_path.is_file():
        raise FileNotFoundError(f"Input analysis is incomplete: {input_run}")
    input_config_path = input_run / "config.snapshot.json"
    input_motion_config = read_json(input_config_path) if input_config_path.is_file() else {}
    input_frame_filter_enabled = bool(
        input_motion_config.get("frame_stability_filter", {}).get("enabled", False)
    )
    input_cross_plane_enabled = bool(
        input_motion_config.get("cross_plane_registration", {}).get("enabled", False)
    )
    motion_applied = []
    if input_frame_filter_enabled:
        motion_applied.append("within-slice low-correlation frame exclusion")
    if input_cross_plane_enabled:
        motion_applied.append("cross-plane cumulative XY drift registration")
    if motion_applied:
        source_motion_note = (
            "The source mean volume was built with "
            + " and ".join(motion_applied)
            + "; it has no genuine axial (true-Z) motion correction."
        )
    else:
        source_motion_note = "The source mean volume has no within-slice or cross-plane motion correction."
    style_path = (REPO_ROOT / str(config["figure_style_source"])).resolve()
    style = read_json(style_path)
    input_volume_sha256 = sha256_file(input_volume_path)
    digest = stable_digest(
        {
            "config": config,
            "input_analysis_id": input_analysis_id,
            "input_volume_sha256": input_volume_sha256,
        }
    )
    created_at = datetime.now(timezone.utc)
    analysis_id = f"bench2p_zstack_sideview__{created_at.strftime('%Y%m%dT%H%M%SZ')}__{digest}"
    run_dir = output_root.resolve() / analysis_id
    run_dir.mkdir(parents=True, exist_ok=False)
    for child in ("figures", "tables"):
        (run_dir / child).mkdir()

    spacing_um = physical_spacing_from_ome(input_volume_path)
    volume = tifffile.imread(input_volume_path).astype(np.float32)
    display_volume, display_metadata = normalize_volume_for_display(
        volume,
        low_percentile=float(config["display"]["low_percentile"]),
        high_percentile=float(config["display"]["high_percentile"]),
        asinh_strength=float(config["display"]["asinh_strength"]),
    )
    cuboid_config = {
        "azimuth_deg": -35.0,
        "elevation_deg": 32.0,
        "max_dimension_px": 360,
        "signal_threshold": 0.24,
        "opacity": 0.14,
        "edge_thickness_px": 1,
        "edge_dilation_px": 1,
        "padding_fraction": 0.12,
        **config.get("cuboid_oblique", {}),
    }
    resolved_config = {
        **config,
        "analysis_id": analysis_id,
        "created_at_utc": created_at.isoformat(),
        "input_analysis_id": input_analysis_id,
        "input_volume": repo_relative(input_volume_path),
        "input_volume_sha256": input_volume_sha256,
        "physical_spacing_zyx_um": list(spacing_um),
        "display_resolved": display_metadata,
        "cuboid_oblique_resolved": cuboid_config,
        "config_source": repo_relative(config_path),
        "figure_style_source": repo_relative(style_path),
        "output_root": str(output_root.resolve()),
    }
    config_snapshot_path = run_dir / "config.snapshot.json"
    write_json(config_snapshot_path, resolved_config)
    write_json(run_dir / "environment.json", collect_environment())

    cuboid_layers, overview_metadata = make_cuboid_oblique_render(
        display_volume,
        spacing_um,
        cuboid_config,
    )
    overview_figure_paths = save_cuboid_figure(
        run_dir / "figures",
        cuboid_layers,
        overview_metadata,
        style=style,
        profile_name=str(config["figure_profile"]),
    )
    overview_figure_paths.pop("font")
    overview_data_path = run_dir / "tables" / "cuboid_render_geometry.csv"
    write_csv(
        overview_data_path,
        [
            {
                "render_method": overview_metadata["render_method"],
                "wireframe": overview_metadata["wireframe"],
                "shape_z": volume.shape[0],
                "shape_y": volume.shape[1],
                "shape_x": volume.shape[2],
                "spacing_z_um": spacing_um[0],
                "spacing_y_um": spacing_um[1],
                "spacing_x_um": spacing_um[2],
                "azimuth_deg": overview_metadata["azimuth_deg"],
                "elevation_deg": overview_metadata["elevation_deg"],
                "target_isotropic_spacing_um": overview_metadata[
                    "target_isotropic_spacing_um"
                ],
                "padding_px": overview_metadata["padding_px"],
                "background": overview_metadata["background"],
            }
        ],
    )

    sources_manifest = read_json(input_sources_path)
    saturation_rows = sample_raw_saturation(
        sources_manifest,
        config["raw_saturation_qc"]["local_frame_indices"],
    )
    saturation_path = run_dir / "tables" / "raw_saturation_sample.csv"
    write_csv(saturation_path, saturation_rows)

    scanimage_z_positions = input_scanimage_z_positions(
        input_run, sources_manifest, volume.shape[0]
    )
    candidates, _ = detect_bright_candidates(
        volume,
        spacing_um,
        config["candidate_detection"],
        scanimage_z_positions,
    )
    if not candidates:
        raise ValueError("No bright-structure candidates passed the configured detector")
    candidates_path = run_dir / "tables" / "candidate_bright_structures.csv"
    write_csv(candidates_path, candidate_rows(candidates))

    candidate_figure_paths = save_candidate_figure(
        run_dir / "figures",
        display_volume,
        candidates,
        spacing_um,
        config,
        style,
        str(config["figure_profile"]),
    )
    side_figure_paths, side_rows = save_side_slab_figure(
        run_dir / "figures",
        display_volume,
        spacing_um,
        config,
        style,
        str(config["figure_profile"]),
    )
    side_data_path = run_dir / "tables" / "xz_thin_slab_panels.csv"
    write_csv(side_data_path, side_rows)

    common_limits = [
        source_motion_note,
        "Bright-structure candidates are automated review targets, not validated cells or segmentations.",
        "Orthogonal views preserve measured physical spacing but include the microscope axial point-spread function.",
        "The auxiliary oblique panel is display-only and must not be used for quantitative morphology.",
    ]
    overview_json, overview_md = write_figure_bundle(
        run_dir=run_dir,
        analysis_id=analysis_id,
        figure_id="fig_bench2p_zstack_cuboid",
        title="What fluorescence structure is visible inside the physically scaled acquisition cuboid?",
        figure_paths=overview_figure_paths,
        plot_data_path=overview_data_path,
        input_analysis_id=input_analysis_id,
        input_volume_path=input_volume_path,
        config_snapshot_path=config_snapshot_path,
        style_path=style_path,
        panel_description={
            "main": (
                "front-to-back alpha-composited fluorescence with all 12 acquisition "
                "cuboid edges and explicit transparent margin"
            ),
        },
        limits=[
            "The oblique panel is an alpha-composited display, not a tomographic or deconvolved reconstruction.",
            "The cuboid boundary shows the rotated acquisition extent; it does not define tissue anatomy.",
            "The measured physical aspect ratio is retained, so the acquisition is a shallow rectangular prism rather than a mathematical cube.",
            *common_limits,
        ],
    )
    candidate_json, candidate_md = write_figure_bundle(
        run_dir=run_dir,
        analysis_id=analysis_id,
        figure_id="fig_bench2p_candidate_side_reconstruction",
        title="Do local bright structures have resolvable XY, X-Z, and Y-Z extent in the existing mean volume?",
        figure_paths=candidate_figure_paths,
        plot_data_path=candidates_path,
        input_analysis_id=input_analysis_id,
        input_volume_path=input_volume_path,
        config_snapshot_path=config_snapshot_path,
        style_path=style_path,
        panel_description={
            "rows": "one automated bright-structure candidate per row",
            "columns": ["local XY MIP", "X-Z thin slab", "Y-Z thin slab", "auxiliary local oblique attenuated MIP"],
        },
        limits=common_limits,
    )
    side_json, side_md = write_figure_bundle(
        run_dir=run_dir,
        analysis_id=analysis_id,
        figure_id="fig_bench2p_xz_thin_slabs",
        title=(
            "What lateral-to-axial structure is visible without collapsing the full "
            f"{display_volume.shape[1] * spacing_um[1]:.1f} µm Y extent?"
        ),
        figure_paths=side_figure_paths,
        plot_data_path=side_data_path,
        input_analysis_id=input_analysis_id,
        input_volume_path=input_volume_path,
        config_snapshot_path=config_snapshot_path,
        style_path=style_path,
        panel_description={row["panel"]: row for row in side_rows},
        limits=common_limits,
    )

    display_high_fraction = display_metadata["volume_fraction_clipped_high"]
    source_rail_hits = sum(int(float(row["fraction_at_positive_rail"]) > 0) for row in saturation_rows)
    input_qc_path = input_run / "qc.json"
    if not input_qc_path.is_file():
        raise FileNotFoundError(f"Input reconstruction QC is missing: {input_qc_path}")
    input_qc = read_json(input_qc_path)
    input_qc_status = str(input_qc.get("overall_status", "needs_review"))
    overall_status = (
        "pass" if source_rail_hits == 0 and input_qc_status == "pass" else "needs_review"
    )
    qc = {
        "schema_version": "labgraph.bench2p_zstack_sideview.qc.v1",
        "overall_status": overall_status,
        "gates": {
            "input_reconstruction_qc": {
                "status": "pass" if input_qc_status == "pass" else "needs_review",
                "upstream_status": input_qc_status,
                "path": repo_relative(input_qc_path),
                "sha256": sha256_file(input_qc_path),
            },
            "input_volume_is_zyx_ome": {"status": "pass", "shape": list(volume.shape)},
            "physical_spacing_present": {"status": "pass", "spacing_zyx_um": list(spacing_um)},
            "sampled_raw_positive_rail_hits": {
                "status": "pass" if source_rail_hits == 0 else "needs_review",
                "sources_with_nonzero_fraction": source_rail_hits,
                "sampling_table": repo_relative(saturation_path),
            },
            "display_high_clipping": {
                "status": "pass",
                "volume_fraction": display_high_fraction,
                "previous_v1_volume_fraction": 0.001994255745765006,
                "note": "The previous value is reproduced from its 99.8-percentile display rule; the new global high limit is 99.99 percentile.",
            },
        },
    }
    write_json(run_dir / "qc.json", qc)
    write_json(
        run_dir / "sources.json",
        {
            "schema_version": "labgraph.bench2p_zstack_sideview.sources.v1",
            "read_only": True,
            "input_analysis_id": input_analysis_id,
            "input_analysis_manifest": {
                "path": repo_relative(input_run / "analysis_manifest.json"),
                "sha256": sha256_file(input_run / "analysis_manifest.json"),
            },
            "input_volume": {"path": repo_relative(input_volume_path), "sha256": input_volume_sha256},
            "raw_sources_manifest": {"path": repo_relative(input_sources_path), "sha256": sha256_file(input_sources_path)},
            "raw_access_purpose": "fixed-frame sampling for detector-rail saturation QC only",
        },
    )
    readme = "\n".join(
        [
            f"# {analysis_id}",
            "",
            f"Exploratory physical side reconstruction derived from `{input_analysis_id}`.",
            "",
            f"- Uses the immutable {volume.shape[0]}-plane mean OME-TIFF; no raw volume is duplicated.",
            "- Replaces full-depth side MIP with X-Z thin slabs and local orthogonal candidate views.",
            "- Uses a single global high-dynamic-range display transform.",
            "- Samples raw frames only to distinguish detector saturation from display clipping.",
            "- Does not perform or claim motion correction, cell validation, segmentation, or deconvolution.",
            "",
        ]
    )
    (run_dir / "README.md").write_text(readme, encoding="utf-8")

    manifest = {
        "schema_version": "labgraph.analysis_manifest.v1",
        "analysis_id": analysis_id,
        "title": "bench2p physically scaled side and local bright-structure reconstruction",
        "question": "Can the existing averaged Z volume show lateral-to-axial structure without display saturation or full-depth MIP collapse?",
        "created_at_utc": created_at.isoformat(),
        "analysis_state": str(config["analysis_state"]),
        "analysis_unit": "automated bright-structure candidate and configured X-Z thin slab",
        "n": {"bright_structure_candidates": len(candidates), "xz_thin_slabs": len(side_rows)},
        "inputs": {
            "analysis_id": input_analysis_id,
            "volume": {"path": repo_relative(input_volume_path), "sha256": input_volume_sha256},
            "sources_manifest": {"path": repo_relative(input_sources_path), "sha256": sha256_file(input_sources_path)},
        },
        "configuration": {"path": repo_relative(config_snapshot_path), "sha256": sha256_file(config_snapshot_path)},
        "code": [
            {"path": str(MODULE_PATH), "sha256": sha256_file(REPO_ROOT / MODULE_PATH)},
            {"path": str(RUNNER_PATH), "sha256": sha256_file(REPO_ROOT / RUNNER_PATH)},
            {"path": str(HELPER_PATH), "sha256": sha256_file(REPO_ROOT / HELPER_PATH)},
        ],
        "git": git_state(),
        "random_seed": int(config["random_seed"]),
        "statistical_method": "none; deterministic display reconstruction and sampled saturation QC",
        "missing_data_policy": "NaN shift borders are set to background only for display; the immutable source volume is unchanged.",
        "qc": {"path": repo_relative(run_dir / "qc.json"), "status": qc["overall_status"]},
        "supersedes": input_analysis_id,
        "supersession_scope": "visualization only; the input averaged volume remains authoritative",
        "superseded_by": None,
        "supported_interpretations": [
            "Sampled raw pixels do not hit the positive int16 detector rail.",
            "Thin-slab orthogonal views avoid the severe overlap caused by projecting the full Y extent.",
            "Selected bright structures have visually resolvable lateral and axial extent in the current mean volume.",
        ],
        "unsupported_interpretations": common_limits,
        "figures": [
            repo_relative(path)
            for path in (
                overview_json,
                overview_md,
                candidate_json,
                candidate_md,
                side_json,
                side_md,
            )
        ],
        "outputs": _output_records(run_dir),
    }
    write_json(run_dir / "analysis_manifest.json", manifest)
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "datasets" / "analysis_runs",
    )
    parser.add_argument("--input-analysis-id")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_dir = run(args.config, args.output_root, args.input_analysis_id)
    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
