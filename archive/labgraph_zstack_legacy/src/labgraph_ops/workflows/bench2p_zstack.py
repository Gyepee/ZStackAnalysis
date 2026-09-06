"""Build an averaged static 3-D volume from sequential bench2p Z stacks.

The source TIFFs are read-only. Each slow-stack slice is averaged independently,
adjacent stacks are joined at one overlapping Z plane, and the result is written
as a provenance-rich exploratory LabGraph analysis run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import ndimage
from skimage.registration import phase_cross_correlation
import tifffile


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = Path("src/labgraph_ops/workflows/bench2p_zstack.py")
DEFAULT_CONFIG = REPO_ROOT / "config" / "bench2p_zstack.json"
DEFAULT_STYLE = REPO_ROOT / "config" / "figure_style.json"
SESSION_RE = re.compile(
    r"^(?P<operator>[^_]+)_(?P<animal>[A-Z]+-\d+)_(?P<date>\d{4}-\d{2}-\d{2})_"
    r"(?P<scan>scan[^_]+)_(?P<session>sess[^_]+)$"
)


@dataclass(frozen=True)
class StackMetadata:
    source_path: str
    session_dir: str
    animal_id: str
    date: str
    scan_id: str
    session_id: str
    setup: str
    scanimage_config_path: str
    scanimage_user_path: str
    scanimage_version: str
    stack_mode: str
    stack_enabled: bool
    n_slices: int
    frames_per_slice: int
    expected_pages: int
    height_px: int
    width_px: int
    dtype: str
    z_step_um: float
    z_positions_um: tuple[float, ...]
    fov_x_um: float
    fov_y_um: float
    pixel_size_x_um: float
    pixel_size_y_um: float
    frame_rate_hz: float
    size_bytes: int
    mtime_ns: int


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def stable_digest(payload: Any, length: int = 8) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:length]


def repo_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def run_git(args: Sequence[str]) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else ""


def git_state() -> dict[str, Any]:
    status = run_git(["status", "--porcelain"])
    return {
        "commit": run_git(["rev-parse", "HEAD"]),
        "branch": run_git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": bool(status),
        "porcelain_entry_count": len(status.splitlines()) if status else 0,
        "porcelain_sha256": hashlib.sha256(status.encode()).hexdigest(),
    }


def normalize_setup(text: str) -> str:
    compact = text.lower().replace("-", "").replace("_", "")
    if "bench2p" in compact:
        return "bench2p"
    if "mini2p1" in compact:
        return "mini2p1"
    if "mini2p2" in compact:
        return "mini2p2"
    return "unknown"


def _float(value: Any, field: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Missing or invalid ScanImage metadata field {field}: {value!r}") from exc


def _int(value: Any, field: str) -> int:
    result = int(round(_float(value, field)))
    if result <= 0:
        raise ValueError(f"ScanImage metadata field {field} must be positive, got {result}")
    return result


def collapse_z_positions(values: Any, frames_per_slice: int) -> tuple[float, ...]:
    if not isinstance(values, (list, tuple, np.ndarray)):
        raise ValueError("SI.hStackManager.zs is not a sequence")
    flat: list[float] = []
    for item in values:
        if isinstance(item, (list, tuple, np.ndarray)):
            if len(item) == 0:
                continue
            item = item[0]
        flat.append(float(item))
    if not flat:
        raise ValueError("SI.hStackManager.zs is empty")

    collapsed = [flat[0]]
    for value in flat[1:]:
        if not np.isclose(value, collapsed[-1], rtol=0.0, atol=1e-6):
            collapsed.append(value)
    if len(collapsed) == 1 and len(flat) >= frames_per_slice:
        collapsed = [flat[index] for index in range(0, len(flat), frames_per_slice)]
    return tuple(collapsed)


def parse_stack_metadata(path: Path) -> StackMetadata:
    path = path.resolve()
    session_match = SESSION_RE.match(path.parent.name)
    if not session_match:
        raise ValueError(f"Unexpected LabGraph session folder name: {path.parent.name}")
    stat = path.stat()
    with tifffile.TiffFile(path) as tif:
        metadata = tif.scanimage_metadata or {}
        frame_data = metadata.get("FrameData", {}) if isinstance(metadata, dict) else {}
        first_page = tif.pages[0]
        height, width = map(int, first_page.shape)
        dtype = str(first_page.dtype)

    config_path = str(frame_data.get("SI.hConfigurationSaver.cfgFilename", ""))
    user_path = str(frame_data.get("SI.hConfigurationSaver.usrFilename", ""))
    setup = normalize_setup(" ".join((config_path, user_path, path.parent.name)))
    n_slices = _int(
        frame_data.get("SI.hStackManager.actualNumSlices"),
        "SI.hStackManager.actualNumSlices",
    )
    frames_per_slice = _int(
        frame_data.get("SI.hStackManager.framesPerSlice"),
        "SI.hStackManager.framesPerSlice",
    )
    z_positions = collapse_z_positions(
        frame_data.get("SI.hStackManager.zs"), frames_per_slice
    )
    if len(z_positions) != n_slices:
        raise ValueError(
            f"Z-position count {len(z_positions)} does not match actualNumSlices {n_slices}"
        )

    fov = np.asarray(frame_data.get("SI.hRoiManager.imagingFovUm"), dtype=float)
    if fov.shape != (4, 2):
        raise ValueError(f"Unexpected imagingFovUm shape: {fov.shape}")
    fov_x_um = float(np.ptp(fov[:, 0]))
    fov_y_um = float(np.ptp(fov[:, 1]))
    version = ".".join(
        str(frame_data.get(key, ""))
        for key in (
            "SI.VERSION_MAJOR",
            "SI.VERSION_MINOR",
            "SI.VERSION_UPDATE",
        )
    )
    parts = session_match.groupdict()
    return StackMetadata(
        source_path=str(path),
        session_dir=str(path.parent.resolve()),
        animal_id=parts["animal"],
        date=parts["date"],
        scan_id=parts["scan"],
        session_id=parts["session"],
        setup=setup,
        scanimage_config_path=config_path,
        scanimage_user_path=user_path,
        scanimage_version=version,
        stack_mode=str(frame_data.get("SI.hStackManager.stackMode", "")),
        stack_enabled=bool(frame_data.get("SI.hStackManager.enable", False)),
        n_slices=n_slices,
        frames_per_slice=frames_per_slice,
        expected_pages=n_slices * frames_per_slice,
        height_px=height,
        width_px=width,
        dtype=dtype,
        z_step_um=_float(
            frame_data.get("SI.hStackManager.actualStackZStepSize"),
            "SI.hStackManager.actualStackZStepSize",
        ),
        z_positions_um=z_positions,
        fov_x_um=fov_x_um,
        fov_y_um=fov_y_um,
        pixel_size_x_um=fov_x_um / width,
        pixel_size_y_um=fov_y_um / height,
        frame_rate_hz=_float(
            frame_data.get("SI.hRoiManager.scanFrameRate"),
            "SI.hRoiManager.scanFrameRate",
        ),
        size_bytes=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
    )


def discover_stack_tif(
    data_root: Path, animal_id: str, date: str, scan_id: str
) -> Path:
    session_dirs = sorted(data_root.glob(f"JJ_{animal_id}_{date}_{scan_id}_sess*"))
    if len(session_dirs) != 1:
        raise FileNotFoundError(
            f"Expected exactly one uploaded folder for {animal_id} {date} {scan_id}; "
            f"found {len(session_dirs)}"
        )
    tifs = sorted(
        path
        for path in session_dirs[0].iterdir()
        if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}
    )
    if len(tifs) != 1:
        raise ValueError(
            f"Expected exactly one TIF in {session_dirs[0]}; found {len(tifs)}"
        )
    return tifs[0]


def validate_stack_set(metadata: Sequence[StackMetadata]) -> None:
    if len(metadata) != 2:
        raise ValueError("bench2p_zstack v1 requires exactly two adjacent stacks")
    for item in metadata:
        if item.setup != "bench2p":
            raise ValueError(f"{item.scan_id} setup is {item.setup}, not bench2p")
        if not item.stack_enabled:
            raise ValueError(f"{item.scan_id} is not a ScanImage Z stack")
        if item.stack_mode.lower() != "slow":
            raise ValueError(f"{item.scan_id} stack mode is {item.stack_mode!r}, not slow")
    first = metadata[0]
    for item in metadata[1:]:
        for field in (
            "animal_id",
            "date",
            "height_px",
            "width_px",
            "frames_per_slice",
        ):
            if getattr(item, field) != getattr(first, field):
                raise ValueError(f"Stack metadata mismatch in {field}")
        for field in (
            "pixel_size_x_um",
            "pixel_size_y_um",
            "z_step_um",
        ):
            if not np.isclose(getattr(item, field), getattr(first, field), atol=1e-6):
                raise ValueError(f"Stack metadata mismatch in {field}")


def normalized_correlation(left: np.ndarray, right: np.ndarray) -> float:
    mask = np.isfinite(left) & np.isfinite(right)
    if mask.sum() < 2:
        return float("nan")
    a = left[mask].astype(np.float64)
    b = right[mask].astype(np.float64)
    a -= a.mean()
    b -= b.mean()
    denominator = np.sqrt(np.sum(a * a) * np.sum(b * b))
    return float(np.sum(a * b) / denominator) if denominator else float("nan")


def estimate_rigid_shift(
    reference: np.ndarray, moving: np.ndarray, upsample_factor: int
) -> tuple[float, float, float]:
    shift, error, _ = phase_cross_correlation(
        reference.astype(np.float32),
        moving.astype(np.float32),
        upsample_factor=upsample_factor,
        normalization=None,
    )
    return float(shift[0]), float(shift[1]), float(error)


def mean_stack(
    metadata: StackMetadata,
    discard_initial_frames: int,
    drift_qc_frames: int,
) -> tuple[np.ndarray, list[dict[str, Any]], int]:
    if not 0 <= discard_initial_frames < metadata.frames_per_slice:
        raise ValueError("discard_initial_frames_per_slice is outside the slice")
    if not 1 <= drift_qc_frames * 2 <= metadata.frames_per_slice:
        raise ValueError("drift_qc_frames must fit at both ends of a slice")

    volume = np.empty(
        (metadata.n_slices, metadata.height_px, metadata.width_px), dtype=np.float32
    )
    rows: list[dict[str, Any]] = []
    source = Path(metadata.source_path)
    with tifffile.TiffFile(source) as tif:
        page_count = len(tif.pages)
        if page_count != metadata.expected_pages:
            raise ValueError(
                f"{metadata.scan_id} has {page_count} TIFF pages; expected "
                f"{metadata.expected_pages}"
            )
        frame_buffer = np.empty(
            (metadata.frames_per_slice, metadata.height_px, metadata.width_px),
            dtype=tif.pages[0].dtype,
        )
        for slice_index in range(metadata.n_slices):
            first_page = slice_index * metadata.frames_per_slice
            for local_index in range(metadata.frames_per_slice):
                frame_buffer[local_index] = tif.pages[first_page + local_index].asarray()
            selected = frame_buffer[discard_initial_frames:]
            mean_image = selected.mean(axis=0, dtype=np.float64).astype(np.float32)
            volume[slice_index] = mean_image

            early = frame_buffer[:drift_qc_frames].mean(axis=0, dtype=np.float64)
            late = frame_buffer[-drift_qc_frames:].mean(axis=0, dtype=np.float64)
            dy, dx, error = estimate_rigid_shift(early, late, upsample_factor=10)
            late_aligned = ndimage.shift(
                late,
                shift=(dy, dx),
                order=1,
                mode="nearest",
                prefilter=False,
            )
            rows.append(
                {
                    "animal_id": metadata.animal_id,
                    "date": metadata.date,
                    "scan_id": metadata.scan_id,
                    "session_id": metadata.session_id,
                    "source_tif": source.name,
                    "source_slice_index": slice_index,
                    "z_um": metadata.z_positions_um[slice_index],
                    "frames_total": metadata.frames_per_slice,
                    "frames_averaged": len(selected),
                    "discarded_initial_frames": discard_initial_frames,
                    "mean_intensity": float(np.mean(mean_image)),
                    "std_intensity": float(np.std(mean_image)),
                    "p01_intensity": float(np.percentile(mean_image, 1)),
                    "p50_intensity": float(np.percentile(mean_image, 50)),
                    "p99_intensity": float(np.percentile(mean_image, 99)),
                    "early_to_late_shift_y_px": dy,
                    "early_to_late_shift_x_px": dx,
                    "early_to_late_shift_magnitude_px": float(np.hypot(dy, dx)),
                    "early_to_late_registration_error": error,
                    "early_to_late_corr_before": normalized_correlation(early, late),
                    "early_to_late_corr_after": normalized_correlation(early, late_aligned),
                }
            )
            if slice_index == 0 or (slice_index + 1) % 5 == 0 or slice_index + 1 == metadata.n_slices:
                print(
                    f"[{metadata.scan_id}] averaged slice {slice_index + 1}/"
                    f"{metadata.n_slices}",
                    flush=True,
                )
    return volume, rows, page_count


def merge_adjacent_stacks(
    first_volume: np.ndarray,
    first_meta: StackMetadata,
    second_volume: np.ndarray,
    second_meta: StackMetadata,
    *,
    align_boundary_xy: bool,
    upsample_factor: int,
    overlap_tolerance_um: float,
    max_shift_px: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    gap_um = second_meta.z_positions_um[0] - first_meta.z_positions_um[-1]
    if abs(gap_um) > overlap_tolerance_um:
        raise ValueError(
            f"Boundary planes do not overlap: gap={gap_um:.6f} um, "
            f"tolerance={overlap_tolerance_um:.6f} um"
        )

    reference = first_volume[-1]
    moving = second_volume[0]
    if align_boundary_xy:
        dy, dx, error = estimate_rigid_shift(reference, moving, upsample_factor)
    else:
        dy, dx, error = 0.0, 0.0, float("nan")
    magnitude = float(np.hypot(dy, dx))
    if magnitude > max_shift_px:
        raise ValueError(
            f"Boundary XY shift {magnitude:.3f} px exceeds {max_shift_px:.3f} px"
        )

    aligned_second = ndimage.shift(
        second_volume,
        shift=(0.0, dy, dx),
        order=1,
        mode="constant",
        cval=np.nan,
        prefilter=False,
    ).astype(np.float32)
    aligned_boundary = aligned_second[0]
    overlap = np.nanmean(np.stack((reference, aligned_boundary)), axis=0).astype(np.float32)
    merged = np.concatenate(
        (first_volume[:-1], overlap[None, ...], aligned_second[1:]), axis=0
    ).astype(np.float32)
    z_positions = np.asarray(
        [
            *first_meta.z_positions_um[:-1],
            np.mean((first_meta.z_positions_um[-1], second_meta.z_positions_um[0])),
            *second_meta.z_positions_um[1:],
        ],
        dtype=float,
    )
    details = {
        "boundary_gap_um": gap_um,
        "boundary_shift_y_px": dy,
        "boundary_shift_x_px": dx,
        "boundary_shift_magnitude_px": magnitude,
        "boundary_registration_error": error,
        "boundary_corr_before": normalized_correlation(reference, moving),
        "boundary_corr_after": normalized_correlation(reference, aligned_boundary),
        "overlap_operation": "nanmean_of_first_stack_last_and_aligned_second_stack_first",
        "input_slice_count": first_meta.n_slices + second_meta.n_slices,
        "output_slice_count": int(merged.shape[0]),
    }
    return merged, z_positions, details


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_ome_volume(
    path: Path,
    volume: np.ndarray,
    *,
    pixel_size_x_um: float,
    pixel_size_y_um: float,
    z_step_um: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(
        path,
        volume.astype(np.float32, copy=False),
        ome=True,
        bigtiff=volume.nbytes > (4 * 1024**3),
        metadata={
            "axes": "ZYX",
            "PhysicalSizeX": pixel_size_x_um,
            "PhysicalSizeXUnit": "µm",
            "PhysicalSizeY": pixel_size_y_um,
            "PhysicalSizeYUnit": "µm",
            "PhysicalSizeZ": z_step_um,
            "PhysicalSizeZUnit": "µm",
        },
    )


def display_volume(
    volume: np.ndarray, low_percentile: float, high_percentile: float, gamma: float
) -> tuple[np.ndarray, dict[str, float]]:
    finite = volume[np.isfinite(volume)]
    low, high = np.percentile(finite, [low_percentile, high_percentile])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        raise ValueError("Invalid display intensity range")
    normalized = np.nan_to_num((volume - low) / (high - low), nan=0.0)
    normalized = np.clip(normalized, 0.0, 1.0)
    normalized = np.power(normalized, gamma, dtype=np.float32)
    return normalized.astype(np.float32), {
        "low_intensity": float(low),
        "high_intensity": float(high),
        "gamma": float(gamma),
    }


def crop_nonzero(image: np.ndarray, threshold: float = 1e-5, pad: int = 4) -> np.ndarray:
    rows, columns = np.where(image > threshold)
    if not len(rows):
        return image
    y0 = max(0, int(rows.min()) - pad)
    y1 = min(image.shape[0], int(rows.max()) + pad + 1)
    x0 = max(0, int(columns.min()) - pad)
    x1 = min(image.shape[1], int(columns.max()) + pad + 1)
    return image[y0:y1, x0:x1]


def mask_and_crop_rotated_volume(
    image: np.ndarray, support: np.ndarray, pad: int = 4
) -> np.ndarray:
    """Crop to a rotated cuboid footprint and mask only its outside background.

    A rotated MIP is still a two-dimensional image, but retaining a black image
    rectangle around it makes the volume look like a flat photograph.  The
    independently rotated support volume distinguishes the physical cuboid
    footprint from genuinely dark voxels inside it.
    """

    support = np.asarray(support, dtype=bool)
    rows, columns = np.where(support)
    if not len(rows):
        raise ValueError("Rotated volume support is empty")
    y0 = max(0, int(rows.min()) - pad)
    y1 = min(image.shape[0], int(rows.max()) + pad + 1)
    x0 = max(0, int(columns.min()) - pad)
    x1 = min(image.shape[1], int(columns.max()) + pad + 1)
    cropped = np.asarray(image[y0:y1, x0:x1], dtype=np.float32).copy()
    cropped_support = support[y0:y1, x0:x1]
    cropped[~cropped_support] = np.nan
    return cropped


def make_projections(
    volume: np.ndarray,
    *,
    pixel_size_xy_um: float,
    z_step_um: float,
    low_percentile: float,
    high_percentile: float,
    gamma: float,
    azimuth_deg: float,
    elevation_deg: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    normalized, display_metadata = display_volume(
        volume, low_percentile, high_percentile, gamma
    )
    z_zoom = z_step_um / pixel_size_xy_um
    isotropic = ndimage.zoom(
        normalized,
        zoom=(z_zoom, 1.0, 1.0),
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    geometric_support = np.ones(isotropic.shape, dtype=np.uint8)
    axial = normalized.max(axis=0)
    side = isotropic.max(axis=1)
    yawed = ndimage.rotate(
        isotropic,
        angle=azimuth_deg,
        axes=(1, 2),
        reshape=True,
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    yawed_support = ndimage.rotate(
        geometric_support,
        angle=azimuth_deg,
        axes=(1, 2),
        reshape=True,
        order=0,
        mode="constant",
        cval=0,
        prefilter=False,
    )
    tilted = ndimage.rotate(
        yawed,
        angle=elevation_deg,
        axes=(0, 1),
        reshape=True,
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    tilted_support = ndimage.rotate(
        yawed_support,
        angle=elevation_deg,
        axes=(0, 1),
        reshape=True,
        order=0,
        mode="constant",
        cval=0,
        prefilter=False,
    )
    oblique = mask_and_crop_rotated_volume(
        tilted.max(axis=0), tilted_support.max(axis=0) > 0
    )
    metadata = {
        **display_metadata,
        "display_percentile_low": low_percentile,
        "display_percentile_high": high_percentile,
        "z_to_xy_resampling_factor": z_zoom,
        "azimuth_deg": azimuth_deg,
        "elevation_deg": elevation_deg,
        "projection_method": "maximum_intensity_after_isotropic_resampling_and_full_extent_rotation",
        "yaw_reshape": True,
        "oblique_background": "transparent_outside_independently_rotated_cuboid_support",
    }
    return {"axial": axial, "side": side, "oblique": oblique}, metadata


def configure_matplotlib(style: dict[str, Any], profile_name: str) -> str:
    profile = style["profiles"][profile_name]
    candidates = [style["font"]["family"], *style["font"].get("fallbacks", [])]
    installed = {font.name for font in font_manager.fontManager.ttflist}
    selected = next((name for name in candidates if name in installed), "DejaVu Sans")
    matplotlib.rcParams.update(
        {
            "font.family": selected,
            "font.size": profile["font_size_pt"],
            "axes.linewidth": profile["axes_linewidth_pt"],
            "xtick.direction": style["axes"]["tick_direction"],
            "ytick.direction": style["axes"]["tick_direction"],
            "axes.spines.top": style["axes"]["show_top_spine"],
            "axes.spines.right": style["axes"]["show_right_spine"],
            "svg.fonttype": style["font"]["svg_fonttype"],
            "pdf.fonttype": style["font"]["pdf_fonttype"],
        }
    )
    return selected


def save_projection_figure(
    figure_dir: Path,
    projections: dict[str, np.ndarray],
    *,
    style: dict[str, Any],
    profile_name: str,
    pixel_size_x_um: float,
    z_span_um: float,
) -> dict[str, Path]:
    figure_id = "fig_bench2p_zstack_oblique"
    selected_font = configure_matplotlib(style, profile_name)
    figure_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.7), constrained_layout=True)
    axes[0].imshow(projections["axial"], cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("axial Z maximum projection")
    bar_um = 100.0
    bar_px = bar_um / pixel_size_x_um
    y = projections["axial"].shape[0] - 22
    x1 = projections["axial"].shape[1] - 18
    axes[0].plot([x1 - bar_px, x1], [y, y], color="white", linewidth=2.5)
    axes[0].text(x1 - bar_px / 2, y - 9, "100 µm", color="white", ha="center", va="bottom")

    axes[1].imshow(
        projections["side"],
        cmap="gray",
        vmin=0,
        vmax=1,
        extent=(0, projections["side"].shape[1] * pixel_size_x_um, z_span_um, 0),
        aspect="equal",
    )
    axes[1].set_title("side maximum projection")
    axes[1].set_xlabel("x (µm)")
    axes[1].set_ylabel("z (µm)")

    transparent_gray = matplotlib.colormaps["gray"].copy()
    transparent_gray.set_bad((1.0, 1.0, 1.0, 0.0))
    oblique = np.ma.masked_invalid(projections["oblique"])
    axes[2].imshow(oblique, cmap=transparent_gray, vmin=0, vmax=1)
    axes[2].contour(
        np.isfinite(projections["oblique"]).astype(float),
        levels=[0.5],
        colors=["0.65"],
        linewidths=0.45,
    )
    axes[2].set_title("oblique cuboid MIP")
    for index, axis in enumerate(axes):
        axis.text(
            -0.04,
            1.02,
            chr(ord("a") + index),
            transform=axis.transAxes,
            fontweight=style["profiles"][profile_name]["panel_label_weight"],
            fontsize=style["profiles"][profile_name]["panel_label_size_pt"],
            va="bottom",
            ha="right",
        )
        if index != 1:
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

    oblique_path = figure_dir / f"{figure_id}_oblique_raster.png"
    plt.imsave(oblique_path, oblique, cmap=transparent_gray, vmin=0, vmax=1)
    return {
        "svg": svg_path,
        "png": png_path,
        "oblique_raster": oblique_path,
        "font": Path(selected_font),
    }


def combined_slice_rows(
    volume: np.ndarray,
    z_positions: np.ndarray,
    first_meta: StackMetadata,
    second_meta: StackMetadata,
) -> list[dict[str, Any]]:
    rows = []
    overlap_index = first_meta.n_slices - 1
    for index, (image, z_um) in enumerate(zip(volume, z_positions, strict=True)):
        if index < overlap_index:
            source_role = first_meta.scan_id
        elif index == overlap_index:
            source_role = f"{first_meta.scan_id}+{second_meta.scan_id}_overlap_mean"
        else:
            source_role = second_meta.scan_id
        rows.append(
            {
                "animal_id": first_meta.animal_id,
                "date": first_meta.date,
                "combined_slice_index": index,
                "z_um": float(z_um),
                "source_role": source_role,
                "mean_intensity": float(np.nanmean(image)),
                "std_intensity": float(np.nanstd(image)),
                "p01_intensity": float(np.nanpercentile(image, 1)),
                "p50_intensity": float(np.nanpercentile(image, 50)),
                "p99_intensity": float(np.nanpercentile(image, 99)),
                "finite_fraction": float(np.isfinite(image).mean()),
            }
        )
    return rows


def collect_environment() -> dict[str, Any]:
    import skimage

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
