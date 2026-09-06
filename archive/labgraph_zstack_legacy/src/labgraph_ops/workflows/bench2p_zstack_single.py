"""Build an averaged static 3-D volume from one complete bench2p slow Z stack.

Companion to `bench2p_zstack.py`, which merges two half-range stacks at one
overlapping boundary plane. Some bench2p slow stacks already span the full
physical Z range in a single acquisition; forcing such a stack through the
boundary-overlap merge (which assumes two adjacent, non-overlapping ranges)
is not applicable. This workflow applies the same per-slice averaging, drift
QC, and provenance conventions to that single-stack case.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import tifffile
from scipy import ndimage

from .bench2p_zstack import (
    DEFAULT_STYLE,
    REPO_ROOT,
    StackMetadata,
    collect_environment,
    discover_stack_tif,
    estimate_rigid_shift,
    git_state,
    make_projections,
    mean_stack,
    normalized_correlation,
    parse_stack_metadata,
    read_json,
    repo_relative,
    save_projection_figure,
    sha256_file,
    stable_digest,
    write_csv,
    write_json,
    write_ome_volume,
)


MODULE_PATH = Path("src/labgraph_ops/workflows/bench2p_zstack_single.py")
RUNNER_PATH = Path("scripts/build_bench2p_zstack_single.py")
HELPER_PATH = Path("src/labgraph_ops/workflows/bench2p_zstack.py")
DEFAULT_CONFIG = REPO_ROOT / "config" / "bench2p_zstack_single.json"


def code_records() -> list[dict[str, str]]:
    return [
        {"path": path.as_posix(), "sha256": sha256_file(REPO_ROOT / path)}
        for path in (RUNNER_PATH, MODULE_PATH, HELPER_PATH)
    ]


def validate_single_stack(metadata: StackMetadata) -> None:
    if metadata.setup != "bench2p":
        raise ValueError(f"{metadata.scan_id} setup is {metadata.setup}, not bench2p")
    if not metadata.stack_enabled:
        raise ValueError(f"{metadata.scan_id} is not a ScanImage Z stack")
    if metadata.stack_mode.lower() != "slow":
        raise ValueError(f"{metadata.scan_id} stack mode is {metadata.stack_mode!r}, not slow")


def mean_stack_frame_filtered(
    metadata: StackMetadata,
    discard_initial_frames: int,
    drift_qc_frames: int,
    mad_k: float,
    min_keep_fraction: float,
) -> tuple[np.ndarray, list[dict[str, Any]], int]:
    """Average, per Z slice, only the raw frames whose stability is not an outlier.

    Stability is each frame's normalized (Pearson) cross-correlation with that
    slice's own per-pixel median frame (a registration-free consensus
    reference). Correlation is used instead of a rigid-shift/registration
    estimate because single raw two-photon frames are too low-SNR for
    phase-correlation registration to be reliable frame-to-frame; that
    approach was tried and produced spurious near-half-image shift estimates
    on many slices even where the slower early-vs-late 5-frame-averaged
    diagnostic showed genuinely low motion.

    A frame is dropped only if its correlation is a genuine low outlier
    relative to the other frames in the *same* slice: below
    ``median - mad_k * 1.4826 * MAD`` of that slice's own correlation
    distribution. The number of frames kept therefore varies slice to slice
    with how many frames actually look unstable, rather than always
    discarding a fixed count. If more than ``1 - min_keep_fraction`` of a
    slice's frames would be dropped, the cutoff is relaxed to keep the
    top ``min_keep_fraction`` by correlation instead, as a floor against a
    pathological slice collapsing to too few averaged frames.

    This filter targets within-slice XY jitter/instability over the
    ~frames_per_slice/frame_rate acquisition window of a single Z plane; it
    does not measure or correct axial (Z) drift across the whole sequential
    stack.
    """
    if not 0 <= discard_initial_frames < metadata.frames_per_slice:
        raise ValueError("discard_initial_frames_per_slice is outside the slice")
    if not 1 <= drift_qc_frames * 2 <= metadata.frames_per_slice:
        raise ValueError("drift_qc_frames must fit at both ends of a slice")
    if mad_k <= 0:
        raise ValueError("mad_k must be positive")
    if not 0 < min_keep_fraction <= 1.0:
        raise ValueError("min_keep_fraction must be in (0, 1]")

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
            usable = frame_buffer[discard_initial_frames:]

            reference = np.median(usable, axis=0)
            correlations = np.empty(usable.shape[0], dtype=np.float64)
            for frame_index in range(usable.shape[0]):
                correlations[frame_index] = normalized_correlation(
                    reference, usable[frame_index]
                )

            median_corr = float(np.median(correlations))
            mad = float(np.median(np.abs(correlations - median_corr)))
            lower_bound = median_corr - mad_k * 1.4826 * mad
            keep_mask = correlations >= lower_bound

            min_keep_count = max(1, int(np.ceil(usable.shape[0] * min_keep_fraction)))
            if keep_mask.sum() < min_keep_count:
                order = np.argsort(correlations)[::-1]
                keep_mask = np.zeros(usable.shape[0], dtype=bool)
                keep_mask[order[:min_keep_count]] = True

            kept_local_indices = np.flatnonzero(keep_mask)
            dropped_local_indices = np.flatnonzero(~keep_mask)
            mean_image = usable[kept_local_indices].mean(axis=0, dtype=np.float64).astype(
                np.float32
            )
            volume[slice_index] = mean_image

            early = frame_buffer[:drift_qc_frames].mean(axis=0, dtype=np.float64)
            late = frame_buffer[-drift_qc_frames:].mean(axis=0, dtype=np.float64)
            dy_el, dx_el, error_el = estimate_rigid_shift(early, late, upsample_factor=10)

            kept_corr = correlations[kept_local_indices]
            dropped_corr = (
                correlations[dropped_local_indices]
                if dropped_local_indices.size
                else np.asarray([np.nan])
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
                    "frames_usable": int(usable.shape[0]),
                    "discarded_initial_frames": discard_initial_frames,
                    "frames_kept": int(kept_local_indices.size),
                    "frames_dropped": int(dropped_local_indices.size),
                    "kept_frame_corr_mean": float(np.mean(kept_corr)),
                    "kept_frame_corr_min": float(np.min(kept_corr)),
                    "dropped_frame_corr_mean": float(np.mean(dropped_corr)),
                    "dropped_frame_corr_max": float(
                        np.max(dropped_corr) if dropped_local_indices.size else np.nan
                    ),
                    "mean_intensity": float(np.mean(mean_image)),
                    "std_intensity": float(np.std(mean_image)),
                    "p01_intensity": float(np.percentile(mean_image, 1)),
                    "p50_intensity": float(np.percentile(mean_image, 50)),
                    "p99_intensity": float(np.percentile(mean_image, 99)),
                    "early_to_late_shift_y_px": dy_el,
                    "early_to_late_shift_x_px": dx_el,
                    "early_to_late_shift_magnitude_px": float(np.hypot(dy_el, dx_el)),
                    "early_to_late_registration_error": error_el,
                    "early_to_late_corr_before": normalized_correlation(early, late),
                }
            )
            if slice_index == 0 or (slice_index + 1) % 5 == 0 or slice_index + 1 == metadata.n_slices:
                print(
                    f"[{metadata.scan_id}] frame-filtered slice {slice_index + 1}/"
                    f"{metadata.n_slices} (kept {kept_local_indices.size}/{usable.shape[0]})",
                    flush=True,
                )
    return volume, rows, page_count


def apply_cross_plane_registration(
    volume: np.ndarray,
    z_positions_um: Sequence[float],
    mad_k: float,
    max_step_shift_px: float,
    upsample_factor: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Correct cumulative XY drift across consecutive Z planes of one sequential stack.

    Adjacent Z planes are only one z_step apart and, given the microscope's axial
    point-spread function, remain highly correlated in structure. Each plane is
    registered (rigid XY, phase cross-correlation) against the immediately
    preceding *raw* plane; the per-step shifts are summed into a cumulative
    trajectory and that cumulative shift is applied to every plane. This targets
    slow XY drift that accumulates while sequentially imaging ~100 planes over
    the whole stack acquisition (order of minutes) -- a different axis from the
    within-slice frame-stability filter above, which only handles jitter among
    the frames captured while parked at a single Z position.

    A step is excluded from the cumulative trajectory (treated as zero relative
    motion for that step) if its plane-to-plane correlation is a low MAD outlier
    relative to the run's other steps, or if the candidate shift magnitude
    exceeds ``max_step_shift_px``: neighboring planes can legitimately differ in
    structure (that is the point of a Z-stack), and injecting a spurious
    large single-step "shift" estimated from decorrelated content would be
    worse than assuming no motion for that step.

    This recovers cross-plane XY translation only. It does not estimate or
    correct genuine axial (true-Z) displacement, which is not observable from a
    single sequential stack without a dense reference or repeat volume.
    """
    if mad_k <= 0:
        raise ValueError("mad_k must be positive")
    if max_step_shift_px <= 0:
        raise ValueError("max_step_shift_px must be positive")

    n_slices = volume.shape[0]
    candidate_dy = np.zeros(n_slices, dtype=np.float64)
    candidate_dx = np.zeros(n_slices, dtype=np.float64)
    step_error = np.full(n_slices, np.nan, dtype=np.float64)
    step_corr = np.full(n_slices, np.nan, dtype=np.float64)
    for index in range(1, n_slices):
        dy, dx, error = estimate_rigid_shift(
            volume[index - 1], volume[index], upsample_factor
        )
        candidate_dy[index] = dy
        candidate_dx[index] = dx
        step_error[index] = error
        step_corr[index] = normalized_correlation(volume[index - 1], volume[index])

    candidate_magnitude = np.hypot(candidate_dy, candidate_dx)
    corr_steps = step_corr[1:]
    finite_corr = corr_steps[np.isfinite(corr_steps)]
    if finite_corr.size:
        median_corr = float(np.median(finite_corr))
        mad = float(np.median(np.abs(finite_corr - median_corr)))
        lower_bound = median_corr - mad_k * 1.4826 * mad
    else:
        lower_bound = -np.inf

    applied = np.zeros(n_slices, dtype=bool)
    applied_dy = np.zeros(n_slices, dtype=np.float64)
    applied_dx = np.zeros(n_slices, dtype=np.float64)
    for index in range(1, n_slices):
        reliable = (
            np.isfinite(step_corr[index])
            and step_corr[index] >= lower_bound
            and candidate_magnitude[index] <= max_step_shift_px
        )
        applied[index] = reliable
        if reliable:
            applied_dy[index] = candidate_dy[index]
            applied_dx[index] = candidate_dx[index]

    cumulative_dy = np.cumsum(applied_dy)
    cumulative_dx = np.cumsum(applied_dx)

    corrected = np.empty_like(volume)
    corrected[0] = volume[0]
    for index in range(1, n_slices):
        corrected[index] = ndimage.shift(
            volume[index],
            shift=(cumulative_dy[index], cumulative_dx[index]),
            order=1,
            mode="nearest",
            prefilter=False,
        )

    rows: list[dict[str, Any]] = []
    for index in range(n_slices):
        rows.append(
            {
                "slice_index": index,
                "z_um": float(z_positions_um[index]),
                "step_candidate_shift_y_px": float(candidate_dy[index]),
                "step_candidate_shift_x_px": float(candidate_dx[index]),
                "step_candidate_shift_magnitude_px": float(candidate_magnitude[index]),
                "step_registration_error": float(step_error[index])
                if np.isfinite(step_error[index])
                else None,
                "step_corr_before": float(step_corr[index])
                if np.isfinite(step_corr[index])
                else None,
                "step_applied": bool(applied[index]),
                "cumulative_shift_y_px": float(cumulative_dy[index]),
                "cumulative_shift_x_px": float(cumulative_dx[index]),
                "cumulative_shift_magnitude_px": float(
                    np.hypot(cumulative_dy[index], cumulative_dx[index])
                ),
            }
        )
    return corrected, rows


def build_run(
    *,
    data_root: Path,
    animal_id: str,
    date: str,
    scan_id: str,
    config_path: Path,
    style_path: Path,
    output_root: Path,
) -> Path:
    config = read_json(config_path)
    style = read_json(style_path)
    source_path = discover_stack_tif(data_root, animal_id, date, scan_id)
    metadata = parse_stack_metadata(source_path)
    validate_single_stack(metadata)

    identity = {
        "config": config,
        "source": {
            "path": str(source_path.resolve()),
            "size": source_path.stat().st_size,
            "mtime_ns": source_path.stat().st_mtime_ns,
        },
    }
    created_at = datetime.now(timezone.utc)
    analysis_id = (
        "bench2p_zstack_single__"
        + created_at.strftime("%Y%m%dT%H%M%SZ")
        + "__"
        + stable_digest(identity)
    )
    run_dir = output_root.resolve() / analysis_id
    if run_dir.exists():
        raise FileExistsError(f"Immutable run already exists: {run_dir}")
    for child in ("tables", "figures", "volumes"):
        (run_dir / child).mkdir(parents=True, exist_ok=False)

    resolved_config = {
        **config,
        "analysis_id": analysis_id,
        "created_at_utc": created_at.isoformat(),
        "data_root": str(data_root.resolve()),
        "animal_id": animal_id,
        "date": date,
        "scan_id": metadata.scan_id,
        "output_root": str(output_root.resolve()),
        "config_source": repo_relative(config_path),
        "figure_style_source": repo_relative(style_path),
    }
    write_json(run_dir / "config.snapshot.json", resolved_config)
    write_json(run_dir / "environment.json", collect_environment())

    frame_filter = config.get("frame_stability_filter", {"enabled": False})
    frame_filter_enabled = bool(frame_filter.get("enabled", False))
    if frame_filter_enabled:
        volume, source_metric_rows, page_count = mean_stack_frame_filtered(
            metadata,
            discard_initial_frames=int(config["discard_initial_frames_per_slice"]),
            drift_qc_frames=int(config["drift_qc_frames"]),
            mad_k=float(frame_filter["mad_k"]),
            min_keep_fraction=float(frame_filter["min_keep_fraction"]),
        )
    else:
        volume, source_metric_rows, page_count = mean_stack(
            metadata,
            discard_initial_frames=int(config["discard_initial_frames_per_slice"]),
            drift_qc_frames=int(config["drift_qc_frames"]),
        )

    cross_plane_config = config.get("cross_plane_registration", {"enabled": False})
    cross_plane_enabled = bool(cross_plane_config.get("enabled", False))
    cross_plane_rows: list[dict[str, Any]] | None = None
    if cross_plane_enabled:
        volume, cross_plane_rows = apply_cross_plane_registration(
            volume,
            metadata.z_positions_um,
            mad_k=float(cross_plane_config["mad_k"]),
            max_step_shift_px=float(cross_plane_config["max_step_shift_px"]),
            upsample_factor=int(cross_plane_config["upsample_factor"]),
        )
        cross_plane_metrics_path = run_dir / "tables" / "cross_plane_registration.csv"
        write_csv(cross_plane_metrics_path, cross_plane_rows)

    volume_path = (
        run_dir
        / "volumes"
        / f"{animal_id}_{date}_{metadata.scan_id}__{metadata.n_slices}_plane_mean_volume.ome.tif"
    )
    write_ome_volume(
        volume_path,
        volume,
        pixel_size_x_um=metadata.pixel_size_x_um,
        pixel_size_y_um=metadata.pixel_size_y_um,
        z_step_um=metadata.z_step_um,
    )

    source_metrics_path = run_dir / "tables" / "source_slice_metrics.csv"
    write_csv(source_metrics_path, source_metric_rows)

    display = config["display"]
    projections, projection_metadata = make_projections(
        volume,
        pixel_size_xy_um=float(
            np.mean((metadata.pixel_size_x_um, metadata.pixel_size_y_um))
        ),
        z_step_um=metadata.z_step_um,
        low_percentile=float(display["percentile_low"]),
        high_percentile=float(display["percentile_high"]),
        gamma=float(display["gamma"]),
        azimuth_deg=float(display["azimuth_deg"]),
        elevation_deg=float(display["elevation_deg"]),
    )
    figure_paths = save_projection_figure(
        run_dir / "figures",
        projections,
        style=style,
        profile_name=str(config["figure_profile"]),
        pixel_size_x_um=metadata.pixel_size_x_um,
        z_span_um=float(metadata.z_positions_um[-1] - metadata.z_positions_um[0]),
    )
    figure_paths.pop("font")
    figure_id = "fig_bench2p_zstack_oblique"

    drift_values = np.asarray(
        [row["early_to_late_shift_magnitude_px"] for row in source_metric_rows],
        dtype=float,
    )
    drift_p95 = float(np.percentile(drift_values, 95))
    drift_threshold = float(config["drift_qc_p95_threshold_px"])
    frame_filter_gate: dict[str, Any] = {"status": "pass", "enabled": frame_filter_enabled}
    if frame_filter_enabled:
        kept = np.asarray([row["frames_kept"] for row in source_metric_rows], dtype=float)
        dropped = np.asarray([row["frames_dropped"] for row in source_metric_rows], dtype=float)
        kept_corr_min = np.asarray(
            [row["kept_frame_corr_min"] for row in source_metric_rows], dtype=float
        )
        dropped_corr_max = np.asarray(
            [row["dropped_frame_corr_max"] for row in source_metric_rows], dtype=float
        )
        frame_filter_gate.update(
            {
                "mad_k": float(frame_filter["mad_k"]),
                "min_keep_fraction": float(frame_filter["min_keep_fraction"]),
                "frames_kept_mean": float(np.mean(kept)),
                "frames_dropped_mean": float(np.mean(dropped)),
                "kept_frame_corr_min_p05": float(np.percentile(kept_corr_min, 5)),
                "dropped_frame_corr_max_p95": float(np.percentile(dropped_corr_max, 95)),
                "note": (
                    "Per Z slice, the frames_dropped raw frames with the lowest normalized "
                    "cross-correlation to that slice's own per-pixel median frame were "
                    "excluded before averaging. This targets within-slice (~single Z plane) "
                    "XY jitter/instability, not "
                    "cross-plane axial drift over the full stack acquisition."
                ),
            }
        )
    cross_plane_gate: dict[str, Any] = {"status": "pass", "enabled": cross_plane_enabled}
    if cross_plane_enabled and cross_plane_rows is not None:
        step_rows = cross_plane_rows[1:]
        applied_flags = np.asarray([row["step_applied"] for row in step_rows], dtype=bool)
        candidate_mag = np.asarray(
            [row["step_candidate_shift_magnitude_px"] for row in step_rows], dtype=float
        )
        cumulative_mag = np.asarray(
            [row["cumulative_shift_magnitude_px"] for row in cross_plane_rows], dtype=float
        )
        cross_plane_gate.update(
            {
                "mad_k": float(cross_plane_config["mad_k"]),
                "max_step_shift_px": float(cross_plane_config["max_step_shift_px"]),
                "steps_applied": int(applied_flags.sum()),
                "steps_rejected": int((~applied_flags).sum()),
                "step_candidate_shift_p95_px": float(np.percentile(candidate_mag, 95)),
                "cumulative_shift_max_px": float(np.max(cumulative_mag)),
                "note": (
                    "Sequential rigid XY registration between consecutive raw Z planes, "
                    "summed into a per-plane cumulative shift and applied before writing "
                    "the volume. Targets cross-plane XY drift accumulated over the whole "
                    "stack acquisition; steps_rejected planes were treated as zero "
                    "relative motion (low-correlation MAD outlier or candidate shift over "
                    "max_step_shift_px) rather than trusting a spurious estimate. Does not "
                    "estimate or correct genuine axial (true-Z) displacement."
                ),
            }
        )
    qc = {
        "schema_version": "labgraph.bench2p_zstack_single.qc.v1",
        "overall_status": "pass" if drift_p95 <= drift_threshold else "needs_review",
        "gates": {
            "setup_is_bench2p": {"status": "pass"},
            "slow_stack_enabled": {"status": "pass"},
            "tiff_page_count_matches_metadata": {
                "status": "pass",
                "observed": {metadata.scan_id: page_count},
            },
            "within_slice_early_late_drift": {
                "status": "pass" if drift_p95 <= drift_threshold else "needs_review",
                "p95_px": drift_p95,
                "max_px": float(np.max(drift_values)),
                "threshold_p95_px": drift_threshold,
                "note": "Descriptive first-N versus last-N frame rigid shift; frames were not motion-corrected in this mean volume.",
            },
            "frame_stability_filter": frame_filter_gate,
            "cross_plane_registration": cross_plane_gate,
        },
    }
    write_json(run_dir / "qc.json", qc)

    if frame_filter_enabled:
        mean_kept = float(np.mean([row["frames_kept"] for row in source_metric_rows]))
        aggregation_description = (
            f"arithmetic mean of, on average, {mean_kept:.0f} of "
            f"{metadata.frames_per_slice} sequential frames per source slice "
            f"(frames whose normalized cross-correlation to that slice's own median "
            f"frame was not a low outlier, MAD-k={float(frame_filter['mad_k']):.1f})"
        )
    else:
        aggregation_description = (
            f"arithmetic mean of {metadata.frames_per_slice} sequential frames per source slice"
        )
    if cross_plane_enabled and cross_plane_rows is not None:
        steps_applied = int(sum(row["step_applied"] for row in cross_plane_rows[1:]))
        aggregation_description += (
            f", followed by cross-plane cumulative XY drift registration "
            f"({steps_applied}/{metadata.n_slices - 1} plane-to-plane steps applied, "
            f"max cumulative shift "
            f"{max(row['cumulative_shift_magnitude_px'] for row in cross_plane_rows):.2f} px)"
        )

    print(f"[{metadata.scan_id}] hashing source TIFF", flush=True)
    source_record = {
        **asdict(metadata),
        "z_positions_um": list(metadata.z_positions_um),
        "sha256": sha256_file(source_path),
        "page_count_verified": page_count,
        "source_role": "raw_bench2p_slow_z_stack_single_full_range",
        "raw_access_purpose": "slice averaging, source QC, and saturation sampling",
    }
    sources_manifest_path = run_dir / "sources.json"
    write_json(
        sources_manifest_path,
        {
            "schema_version": "labgraph.bench2p_zstack.sources.v1",
            "source_of_truth_root": str(data_root.resolve()),
            "read_only": True,
            "sources": [source_record],
        },
    )

    plot_data_path = run_dir / "figures" / f"{figure_id}_plot_data.csv"
    write_csv(plot_data_path, source_metric_rows)
    figure_json_path = run_dir / "figures" / f"{figure_id}.json"
    figure_md_path = run_dir / "figures" / f"{figure_id}.md"
    figure_metadata = {
        "figure_id": figure_id,
        "title": f"{animal_id} bench2p single-stack averaged Z volume and oblique projection",
        "created_at_utc": created_at.isoformat(),
        "analysis_id": analysis_id,
        "analysis_manifest": repo_relative(run_dir / "analysis_manifest.json"),
        "generating_code": code_records(),
        "git": git_state(),
        "config": {
            "path": repo_relative(run_dir / "config.snapshot.json"),
            "sha256": sha256_file(run_dir / "config.snapshot.json"),
        },
        "style": {
            "path": repo_relative(style_path),
            "sha256": sha256_file(style_path),
            "profile": config["figure_profile"],
        },
        "input_sources_manifest": {
            "path": repo_relative(sources_manifest_path),
            "sha256": sha256_file(sources_manifest_path),
        },
        "plotted_data": {
            "path": repo_relative(plot_data_path),
            "sha256": sha256_file(plot_data_path),
            "sample_unit": "averaged Z slice",
            "n": int(volume.shape[0]),
        },
        "panels": {
            "a": {
                "computation": "maximum intensity over Z of the display-normalized mean volume"
            },
            "b": {
                "computation": "maximum intensity over Y after physically isotropic Z resampling"
            },
            "c": {"computation": projection_metadata["projection_method"], **projection_metadata},
        },
        "eligibility": "One readable bench2p slow stack spanning the full acquired Z range.",
        "exclusions": "Non-stack bench2p scans and same-day mini2p2 acquisitions were excluded by exact scan ID selection.",
        "missing_data": "None; a single stack has no boundary merge or shift borders.",
        "statistical_method": "none; descriptive exploratory visualization",
        "outputs": {
            kind: {"path": repo_relative(path), "sha256": sha256_file(path)}
            for kind, path in figure_paths.items()
        },
        "supersedes": None,
        "superseded_by": None,
    }
    write_json(figure_json_path, figure_metadata)
    figure_md_path.write_text(
        f"""# {figure_id}

## Question

Can one complete bench2p slow stack be reconstructed as a static averaged fluorescence volume and viewed from an oblique angle?

## Panels

- **a:** axial maximum projection of the {metadata.n_slices}-plane mean volume.
- **b:** side maximum projection after resampling Z to the measured XY pixel scale.
- **c:** oblique maximum projection after {display['azimuth_deg']}° azimuth and {display['elevation_deg']}° elevation rotations.

Each source slice is the {aggregation_description}. Unlike the two-stack workflow, there is no boundary merge: this stack already spans the full acquired Z range on its own.

## Sample And Eligibility

The sample unit is an averaged Z slice (`n = {volume.shape[0]}`). Only the one explicitly selected bench2p slow stack was eligible. Same-day single-plane bench2p tests and mini2p2 behavior acquisitions were not included.

## Supported Observation

The file supports a continuous exploratory fluorescence volume spanning approximately {metadata.z_positions_um[-1] - metadata.z_positions_um[0]:.1f} µm in Z, subject to the QC and display choices recorded in this run.

## Provisional Interpretation And Limits

This is a sequential static Z-stack reconstruction, not time-resolved volumetric imaging. Display percentiles, gamma, maximum projections, and rotation affect visibility and must not be interpreted as quantitative fluorescence normalization. {"Cross-plane registration corrects cumulative XY translation between consecutive planes only; it does not estimate or correct genuine axial (true-Z) displacement." if cross_plane_enabled else "No cross-plane (axial) motion correction is applied."} The early-versus-late within-slice drift QC is reported separately and is descriptive only.

## Reproduction

- Plot data: `{repo_relative(plot_data_path)}`
- Volume: `{repo_relative(volume_path)}`
- Source manifest: `{repo_relative(sources_manifest_path)}`
- Config snapshot: `{repo_relative(run_dir / 'config.snapshot.json')}`
- Analysis manifest: `{repo_relative(run_dir / 'analysis_manifest.json')}`
- Generating code: `{RUNNER_PATH.as_posix()}`, `{MODULE_PATH.as_posix()}`
""",
        encoding="utf-8",
    )

    readme_path = run_dir / "README.md"
    readme_path.write_text(
        f"""# {analysis_id}

Exploratory LabGraph bench2p single-stack Z reconstruction for `{animal_id}` on `{date}`.

- Input: `{metadata.scan_id}` (one complete slow stack; no boundary merge)
- Aggregation: {aggregation_description}
- Output: {volume.shape[0]} × {volume.shape[1]} × {volume.shape[2]} float32 OME-TIFF
- QC status: `{qc['overall_status']}`

See `analysis_manifest.json`, `sources.json`, `qc.json`, `config.snapshot.json`, `tables/`, `volumes/`, and `figures/` for the complete handoff.
""",
        encoding="utf-8",
    )

    manifest_candidates = sorted(
        path
        for path in run_dir.rglob("*")
        if path.is_file() and path.name != "analysis_manifest.json"
    )
    manifest = {
        "schema_version": "labgraph.analysis_manifest.v1",
        "analysis_id": analysis_id,
        "title": "bench2p single-stack Z slice averaging and oblique visualization",
        "question": "Can one complete bench2p slow stack form a traceable static fluorescence volume?",
        "created_at_utc": created_at.isoformat(),
        "analysis_state": config["analysis_state"],
        "analysis_unit": "averaged Z slice",
        "n": int(volume.shape[0]),
        "grouping_variables": ["animal_id", "date", "scan_id", "z_um"],
        "missing_data_policy": "None; a single full-range stack has no boundary merge.",
        "statistical_method": "none; descriptive reconstruction and QC",
        "eligibility": [
            {
                "animal_id": animal_id,
                "date": date,
                "scan_id": metadata.scan_id,
                "status": "included",
            }
        ],
        "exclusions": [
            "all unselected same-day bench2p single-plane/test scans",
            "all same-day mini2p2 behavior acquisitions",
        ],
        "inputs": {
            "sources_manifest": {
                "path": repo_relative(sources_manifest_path),
                "sha256": sha256_file(sources_manifest_path),
            }
        },
        "qc": {
            "path": repo_relative(run_dir / "qc.json"),
            "sha256": sha256_file(run_dir / "qc.json"),
            "status": qc["overall_status"],
        },
        "code": code_records(),
        "git": git_state(),
        "configuration": {
            "source_path": repo_relative(config_path),
            "source_sha256": sha256_file(config_path),
            "snapshot_path": repo_relative(run_dir / "config.snapshot.json"),
            "snapshot_sha256": sha256_file(run_dir / "config.snapshot.json"),
            "figure_style_path": repo_relative(style_path),
            "figure_style_sha256": sha256_file(style_path),
        },
        "environment": {
            "path": repo_relative(run_dir / "environment.json"),
            "sha256": sha256_file(run_dir / "environment.json"),
        },
        "random_seed": config["random_seed"],
        "outputs": [
            {
                "path": repo_relative(path),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in manifest_candidates
        ],
        "supersedes": None,
        "superseded_by": None,
        "supported_interpretations": [
            "The selected TIFF forms a spatially continuous exploratory static fluorescence Z stack on its own, without a boundary merge.",
            *(
                [
                    "Cumulative XY drift measured between consecutive raw Z planes was estimated and applied before writing this volume; see qc.json 'cross_plane_registration' and tables/cross_plane_registration.csv for the per-step trajectory and which steps were rejected as unreliable."
                ]
                if cross_plane_enabled
                else []
            ),
        ],
        "unsupported_interpretations": [
            "This run does not establish time-resolved volumetric activity, cell identity, anatomical connectivity, or a biological group effect.",
            *(
                [
                    "The per-slice frame-stability filter removes frames with large rigid shift relative to that slice's own median frame within its ~frames_per_slice/frame_rate acquisition window; it does not measure or correct axial (Z) drift across the whole sequential stack, so it is not guaranteed to remove any Z-elongated smearing of the same structure across multiple Z planes."
                ]
                if frame_filter_enabled
                else []
            ),
            *(
                [
                    "Cross-plane registration corrects cumulative XY translation between consecutive planes only; it does not estimate or correct genuine axial (true-Z) displacement, and steps whose plane-to-plane correlation was too low or whose candidate shift was implausibly large were left uncorrected (treated as zero motion) rather than force-aligned, so residual Z-elongated smearing is still possible."
                ]
                if cross_plane_enabled
                else []
            ),
        ],
    }
    write_json(run_dir / "analysis_manifest.json", manifest)
    print(f"Completed {analysis_id}: {run_dir}", flush=True)
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ["LABGRAPH_DATA_ROOT"])
        if os.environ.get("LABGRAPH_DATA_ROOT")
        else None,
        help="External uploaded-session root; may also be LABGRAPH_DATA_ROOT.",
    )
    parser.add_argument("--animal", required=True, dest="animal_id")
    parser.add_argument("--date", required=True)
    parser.add_argument("--scan", required=True, dest="scan_id")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--style", type=Path, default=DEFAULT_STYLE)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "datasets" / "analysis_runs",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.data_root is None:
        raise SystemExit("--data-root or LABGRAPH_DATA_ROOT is required")
    build_run(
        data_root=args.data_root.expanduser(),
        animal_id=args.animal_id,
        date=args.date,
        scan_id=args.scan_id,
        config_path=args.config.expanduser(),
        style_path=args.style.expanduser(),
        output_root=args.output_root.expanduser(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
