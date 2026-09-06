"""Run and record the LabGraph bench2p Z-stack reconstruction workflow."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .bench2p_zstack import (
    DEFAULT_CONFIG,
    DEFAULT_STYLE,
    MODULE_PATH,
    REPO_ROOT,
    combined_slice_rows,
    collect_environment,
    discover_stack_tif,
    git_state,
    make_projections,
    mean_stack,
    merge_adjacent_stacks,
    parse_stack_metadata,
    read_json,
    repo_relative,
    save_projection_figure,
    sha256_file,
    stable_digest,
    validate_stack_set,
    write_csv,
    write_json,
    write_ome_volume,
)


RUNNER_PATH = Path("src/labgraph_ops/workflows/bench2p_zstack_run.py")


def code_records() -> list[dict[str, str]]:
    return [
        {"path": path.as_posix(), "sha256": sha256_file(REPO_ROOT / path)}
        for path in (RUNNER_PATH, MODULE_PATH)
    ]


def build_run(
    *,
    data_root: Path,
    animal_id: str,
    date: str,
    scan_ids: Sequence[str],
    config_path: Path,
    style_path: Path,
    output_root: Path,
) -> Path:
    config = read_json(config_path)
    style = read_json(style_path)
    sources = [discover_stack_tif(data_root, animal_id, date, scan) for scan in scan_ids]
    metadata = [parse_stack_metadata(path) for path in sources]
    metadata.sort(key=lambda item: item.z_positions_um[0])
    validate_stack_set(metadata)

    identity = {
        "config": config,
        "sources": [
            {
                "path": str(path.resolve()),
                "size": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
            }
            for path in sources
        ],
    }
    created_at = datetime.now(timezone.utc)
    analysis_id = (
        "bench2p_zstack__"
        + created_at.strftime("%Y%m%dT%H%M%SZ")
        + "__"
        + stable_digest(identity)
    )
    run_dir = output_root.resolve() / analysis_id
    if run_dir.exists():
        raise FileExistsError(f"Immutable run already exists: {run_dir}")
    for child in ("tables", "figures", "volumes", "logs"):
        (run_dir / child).mkdir(parents=True, exist_ok=False)

    resolved_config = {
        **config,
        "analysis_id": analysis_id,
        "created_at_utc": created_at.isoformat(),
        "data_root": str(data_root.resolve()),
        "animal_id": animal_id,
        "date": date,
        "scan_ids": [item.scan_id for item in metadata],
        "output_root": str(output_root.resolve()),
        "config_source": repo_relative(config_path),
        "figure_style_source": repo_relative(style_path),
    }
    write_json(run_dir / "config.snapshot.json", resolved_config)
    write_json(run_dir / "environment.json", collect_environment())

    volumes: list[np.ndarray] = []
    source_metric_rows: list[dict[str, Any]] = []
    page_counts: dict[str, int] = {}
    volume_paths: dict[str, Path] = {}
    for item in metadata:
        volume, rows, page_count = mean_stack(
            item,
            discard_initial_frames=int(config["discard_initial_frames_per_slice"]),
            drift_qc_frames=int(config["drift_qc_frames"]),
        )
        volumes.append(volume)
        source_metric_rows.extend(rows)
        page_counts[item.scan_id] = page_count
        volume_path = run_dir / "volumes" / f"{item.scan_id}__slice_mean.ome.tif"
        write_ome_volume(
            volume_path,
            volume,
            pixel_size_x_um=item.pixel_size_x_um,
            pixel_size_y_um=item.pixel_size_y_um,
            z_step_um=item.z_step_um,
        )
        volume_paths[item.scan_id] = volume_path

    merged, z_positions, merge_qc = merge_adjacent_stacks(
        volumes[0],
        metadata[0],
        volumes[1],
        metadata[1],
        align_boundary_xy=bool(config["align_stack_boundary_xy"]),
        upsample_factor=int(config["boundary_upsample_factor"]),
        overlap_tolerance_um=float(config["boundary_overlap_tolerance_um"]),
        max_shift_px=float(config["boundary_max_shift_px"]),
    )
    z_step_um = float(np.median(np.diff(z_positions)))
    merged_path = run_dir / "volumes" / f"{animal_id}_{date}__101_plane_mean_volume.ome.tif"
    write_ome_volume(
        merged_path,
        merged,
        pixel_size_x_um=metadata[0].pixel_size_x_um,
        pixel_size_y_um=metadata[0].pixel_size_y_um,
        z_step_um=z_step_um,
    )
    volume_paths["merged"] = merged_path

    source_metrics_path = run_dir / "tables" / "source_slice_metrics.csv"
    combined_rows = combined_slice_rows(merged, z_positions, metadata[0], metadata[1])
    combined_metrics_path = run_dir / "tables" / "combined_slice_metrics.csv"
    write_csv(source_metrics_path, source_metric_rows)
    write_csv(combined_metrics_path, combined_rows)

    display = config["display"]
    projections, projection_metadata = make_projections(
        merged,
        pixel_size_xy_um=float(
            np.mean((metadata[0].pixel_size_x_um, metadata[0].pixel_size_y_um))
        ),
        z_step_um=z_step_um,
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
        pixel_size_x_um=metadata[0].pixel_size_x_um,
        z_span_um=float(z_positions[-1] - z_positions[0]),
    )
    figure_paths.pop("font")

    drift_values = np.asarray(
        [row["early_to_late_shift_magnitude_px"] for row in source_metric_rows], dtype=float
    )
    drift_p95 = float(np.percentile(drift_values, 95))
    drift_threshold = float(config["drift_qc_p95_threshold_px"])
    qc = {
        "schema_version": "labgraph.bench2p_zstack.qc.v1",
        "overall_status": "pass" if drift_p95 <= drift_threshold else "needs_review",
        "gates": {
            "setup_is_bench2p": {"status": "pass"},
            "slow_stack_enabled": {"status": "pass"},
            "tiff_page_count_matches_metadata": {
                "status": "pass",
                "observed": page_counts,
            },
            "stack_geometry_matches": {"status": "pass"},
            "single_boundary_plane_overlaps": {
                "status": "pass",
                "observed_gap_um": merge_qc["boundary_gap_um"],
                "tolerance_um": config["boundary_overlap_tolerance_um"],
            },
            "within_slice_early_late_drift": {
                "status": "pass" if drift_p95 <= drift_threshold else "needs_review",
                "p95_px": drift_p95,
                "max_px": float(np.max(drift_values)),
                "threshold_p95_px": drift_threshold,
                "note": "Descriptive first-N versus last-N frame rigid shift; frames were not motion-corrected in this v1 mean volume.",
            },
        },
        "merge": merge_qc,
    }
    write_json(run_dir / "qc.json", qc)

    source_records = []
    for item in metadata:
        source_path = Path(item.source_path)
        print(f"[{item.scan_id}] hashing source TIFF", flush=True)
        source_records.append(
            {
                **asdict(item),
                "z_positions_um": list(item.z_positions_um),
                "sha256": sha256_file(source_path),
                "page_count_verified": page_counts[item.scan_id],
                "source_role": "raw_bench2p_slow_z_stack",
                "raw_access_purpose": "slice averaging, source QC, and adjacent-stack reconstruction",
            }
        )
    sources_manifest_path = run_dir / "sources.json"
    write_json(
        sources_manifest_path,
        {
            "schema_version": "labgraph.bench2p_zstack.sources.v1",
            "source_of_truth_root": str(data_root.resolve()),
            "read_only": True,
            "sources": source_records,
        },
    )

    plot_data_path = run_dir / "figures" / "fig_bench2p_zstack_oblique_plot_data.csv"
    write_csv(plot_data_path, combined_rows)
    figure_id = "fig_bench2p_zstack_oblique"
    figure_json_path = run_dir / "figures" / f"{figure_id}.json"
    figure_md_path = run_dir / "figures" / f"{figure_id}.md"
    figure_metadata = {
        "figure_id": figure_id,
        "title": "ROS-2335 bench2p averaged Z stack and oblique projection",
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
            "n": int(merged.shape[0]),
        },
        "panels": {
            "a": {
                "computation": "maximum intensity over Z of the display-normalized combined mean volume"
            },
            "b": {
                "computation": "maximum intensity over Y after physically isotropic Z resampling"
            },
            "c": {"computation": projection_metadata["projection_method"], **projection_metadata},
        },
        "eligibility": "Two readable bench2p slow stacks with matched geometry and one overlapping boundary plane.",
        "exclusions": "Non-stack bench2p scans and same-day mini2p2 behavior acquisitions were excluded by exact scan ID selection.",
        "missing_data": "NaN borders introduced by the second-stack XY shift are ignored in the overlap mean and rendered as background.",
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

Can the two adjacent 51-slice bench2p acquisitions be reconstructed as one static averaged fluorescence volume and viewed from an oblique angle?

## Panels

- **a:** axial maximum projection of the 101-plane mean volume.
- **b:** side maximum projection after resampling Z to the measured XY pixel scale.
- **c:** oblique maximum projection after {display['azimuth_deg']}° azimuth and {display['elevation_deg']}° elevation rotations.

Each source slice is the arithmetic mean of {metadata[0].frames_per_slice} sequential frames. The final boundary plane is the mean of the first stack's last plane and the XY-aligned second stack's first plane.

## Sample And Eligibility

The sample unit is an averaged Z slice (`n = {merged.shape[0]}`). Only the two explicitly selected bench2p slow stacks were eligible. Same-day single-plane bench2p tests and mini2p2 behavior acquisitions were not included.

## Supported Observation

The files support a continuous exploratory fluorescence volume spanning approximately {z_positions[-1] - z_positions[0]:.1f} µm in Z, subject to the QC and display choices recorded in this run.

## Provisional Interpretation And Limits

This is a sequential static Z-stack reconstruction, not time-resolved volumetric imaging. Display percentiles, gamma, maximum projections, and rotation affect visibility and must not be interpreted as quantitative fluorescence normalization. Version 1 averages raw frames without within-slice motion correction; the early-versus-late drift QC is reported separately.

## Reproduction

- Plot data: `{repo_relative(plot_data_path)}`
- Combined volume: `{repo_relative(merged_path)}`
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

Exploratory LabGraph bench2p Z-stack reconstruction for `{animal_id}` on `{date}`.

- Inputs: `{metadata[0].scan_id}`, `{metadata[1].scan_id}`
- Aggregation: arithmetic mean of {metadata[0].frames_per_slice} frames per source slice
- Merge: one overlapping boundary plane, with rigid XY alignment of the second stack
- Output: {merged.shape[0]} × {merged.shape[1]} × {merged.shape[2]} float32 OME-TIFF
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
        "title": "bench2p sequential Z-stack slice averaging and oblique visualization",
        "question": "Can two adjacent 51-slice bench2p stacks form one traceable static fluorescence volume?",
        "created_at_utc": created_at.isoformat(),
        "analysis_state": config["analysis_state"],
        "analysis_unit": "averaged Z slice",
        "n": int(merged.shape[0]),
        "grouping_variables": ["animal_id", "date", "scan_id", "z_um"],
        "missing_data_policy": "NaN shift borders retained in the mean volume; ignored only for overlap aggregation and display.",
        "statistical_method": "none; descriptive reconstruction and QC",
        "eligibility": [
            {
                "animal_id": animal_id,
                "date": date,
                "scan_id": item.scan_id,
                "status": "included",
            }
            for item in metadata
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
            "The selected TIFFs form a spatially continuous exploratory static fluorescence Z stack after one boundary-plane merge."
        ],
        "unsupported_interpretations": [
            "This run does not establish time-resolved volumetric activity, cell identity, anatomical connectivity, or a biological group effect."
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
    parser.add_argument("--scan", action="append", dest="scan_ids", required=True)
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
        scan_ids=args.scan_ids,
        config_path=args.config.expanduser(),
        style_path=args.style.expanduser(),
        output_root=args.output_root.expanduser(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
