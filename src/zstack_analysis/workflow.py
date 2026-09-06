"""Provenance-complete Z-stack reconstruction workflow governed by LabGraph."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import asdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np

from .pipeline import (
    METHOD_STATEMENT,
    MOTION_STATEMENT,
    _display_transform,
    _page_level_z_values,
    adjacent_plane_rows,
    collect_environment,
    page_grouping_qc,
    physical_projections,
    read_json,
    reconstruct_median_volume,
    resolve_source_channel,
    sha256_file,
    stable_digest,
    validate_z_geometry,
    write_json,
    write_rows,
)
from .scanimage_io import parse_stack_metadata, write_ome_volume
from .version import PIPELINE_ID, PIPELINE_STAGE, PIPELINE_VERSION


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "median_projection.json"
DEFAULT_ANALYSIS_SPEC = PROJECT_ROOT / "config" / "fov_recovery_analysis.json"
DEFAULT_FIGURE_STYLE = PROJECT_ROOT / "config" / "figure_style.json"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "analysis_runs"
DEFAULT_FAILED_ROOT = PROJECT_ROOT / "failed_runs"
DEFAULT_ACQUISITION_SPEC_NAME = "zstack_acquisition.json"
ENTRY_SCRIPT = PROJECT_ROOT / "scripts" / "reconstruct_median_stack.py"


def repository_path(path: Path) -> str:
    """Use portable repository-relative locations for repository-owned files."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def validate_analysis_spec(spec: dict[str, Any]) -> None:
    required = (
        "schema_version",
        "spec_id",
        "title",
        "analysis_state",
        "primary_question",
        "analysis_units",
        "input_boundary",
        "eligibility",
        "missing_data_policy",
        "statistical_method",
        "supported_interpretations",
        "unsupported_interpretations",
        "pipeline",
    )
    missing = [key for key in required if key not in spec]
    if missing:
        raise ValueError(f"Analysis spec is missing required fields: {missing}")
    expected_pipeline = {
        "id": PIPELINE_ID,
        "version": PIPELINE_VERSION,
        "stage": PIPELINE_STAGE,
    }
    if spec["pipeline"] != expected_pipeline:
        raise ValueError(
            "Analysis spec pipeline identity does not match the running code: "
            f"expected {expected_pipeline}, found {spec['pipeline']}"
        )
    if spec["analysis_state"] not in {"exploratory", "confirmatory"}:
        raise ValueError("analysis_state must be exploratory or confirmatory")


def validate_figure_style(style: dict[str, Any], profile_name: str) -> None:
    if style.get("schema_version") != "labgraph.figure_style.v1":
        raise ValueError("Expected LabGraph figure style schema v1")
    if profile_name not in style.get("profiles", {}):
        raise ValueError(f"Unknown figure style profile: {profile_name}")
    output = style.get("output", {})
    if "svg" not in output.get("editable_vector_formats", []):
        raise ValueError("Figure style must require editable SVG output")
    if int(output.get("png_dpi", 0)) < 300:
        raise ValueError("Figure style must require PNG output at >=300 dpi")


def configure_matplotlib(style: dict[str, Any], profile_name: str) -> None:
    profile = style["profiles"][profile_name]
    font = style["font"]
    axes = style["axes"]
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [font["family"], *font.get("fallbacks", [])],
            "font.size": float(profile["font_size_pt"]),
            "axes.titlesize": float(profile["font_size_pt"]),
            "axes.labelsize": float(profile["font_size_pt"]),
            "axes.linewidth": float(profile["axes_linewidth_pt"]),
            "axes.spines.top": bool(axes["show_top_spine"]),
            "axes.spines.right": bool(axes["show_right_spine"]),
            "xtick.direction": str(axes["tick_direction"]),
            "ytick.direction": str(axes["tick_direction"]),
            "axes.grid": bool(axes["default_grid"]),
            "figure.facecolor": str(axes["background"]),
            "axes.facecolor": str(axes["background"]),
            "savefig.facecolor": str(axes["background"]),
            "svg.fonttype": str(font["svg_fonttype"]),
            "pdf.fonttype": int(font["pdf_fonttype"]),
        }
    )


def collect_git_state() -> dict[str, Any]:
    def run(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

    branch_result = run("branch", "--show-current")
    status_result = run("status", "--porcelain", "--untracked-files=all")
    commit_result = run("rev-parse", "HEAD")
    commit = commit_result.stdout.strip() if commit_result.returncode == 0 else None
    return {
        "repository": "https://github.com/Gyepee/ZStackAnalysis",
        "commit": commit,
        "branch": branch_result.stdout.strip() or None,
        "dirty": bool(status_result.stdout.strip()),
        "state": "committed" if commit is not None else "unborn",
    }


def _parse_iso_date(value: str, field: str) -> date:
    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must use YYYY-MM-DD: {value!r}") from error


def discover_acquisition_spec(
    source_tiff: Path, explicit_path: Path | None
) -> tuple[Path | None, dict[str, Any] | None]:
    candidate = (
        explicit_path.resolve()
        if explicit_path is not None
        else source_tiff.parent / DEFAULT_ACQUISITION_SPEC_NAME
    )
    if not candidate.exists():
        if explicit_path is not None:
            raise FileNotFoundError(f"Acquisition spec does not exist: {candidate}")
        return None, None
    return candidate, read_json(candidate)


def validate_acquisition_spec(
    declaration: dict[str, Any], declaration_path: Path, source_tiff: Path, metadata: Any
) -> dict[str, Any]:
    required = (
        "schema_version",
        "data_kind",
        "animal_id",
        "acquisition_date",
        "scan_id",
        "session_id",
        "source_tiff",
        "acquisition_purpose",
    )
    missing = [key for key in required if key not in declaration]
    if missing:
        raise ValueError(f"Acquisition spec is missing required fields: {missing}")
    if declaration["schema_version"] != "zstack_analysis.acquisition.v1":
        raise ValueError("Unsupported acquisition spec schema_version")
    if declaration["data_kind"] != "zstack":
        raise ValueError("Acquisition spec data_kind must be 'zstack'")
    expected = {
        "animal_id": metadata.animal_id,
        "acquisition_date": metadata.date,
        "scan_id": metadata.scan_id,
        "session_id": metadata.session_id,
    }
    conflicts = {
        key: {"declared": declaration[key], "tiff_metadata": value}
        for key, value in expected.items()
        if str(declaration[key]) != str(value)
    }
    source_value = Path(str(declaration["source_tiff"]))
    declared_source = (
        source_value.resolve()
        if source_value.is_absolute()
        else (declaration_path.parent / source_value).resolve()
    )
    if declared_source != source_tiff.resolve():
        conflicts["source_tiff"] = {
            "declared": str(declared_source),
            "selected": str(source_tiff.resolve()),
        }
    if conflicts:
        raise ValueError(f"Acquisition spec conflicts with TIFF identity: {conflicts}")
    acquisition_date = _parse_iso_date(str(declaration["acquisition_date"]), "acquisition_date")
    surgery_value = declaration.get("surgery_date")
    declared_day = declaration.get("post_surgery_day")
    computed_day: int | None = None
    if surgery_value is not None:
        surgery_date = _parse_iso_date(str(surgery_value), "surgery_date")
        computed_day = (acquisition_date - surgery_date).days
        if computed_day < 0:
            raise ValueError("acquisition_date precedes surgery_date")
        if declared_day is not None and int(declared_day) != computed_day:
            raise ValueError(
                "post_surgery_day disagrees with acquisition_date - surgery_date"
            )
    elif declared_day is not None:
        computed_day = int(declared_day)
    return {
        **declaration,
        "computed_post_surgery_day": computed_day,
        "validation_status": "pass",
    }


def resolve_acquisition_context(
    source_tiff: Path, metadata: Any, explicit_path: Path | None
) -> dict[str, Any]:
    path, declaration = discover_acquisition_spec(source_tiff, explicit_path)
    if declaration is None:
        return {
            "status": "legacy_metadata_incomplete",
            "path": None,
            "sha256": None,
            "declaration": None,
            "post_surgery_day": None,
            "reason": (
                f"No {DEFAULT_ACQUISITION_SPEC_NAME} was found; reconstruction is "
                "allowed, but the scan is not comparison-ready."
            ),
        }
    validated = validate_acquisition_spec(declaration, path, source_tiff, metadata)
    return {
        "status": "pass",
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "declaration": validated,
        "post_surgery_day": validated["computed_post_surgery_day"],
        "reason": None,
    }


def load_review_flags(source_tiff: Path, metadata: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    flag_root = PROJECT_ROOT / "review_flags"
    for path in sorted(flag_root.glob("*.json")):
        payload = read_json(path)
        subject = payload.get("subject", {})
        subject_matches = all(
            str(subject.get(key, "")) == str(value)
            for key, value in {
                "animal_id": metadata.animal_id,
                "date": metadata.date,
                "scan_id": metadata.scan_id,
            }.items()
        )
        source_matches = str(payload.get("source_tiff", "")) == str(source_tiff)
        if subject_matches or source_matches:
            records.append(
                {
                    "path": repository_path(path),
                    "sha256": sha256_file(path),
                    "status": payload.get("status"),
                    "reason_code": payload.get("reason_code"),
                    "interpretation": payload.get("interpretation"),
                    "recorded_at_utc": payload.get("recorded_at_utc"),
                }
            )
    return records


def choose_source_channel(
    requested_channel: int | None,
    acquisition_context: dict[str, Any],
    saved_channels: Sequence[int],
) -> tuple[int, int]:
    declared = None
    declaration = acquisition_context.get("declaration")
    if declaration is not None:
        declared = declaration.get("source_channel")
    if declared is not None and requested_channel is not None:
        if int(declared) != int(requested_channel):
            raise ValueError(
                f"CLI channel {requested_channel} conflicts with declared channel {declared}"
            )
    selected = requested_channel if requested_channel is not None else declared
    return resolve_source_channel(saved_channels, selected)


def identity_columns(
    analysis_id: str, metadata: Any, source_channel: int, post_surgery_day: int | None
) -> dict[str, Any]:
    return {
        "analysis_id": analysis_id,
        "pipeline_id": PIPELINE_ID,
        "pipeline_version": PIPELINE_VERSION,
        "animal_id": metadata.animal_id,
        "acquisition_date": metadata.date,
        "post_surgery_day": post_surgery_day,
        "scan_id": metadata.scan_id,
        "session_id": metadata.session_id,
        "source_channel": source_channel,
    }


def attach_identity(rows: list[dict[str, Any]], identity: dict[str, Any]) -> list[dict[str, Any]]:
    return [{**identity, **row} for row in rows]


def create_analysis_id(
    metadata: Any,
    source_channel: int,
    created_at: datetime,
    identity_digest_payload: dict[str, Any],
) -> str:
    version_slug = PIPELINE_VERSION.replace(".", "-")
    scope = (
        f"{metadata.animal_id}_{metadata.date}_{metadata.scan_id}_"
        f"{metadata.session_id}_ch{source_channel}"
    )
    digest = stable_digest(identity_digest_payload)
    return (
        f"{PIPELINE_ID}-v{version_slug}__{scope}__"
        f"{created_at.strftime('%Y%m%dT%H%M%SZ')}__{digest}"
    )


def _metadata_lines(
    metadata: Any,
    source_channel: int,
    analysis_id: str,
    post_surgery_day: int | None,
    review_status: str,
) -> tuple[str, str]:
    day_text = "post-op day unknown" if post_surgery_day is None else f"post-op day {post_surgery_day}"
    first = (
        f"{metadata.animal_id} | {metadata.date} | {metadata.scan_id} | "
        f"{metadata.session_id} | channel {source_channel} | {day_text}"
    )
    second = (
        f"{PIPELINE_ID} v{PIPELINE_VERSION} ({PIPELINE_STAGE}) | "
        f"{metadata.n_slices} planes × {metadata.frames_per_slice} frames | "
        f"Δz {metadata.z_step_um:g} µm | {review_status} | run {analysis_id[-8:]}"
    )
    return first, second


def _save_svg_png(figure: Any, base_path: Path, png_dpi: int) -> dict[str, Path]:
    svg_path = base_path.with_suffix(".svg")
    png_path = base_path.with_suffix(".png")
    figure.savefig(svg_path, bbox_inches="tight")
    figure.savefig(png_path, dpi=png_dpi, bbox_inches="tight")
    return {"svg": svg_path, "png": png_path}


def save_projection_figures(
    run_dir: Path,
    volume: np.ndarray,
    metadata: Any,
    config: dict[str, Any],
    style: dict[str, Any],
    analysis_id: str,
    source_channel: int,
    post_surgery_day: int | None,
    review_status: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
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
    representative_index = int(
        round(float(config["projection"]["representative_z_fraction"]) * (volume.shape[0] - 1))
    )
    arrays = {
        **{key: np.asarray(value, dtype=np.float32) for key, value in projections.items()},
        "representative_median_plane": np.asarray(volume[representative_index], dtype=np.float32),
    }
    array_path = run_dir / "figures" / "projection_source_arrays_float32.npz"
    np.savez_compressed(array_path, **arrays)

    mean_keys = ("mean_xy_over_z", "mean_xz_over_y", "mean_yz_over_x")
    mean_values = np.concatenate([projections[key].ravel() for key in mean_keys])
    mean_low, mean_high = np.percentile(
        mean_values,
        [float(display["low_percentile"]), float(display["high_percentile"])],
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
            (metadata.z_positions_um[-1] - metadata.z_positions_um[0]) + metadata.z_step_um,
            0,
        ],
        "yz_over_x": [
            0,
            metadata.height_px * metadata.pixel_size_y_um,
            (metadata.z_positions_um[-1] - metadata.z_positions_um[0]) + metadata.z_step_um,
            0,
        ],
    }
    col_specs = (
        ("xy_over_z", "XY; projected across Z"),
        ("xz_over_y", "XZ; projected across Y"),
        ("yz_over_x", "YZ; projected across X"),
    )
    identity_line, pipeline_line = _metadata_lines(
        metadata,
        source_channel,
        analysis_id,
        post_surgery_day,
        review_status,
    )
    png_dpi = int(style["output"]["png_dpi"])
    descriptors: list[dict[str, Any]] = []

    figure, axes = plt.subplots(2, 3, figsize=(11.2, 6.3), constrained_layout=True)
    panels = []
    for row_index, (prefix, row_label) in enumerate(
        (("mean", "mean projection"), ("max", "maximum projection"))
    ):
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
            panels.append(
                {
                    "panel": chr(ord("a") + len(panels)),
                    "array_key": key,
                    "computation": f"{prefix} across {suffix.rsplit('_', 1)[-1]} of the median-per-plane volume",
                    "variables": ["fluorescence intensity", "physical position"],
                    "sample_unit": "one Z-stack acquisition and channel",
                    "n": 1,
                }
            )
    figure.suptitle(f"{identity_line}\n{pipeline_line}\nmean and maximum projections")
    paths = _save_svg_png(
        figure,
        run_dir / "figures" / "fig_mean_vs_max_projections",
        png_dpi,
    )
    plt.close(figure)
    descriptors.append(
        {
            "figure_id": "fig_mean_vs_max_projections",
            "title": "Mean and maximum projections of the median-per-plane volume",
            "paths": paths,
            "array_keys": [panel["array_key"] for panel in panels],
            "panels": panels,
            "display_range": {"low": float(low), "high": float(high), "gamma": gamma},
            "observations": "The panels provide a descriptive view of one reconstructed scan under one shared display transform.",
        }
    )

    figure, axes = plt.subplots(1, 3, figsize=(11.2, 3.8), constrained_layout=True)
    panels = []
    for axis, (suffix, title) in zip(axes, col_specs, strict=True):
        key = f"mean_{suffix}"
        axis.imshow(
            _display_transform(projections[key], float(mean_low), float(mean_high), gamma),
            cmap=str(config["figure"]["cmap"]),
            vmin=0,
            vmax=1,
            extent=extent_by_column[suffix],
            aspect="equal",
        )
        axis.set_title(title)
        axis.set_xlabel("µm")
        axis.set_ylabel("µm")
        panels.append(
            {
                "panel": chr(ord("a") + len(panels)),
                "array_key": key,
                "computation": f"mean across {suffix.rsplit('_', 1)[-1]} of the median-per-plane volume",
                "variables": ["mean fluorescence intensity", "physical position"],
                "sample_unit": "one Z-stack acquisition and channel",
                "n": 1,
            }
        )
    figure.suptitle(f"{identity_line}\n{pipeline_line}\nmean projections")
    paths = _save_svg_png(
        figure,
        run_dir / "figures" / "fig_mean_projections",
        png_dpi,
    )
    plt.close(figure)
    descriptors.append(
        {
            "figure_id": "fig_mean_projections",
            "title": "Mean projections of the median-per-plane volume",
            "paths": paths,
            "array_keys": [panel["array_key"] for panel in panels],
            "panels": panels,
            "display_range": {"low": float(mean_low), "high": float(mean_high), "gamma": gamma},
            "observations": "The panels show mean intensity along each physical projection axis for one scan.",
        }
    )

    figure, axis = plt.subplots(figsize=(5.3, 5.5), constrained_layout=True)
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
            f"median XY plane {representative_index}; "
            f"z={metadata.z_positions_um[representative_index]:.3f} µm"
        ),
        xlabel="X (µm)",
        ylabel="Y (µm)",
    )
    figure.suptitle(f"{identity_line}\n{pipeline_line}")
    paths = _save_svg_png(
        figure,
        run_dir / "figures" / "fig_representative_median_plane",
        png_dpi,
    )
    plt.close(figure)
    descriptors.append(
        {
            "figure_id": "fig_representative_median_plane",
            "title": "Representative pixelwise-median XY plane",
            "paths": paths,
            "array_keys": ["representative_median_plane"],
            "panels": [
                {
                    "panel": "a",
                    "array_key": "representative_median_plane",
                    "computation": f"pixelwise median of all usable frames at plane {representative_index}",
                    "variables": ["fluorescence intensity", "X", "Y"],
                    "sample_unit": "one Z-stack acquisition and channel",
                    "n": 1,
                    "technical_frame_count": int(metadata.frames_per_slice),
                }
            ],
            "display_range": {"low": float(low), "high": float(high), "gamma": gamma},
            "observations": "The panel shows one configured representative depth and is not a depth-recovery endpoint.",
        }
    )

    rows = [
        {
            "projection": name,
            "aggregation_axis": name.rsplit("_", 1)[-1],
            "source_dtype": "float32 median volume",
            "minimum": float(np.nanmin(array)),
            "mean": float(np.nanmean(array)),
            "maximum": float(np.nanmax(array)),
        }
        for name, array in projections.items()
    ]
    display_metadata = {
        "low_intensity": float(low),
        "high_intensity": float(high),
        "low_percentile": float(display["low_percentile"]),
        "high_percentile": float(display["high_percentile"]),
        "gamma": gamma,
        "same_transform_for_mean_vs_max_panels": True,
        "mean_only_figure_low_intensity": float(mean_low),
        "mean_only_figure_high_intensity": float(mean_high),
        "mean_only_figure_uses_one_transform_for_all_three_panels": True,
        "representative_plane_index": representative_index,
    }
    return display_metadata, rows, descriptors


def _hashed_file(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(root)),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def write_figure_bundles(
    run_dir: Path,
    descriptors: list[dict[str, Any]],
    *,
    analysis_id: str,
    created_at: datetime,
    metadata: Any,
    source_channel: int,
    post_surgery_day: int | None,
    analysis_spec: dict[str, Any],
    analysis_spec_path: Path,
    analysis_config_path: Path,
    figure_style_path: Path,
    git_state: dict[str, Any],
    eligibility: dict[str, Any],
    review_flags: list[dict[str, Any]],
    supersedes: list[str],
) -> None:
    array_path = run_dir / "figures" / "projection_source_arrays_float32.npz"
    plot_data = {
        "path": str(array_path.relative_to(run_dir)),
        "sha256": sha256_file(array_path),
        "format": "compressed NPZ of exact float32 image arrays before display transform",
    }
    script = {
        "path": repository_path(ENTRY_SCRIPT),
        "sha256": sha256_file(ENTRY_SCRIPT),
    }
    input_id = (
        f"{metadata.animal_id}/{metadata.date}/{metadata.scan_id}/"
        f"{metadata.session_id}/channel-{source_channel}"
    )
    for descriptor in descriptors:
        figure_id = descriptor["figure_id"]
        outputs = {
            kind: _hashed_file(path, run_dir)
            for kind, path in descriptor["paths"].items()
        }
        payload = {
            "schema_version": "labgraph.figure_bundle.v1",
            "figure_id": figure_id,
            "title": descriptor["title"],
            "created_at_utc": created_at.isoformat(),
            "analysis_id": analysis_id,
            "pipeline": {
                "id": PIPELINE_ID,
                "version": PIPELINE_VERSION,
                "stage": PIPELINE_STAGE,
            },
            "question": analysis_spec["primary_question"],
            "identity": {
                "animal_id": metadata.animal_id,
                "acquisition_date": metadata.date,
                "post_surgery_day": post_surgery_day,
                "scan_id": metadata.scan_id,
                "session_id": metadata.session_id,
                "source_channel": source_channel,
            },
            "generating_script": script,
            "git": git_state,
            "configs": {
                "analysis_spec": {
                    "path": repository_path(analysis_spec_path),
                    "sha256": sha256_file(analysis_spec_path),
                },
                "analysis_config": {
                    "path": repository_path(analysis_config_path),
                    "sha256": sha256_file(analysis_config_path),
                },
                "figure_style": {
                    "path": repository_path(figure_style_path),
                    "sha256": sha256_file(figure_style_path),
                    "upstream_authority": "LabGraph/config/figure_style.json",
                },
            },
            "input_ids": [input_id],
            "source_manifest_references": [
                "analysis_manifest.json#inputs",
                "config.snapshot.json#acquisition_declaration",
            ],
            "exact_plotted_data": {**plot_data, "array_keys": descriptor["array_keys"]},
            "panels": descriptor["panels"],
            "analysis_units": analysis_spec["analysis_units"],
            "eligibility": eligibility,
            "exclusions": review_flags,
            "missing_data_handling": analysis_spec["missing_data_policy"],
            "statistical_method": analysis_spec["statistical_method"],
            "display_transform": descriptor["display_range"],
            "metadata_displayed_in_artwork": [
                "animal_id",
                "acquisition_date",
                "post_surgery_day",
                "scan_id",
                "session_id",
                "source_channel",
                "pipeline_id",
                "pipeline_version",
                "pipeline_stage",
                "plane_count",
                "frames_per_plane",
                "z_step_um",
                "review_status",
                "analysis_id_suffix",
            ],
            "outputs": outputs,
            "supersession": {"status": "current_at_creation", "supersedes": supersedes},
            "observations": descriptor["observations"],
            "safe_interpretation": analysis_spec["supported_interpretations"],
            "claims_not_made": analysis_spec["unsupported_interpretations"],
        }
        write_json(run_dir / "figures" / f"{figure_id}.json", payload)
        note = f"""# {descriptor['title']}

## Question

{analysis_spec['primary_question']}

## Calculation and observations

This figure was generated from `{plot_data['path']}` using array keys
`{', '.join(descriptor['array_keys'])}`. Each panel is described in
`{figure_id}.json`. The reconstruction method is: {METHOD_STATEMENT}.

## Sample unit and n

The displayed sample unit is one physical Z-stack acquisition from
`{metadata.animal_id}`, `{metadata.date}`, `{metadata.scan_id}`,
`{metadata.session_id}`, channel {source_channel}; n = 1 scan. Frames and planes
are nested technical observations, not independent biological replicates.

## Eligibility, exclusions, and missing data

Longitudinal comparison status: `{eligibility['longitudinal_comparison']['status']}`.
Post-surgery day: `{post_surgery_day}`. Source review flags: `{len(review_flags)}`.
Missing days are not interpolated and unavailable surgery dates remain explicit.

## Supported observation and provisional interpretation

{descriptor['observations']} This descriptive view can support later within-animal
comparison after longitudinal metadata and source review gates pass.

## Not established

This figure does not by itself identify the optimal mounting day, prove biological
recovery from brightness alone, or perform axial-motion correction.

## Reproduction

Use `analysis_manifest.json`, `config.snapshot.json`, `{figure_id}.json`, the exact
NPZ arrays above, and `{script['path']}`. SVG is the editable artwork and PNG is the
300 dpi review copy.
"""
        (run_dir / "figures" / f"{figure_id}.md").write_text(note, encoding="utf-8")


def build_eligibility(
    acquisition_context: dict[str, Any], review_flags: list[dict[str, Any]]
) -> dict[str, Any]:
    reasons: list[str] = []
    if acquisition_context["post_surgery_day"] is None:
        reasons.append("post_surgery_day_unknown")
    if review_flags:
        reasons.extend(
            f"unresolved_review_flag:{flag['reason_code']}" for flag in review_flags
        )
    status = "eligible" if not reasons else "hold"
    return {
        "reconstruction": {"status": "eligible", "reasons": []},
        "longitudinal_comparison": {"status": status, "reasons": reasons},
        "rule_source": "config.snapshot.json#analysis_spec/eligibility",
    }


def run_readme(
    *,
    analysis_id: str,
    metadata: Any,
    source_channel: int,
    post_surgery_day: int | None,
    analysis_spec: dict[str, Any],
    eligibility: dict[str, Any],
    review_flags: list[dict[str, Any]],
    command: list[str],
) -> str:
    return f"""# {analysis_id}

- Pipeline: `{PIPELINE_ID}` v`{PIPELINE_VERSION}` (`{PIPELINE_STAGE}`)
- Animal: `{metadata.animal_id}`
- Acquisition: `{metadata.date}` / `{metadata.scan_id}` / `{metadata.session_id}` / channel `{source_channel}`
- Post-surgery day: `{post_surgery_day}`
- Analysis state: `{analysis_spec['analysis_state']}`
- Reconstruction QC: `pass`
- Longitudinal comparison: `{eligibility['longitudinal_comparison']['status']}`
- Source review flags: `{len(review_flags)}`

## Question

{analysis_spec['primary_question']}

## Analysis unit

One physical Z-stack scan and one channel is the reconstruction unit. The animal is
the biological unit for longitudinal inference; days/scans are repeated observations
nested within animal, and frames/planes are technical observations.

## Method and interpretation boundary

{METHOD_STATEMENT}. {MOTION_STATEMENT}. This initial-development run is descriptive
and does not estimate the optimal mounting day.

## Reproduce

```bash
{shlex.join(command)}
```

See `analysis_manifest.json` for hashes and `figures/*.json` plus `figures/*.md` for
panel-level provenance and interpretation limits.
"""


def build_run(
    *,
    source_tiff: Path,
    read_copy: Path | None,
    config_path: Path,
    analysis_spec_path: Path,
    figure_style_path: Path,
    acquisition_spec_path: Path | None,
    output_root: Path,
    failed_root: Path,
    source_channel: int | None,
    supersedes: Sequence[str] = (),
    invocation: Sequence[str] | None = None,
) -> Path:
    source_tiff = source_tiff.resolve()
    config_path = config_path.resolve()
    analysis_spec_path = analysis_spec_path.resolve()
    figure_style_path = figure_style_path.resolve()
    config = read_json(config_path)
    analysis_spec = read_json(analysis_spec_path)
    style = read_json(figure_style_path)
    validate_analysis_spec(analysis_spec)
    profile_name = str(config["figure"]["style_profile"])
    validate_figure_style(style, profile_name)
    configure_matplotlib(style, profile_name)

    metadata = parse_stack_metadata(source_tiff)
    if metadata.setup != "bench2p" or not metadata.stack_enabled:
        raise ValueError("Input is not an enabled Bench2p ScanImage stack")
    acquisition_context = resolve_acquisition_context(
        source_tiff, metadata, acquisition_spec_path
    )
    source_channel, channel_position = choose_source_channel(
        source_channel, acquisition_context, metadata.saved_channels
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
    source_sha256 = sha256_file(source_tiff)
    read_copy_sha256 = None
    if read_copy is not None:
        read_copy_sha256 = sha256_file(read_path)
        if read_copy_sha256 != source_sha256:
            raise ValueError("read copy content hash differs from authoritative TIFF")

    review_flags = load_review_flags(source_tiff, metadata)
    eligibility = build_eligibility(acquisition_context, review_flags)
    review_status = "needs review" if review_flags else "QC pass"
    created_at = datetime.now(timezone.utc)
    digest_payload = {
        "source_sha256": source_sha256,
        "source_channel": source_channel,
        "analysis_config_sha256": sha256_file(config_path),
        "analysis_spec_sha256": sha256_file(analysis_spec_path),
        "figure_style_sha256": sha256_file(figure_style_path),
        "acquisition_spec_sha256": acquisition_context["sha256"],
    }
    analysis_id = create_analysis_id(
        metadata, source_channel, created_at, digest_payload
    )
    final_dir = output_root.resolve() / analysis_id
    incomplete_dir = failed_root.resolve() / f".{analysis_id}.incomplete"
    if final_dir.exists() or incomplete_dir.exists():
        raise FileExistsError(f"Run already exists: {analysis_id}")
    for child in ("volumes", "tables", "figures", "logs"):
        (incomplete_dir / child).mkdir(parents=True, exist_ok=True)

    command = list(invocation) if invocation is not None else [sys.executable, *sys.argv]
    git_state = collect_git_state()
    identity = identity_columns(
        analysis_id,
        metadata,
        source_channel,
        acquisition_context["post_surgery_day"],
    )
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
        write_rows(
            incomplete_dir / "tables" / "frame_to_median_qc.csv",
            attach_identity(frame_rows, identity),
        )
        write_rows(
            incomplete_dir / "tables" / "plane_median_qc.csv",
            attach_identity(plane_rows, identity),
        )
        write_rows(
            incomplete_dir / "tables" / "adjacent_plane_correlation.csv",
            attach_identity(adjacent_plane_rows(volume, metadata), identity),
        )
        display_metadata, projection_rows, figure_descriptors = save_projection_figures(
            incomplete_dir,
            volume,
            metadata,
            config,
            style,
            analysis_id,
            source_channel,
            acquisition_context["post_surgery_day"],
            review_status,
        )
        write_rows(
            incomplete_dir / "tables" / "projection_summary.csv",
            attach_identity(projection_rows, identity),
        )

        qc = {
            "overall_status": "needs_review" if review_flags else "pass",
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
            "source_review_flags": review_flags,
            "interpretation_limit": (
                "Median aggregation is robust to a minority of transient outlier "
                "frames but is not motion correction and cannot establish axial stability."
            ),
        }
        write_json(incomplete_dir / "qc.json", qc)
        environment = collect_environment()
        write_json(incomplete_dir / "environment.json", environment)
        resolved = {
            "schema_version": "zstack_analysis.resolved_config.v2",
            "analysis_id": analysis_id,
            "created_at_utc": created_at.isoformat(),
            "pipeline": {
                "id": PIPELINE_ID,
                "version": PIPELINE_VERSION,
                "stage": PIPELINE_STAGE,
            },
            "analysis_spec": analysis_spec,
            "analysis_config": config,
            "figure_style": style,
            "acquisition_declaration": acquisition_context,
            "authoritative_source_tiff": str(source_tiff),
            "local_read_copy_used": read_copy is not None,
            "selected_source_channel": source_channel,
            "source_channel_position": channel_position,
            "metadata": asdict(metadata),
            "display_resolved": display_metadata,
            "method_statement": METHOD_STATEMENT,
            "motion_statement": MOTION_STATEMENT,
            "random_seed": config.get("random_seed"),
        }
        write_json(incomplete_dir / "config.snapshot.json", resolved)
        write_json(
            incomplete_dir / "logs" / "validation.json",
            {
                "status": "pass",
                "created_at_utc": created_at.isoformat(),
                "analysis_spec": "pass",
                "figure_style": "pass",
                "acquisition_declaration": acquisition_context["status"],
                "source_identity": "pass",
                "source_hash": "pass",
                "read_copy_hash": "not_applicable" if read_copy is None else "pass",
                "metadata_gates": "pass",
                "review_flag_count": len(review_flags),
            },
        )
        write_figure_bundles(
            incomplete_dir,
            figure_descriptors,
            analysis_id=analysis_id,
            created_at=created_at,
            metadata=metadata,
            source_channel=source_channel,
            post_surgery_day=acquisition_context["post_surgery_day"],
            analysis_spec=analysis_spec,
            analysis_spec_path=analysis_spec_path,
            analysis_config_path=config_path,
            figure_style_path=figure_style_path,
            git_state=git_state,
            eligibility=eligibility,
            review_flags=review_flags,
            supersedes=list(supersedes),
        )
        (incomplete_dir / "README.md").write_text(
            run_readme(
                analysis_id=analysis_id,
                metadata=metadata,
                source_channel=source_channel,
                post_surgery_day=acquisition_context["post_surgery_day"],
                analysis_spec=analysis_spec,
                eligibility=eligibility,
                review_flags=review_flags,
                command=command,
            ),
            encoding="utf-8",
        )

        code_paths = (
            PROJECT_ROOT / "src" / "zstack_analysis" / "workflow.py",
            PROJECT_ROOT / "src" / "zstack_analysis" / "pipeline.py",
            PROJECT_ROOT / "src" / "zstack_analysis" / "scanimage_io.py",
            PROJECT_ROOT / "src" / "zstack_analysis" / "version.py",
            ENTRY_SCRIPT,
        )
        pre_manifest_files = sorted(
            path for path in incomplete_dir.rglob("*") if path.is_file()
        )
        manifest = {
            "schema_version": "zstack_analysis.run.v2",
            "analysis_id": analysis_id,
            "title": analysis_spec["title"],
            "question": analysis_spec["primary_question"],
            "created_at_utc": created_at.isoformat(),
            "analysis_state": analysis_spec["analysis_state"],
            "pipeline": {
                "id": PIPELINE_ID,
                "version": PIPELINE_VERSION,
                "stage": PIPELINE_STAGE,
                "method_statement": METHOD_STATEMENT,
                "motion_statement": MOTION_STATEMENT,
            },
            "identity": identity,
            "input_boundary": analysis_spec["input_boundary"],
            "inputs": {
                "raw_scanimage_tiff": {
                    "asset_id": (
                        f"{metadata.animal_id}/{metadata.date}/{metadata.scan_id}/"
                        f"{metadata.session_id}/{source_tiff.name}"
                    ),
                    "external_path": str(source_tiff),
                    "file_name": source_tiff.name,
                    "size_bytes": source_tiff.stat().st_size,
                    "mtime_ns": source_tiff.stat().st_mtime_ns,
                    "sha256": source_sha256,
                    "read_only": True,
                    "saved_channels": list(metadata.saved_channels),
                    "selected_channel": source_channel,
                },
                "local_read_copy": None
                if read_copy is None
                else {
                    "path": str(read_path),
                    "sha256": read_copy_sha256,
                    "verified_identical_to_authoritative_source": True,
                },
                "acquisition_declaration": {
                    "status": acquisition_context["status"],
                    "path": acquisition_context["path"],
                    "sha256": acquisition_context["sha256"],
                },
            },
            "eligibility": eligibility,
            "exclusions_and_review_flags": review_flags,
            "analysis_units": analysis_spec["analysis_units"],
            "grouping_variables": [
                "animal_id",
                "post_surgery_day",
                "acquisition_date",
                "scan_id",
                "session_id",
                "source_channel",
            ],
            "missing_data_policy": analysis_spec["missing_data_policy"],
            "statistical_method": analysis_spec["statistical_method"],
            "entry_script": {
                "path": repository_path(ENTRY_SCRIPT),
                "sha256": sha256_file(ENTRY_SCRIPT),
                "invocation": command,
            },
            "git": git_state,
            "code": [
                {"path": repository_path(path), "sha256": sha256_file(path)}
                for path in code_paths
            ],
            "configs": {
                "analysis_spec": {
                    "path": repository_path(analysis_spec_path),
                    "sha256": sha256_file(analysis_spec_path),
                },
                "analysis_config": {
                    "path": repository_path(config_path),
                    "sha256": sha256_file(config_path),
                },
                "figure_style": {
                    "path": repository_path(figure_style_path),
                    "sha256": sha256_file(figure_style_path),
                    "adopted_from": "LabGraph/config/figure_style.json",
                },
                "resolved_snapshot": {
                    "path": "config.snapshot.json",
                    "sha256": sha256_file(incomplete_dir / "config.snapshot.json"),
                },
            },
            "environment": {
                "path": "environment.json",
                "sha256": sha256_file(incomplete_dir / "environment.json"),
                "random_seed": config.get("random_seed"),
            },
            "qc": {
                "overall_status": qc["overall_status"],
                "path": "qc.json",
                "sha256": sha256_file(incomplete_dir / "qc.json"),
                "gates": {
                    "z_geometry": z_qc["status"],
                    "page_grouping": grouping_qc["status"],
                    "page_count": "pass",
                    "source_channel": "pass",
                    "acquisition_declaration": acquisition_context["status"],
                },
            },
            "outputs": [
                _hashed_file(path, incomplete_dir) for path in pre_manifest_files
            ],
            "supersession": {
                "status": "current_at_creation",
                "supersedes": list(supersedes),
                "superseded_by": [],
                "immutability_rule": "Completed runs are never edited; corrections create a new run.",
            },
            "supported_interpretations": analysis_spec["supported_interpretations"],
            "unsupported_interpretations": analysis_spec["unsupported_interpretations"],
        }
        write_json(incomplete_dir / "analysis_manifest.json", manifest)
        output_root.resolve().mkdir(parents=True, exist_ok=True)
        shutil.move(str(incomplete_dir), str(final_dir))
    except BaseException:
        if incomplete_dir.exists():
            write_json(
                incomplete_dir / "INCOMPLETE.json",
                {
                    "analysis_id": analysis_id,
                    "status": "incomplete",
                    "completed_run_immutable": False,
                },
            )
        raise
    return final_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-tiff", type=Path, required=True)
    parser.add_argument("--read-copy", type=Path)
    parser.add_argument(
        "--acquisition-spec",
        type=Path,
        help=(
            "Acquisition JSON; defaults to zstack_acquisition.json beside the TIFF. "
            "Legacy scans without one remain reconstructable but comparison-ineligible."
        ),
    )
    parser.add_argument(
        "--channel",
        type=int,
        help="Saved ScanImage channel; required if neither TIFF nor acquisition JSON is unambiguous",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--analysis-spec", type=Path, default=DEFAULT_ANALYSIS_SPEC)
    parser.add_argument("--figure-style", type=Path, default=DEFAULT_FIGURE_STYLE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--failed-root", type=Path, default=DEFAULT_FAILED_ROOT)
    parser.add_argument(
        "--supersedes",
        action="append",
        default=[],
        help="Prior immutable analysis_id replaced by this run; repeat as needed",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    invocation = [
        sys.executable,
        repository_path(ENTRY_SCRIPT),
        *(list(argv) if argv is not None else sys.argv[1:]),
    ]
    run_dir = build_run(
        source_tiff=args.source_tiff,
        read_copy=args.read_copy,
        config_path=args.config,
        analysis_spec_path=args.analysis_spec,
        figure_style_path=args.figure_style,
        acquisition_spec_path=args.acquisition_spec,
        output_root=args.output_root,
        failed_root=args.failed_root,
        source_channel=args.channel,
        supersedes=args.supersedes,
        invocation=invocation,
    )
    print(json.dumps({"analysis_run": str(run_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
