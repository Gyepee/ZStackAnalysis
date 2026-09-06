"""Read-only validation for completed ZStackAnalysis run bundles."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .pipeline import sha256_file


REQUIRED_IDENTITY_COLUMNS = {
    "analysis_id",
    "pipeline_id",
    "pipeline_version",
    "animal_id",
    "acquisition_date",
    "post_surgery_day",
    "scan_id",
    "session_id",
    "source_channel",
}


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _inside(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"Manifest path escapes run directory: {relative}") from error
    return candidate


def validate_run_directory(run_dir: Path) -> dict[str, Any]:
    """Validate hashes, stable IDs, and complete figure bundles without writing."""
    run_dir = run_dir.resolve()
    errors: list[str] = []
    manifest_path = run_dir / "analysis_manifest.json"
    if not manifest_path.exists():
        return {
            "status": "fail",
            "run_dir": str(run_dir),
            "errors": ["analysis_manifest.json is missing"],
        }
    manifest = _read_json(manifest_path)
    required_manifest = {
        "analysis_id",
        "title",
        "question",
        "analysis_state",
        "pipeline",
        "identity",
        "inputs",
        "eligibility",
        "analysis_units",
        "entry_script",
        "git",
        "configs",
        "qc",
        "outputs",
        "supersession",
        "supported_interpretations",
        "unsupported_interpretations",
    }
    missing = sorted(required_manifest - set(manifest))
    if missing:
        errors.append(f"manifest fields missing: {missing}")

    for item in manifest.get("outputs", []):
        relative = item.get("path")
        if not isinstance(relative, str):
            errors.append("output path is missing or not a string")
            continue
        try:
            path = _inside(run_dir, relative)
        except ValueError as error:
            errors.append(str(error))
            continue
        if not path.is_file():
            errors.append(f"output missing: {relative}")
            continue
        if path.stat().st_size != item.get("size_bytes"):
            errors.append(f"size mismatch: {relative}")
        if sha256_file(path) != item.get("sha256"):
            errors.append(f"hash mismatch: {relative}")

    for table_path in sorted((run_dir / "tables").glob("*.csv")):
        with table_path.open(encoding="utf-8", newline="") as stream:
            header = set(next(csv.reader(stream), []))
        absent = sorted(REQUIRED_IDENTITY_COLUMNS - header)
        if absent:
            errors.append(f"stable identity columns missing from {table_path.name}: {absent}")

    for json_path in sorted((run_dir / "figures").glob("fig_*.json")):
        figure = _read_json(json_path)
        figure_id = figure.get("figure_id")
        if json_path.stem != figure_id:
            errors.append(f"figure_id/path mismatch: {json_path.name}")
            continue
        for suffix in (".svg", ".png", ".md"):
            if not (run_dir / "figures" / f"{figure_id}{suffix}").is_file():
                errors.append(f"figure bundle member missing: {figure_id}{suffix}")
        exact_data = figure.get("exact_plotted_data", {})
        data_path_value = exact_data.get("path")
        if not isinstance(data_path_value, str):
            errors.append(f"exact plotted data path missing: {figure_id}")
        else:
            try:
                data_path = _inside(run_dir, data_path_value)
                if not data_path.is_file():
                    errors.append(f"exact plotted data missing: {data_path_value}")
                elif sha256_file(data_path) != exact_data.get("sha256"):
                    errors.append(f"exact plotted data hash mismatch: {figure_id}")
            except ValueError as error:
                errors.append(str(error))
        if not figure.get("panels"):
            errors.append(f"panel map missing: {figure_id}")
        if figure.get("analysis_id") != manifest.get("analysis_id"):
            errors.append(f"figure/analysis ID mismatch: {figure_id}")

    expected_figures = {
        "fig_mean_vs_max_projections",
        "fig_mean_projections",
        "fig_representative_median_plane",
    }
    present_figures = {
        path.stem for path in (run_dir / "figures").glob("fig_*.json")
    }
    absent_figures = sorted(expected_figures - present_figures)
    if absent_figures:
        errors.append(f"required figure bundles missing: {absent_figures}")
    return {
        "status": "pass" if not errors else "fail",
        "run_dir": str(run_dir),
        "analysis_id": manifest.get("analysis_id"),
        "checked_output_count": len(manifest.get("outputs", [])),
        "checked_figure_count": len(present_figures),
        "errors": errors,
    }
