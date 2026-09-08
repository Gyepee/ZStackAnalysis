"""Evaluate A-D reconstruction repeatability on overlapping physical volumes.

The 3-D alignment in this module is an evaluation-only alignment between
separate reconstructed stacks. It is not used to correct either acquisition
and is not an axial-motion correction method.
"""

from __future__ import annotations

import argparse
import itertools
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage.feature import peak_local_max
from skimage.registration import phase_cross_correlation
import tifffile

from .bench2p_zstack import (
    DEFAULT_STYLE,
    REPO_ROOT,
    configure_matplotlib,
    git_state,
    normalized_correlation,
    read_json,
    repo_relative,
    sha256_file,
    stable_digest,
    write_json,
)
from .bench2p_zstack_xy_registration import (
    DEFAULT_CONFIG,
    METHODS,
    MODULE_PATH as REGISTRATION_MODULE_PATH,
    SCOPE_STATEMENT,
    _write_rows,
)


MODULE_PATH = Path("src/labgraph_ops/workflows/bench2p_zstack_xy_registration_repeat.py")
RUNNER_PATH = Path("scripts/compare_bench2p_zstack_xy_registration_repeats.py")
EVALUATION_SCOPE = (
    "evaluation-only 3D alignment between repeat stacks; no axial-motion correction"
)


def _load_run(run_dir: Path) -> dict[str, Any]:
    manifest = read_json(run_dir / "analysis_manifest.json")
    if manifest.get("scope_statement") != SCOPE_STATEMENT:
        raise ValueError(f"Not a compatible XY-registration run: {run_dir}")
    sources = read_json(run_dir / "sources.json")
    source = sources["authoritative_source"]
    volumes = {}
    for method in METHODS:
        path = run_dir / "volumes" / f"{source['scan_id']}__{method}.ome.tif"
        if not path.is_file():
            raise FileNotFoundError(path)
        volumes[method] = path
    return {
        "run_dir": run_dir,
        "analysis_id": manifest["analysis_id"],
        "qc_status": read_json(run_dir / "qc.json")["overall_status"],
        "source": source,
        "volumes": volumes,
    }


def common_physical_grid(
    first_source: dict[str, Any],
    second_source: dict[str, Any],
    *,
    target_spacing_um: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    first_z = np.asarray(first_source["z_positions_um"], dtype=float)
    second_z = np.asarray(second_source["z_positions_um"], dtype=float)
    z_start = max(float(first_z[0]), float(second_z[0]))
    z_stop = min(float(first_z[-1]), float(second_z[-1]))
    x_stop = min(float(first_source["fov_x_um"]), float(second_source["fov_x_um"]))
    y_stop = min(float(first_source["fov_y_um"]), float(second_source["fov_y_um"]))
    if z_stop <= z_start or x_stop <= 0 or y_stop <= 0:
        raise ValueError("Repeat stacks have no common physical volume")
    z = np.arange(z_start, z_stop + target_spacing_um * 0.25, target_spacing_um)
    y = np.arange(0.0, y_stop, target_spacing_um)
    x = np.arange(0.0, x_stop, target_spacing_um)
    if min(z.size, y.size, x.size) < 4:
        raise ValueError("Common physical grid is too small")
    return z, y, x, {
        "common_z_start_scanimage_um": z_start,
        "common_z_stop_scanimage_um": z_stop,
        "common_y_stop_um": y_stop,
        "common_x_stop_um": x_stop,
        "target_spacing_um": target_spacing_um,
    }


def resample_to_physical_grid(
    volume: np.ndarray,
    source: dict[str, Any],
    z_um: np.ndarray,
    y_um: np.ndarray,
    x_um: np.ndarray,
) -> np.ndarray:
    z_positions = np.asarray(source["z_positions_um"], dtype=float)
    z_index = np.interp(z_um, z_positions, np.arange(z_positions.size, dtype=float))
    y_index = y_um / float(source["pixel_size_y_um"])
    x_index = x_um / float(source["pixel_size_x_um"])
    zz, yy, xx = np.meshgrid(z_index, y_index, x_index, indexing="ij")
    result = ndimage.map_coordinates(
        np.asarray(volume, dtype=np.float32),
        [zz, yy, xx],
        order=1,
        mode="nearest",
        prefilter=False,
    )
    return np.asarray(result, dtype=np.float32)


def _standardize_for_alignment(volume: np.ndarray) -> np.ndarray:
    result = ndimage.gaussian_filter(np.asarray(volume, dtype=np.float64), sigma=0.7)
    result -= float(np.median(result))
    mad = float(np.median(np.abs(result)))
    if mad:
        result /= 1.4826 * mad
    windows = [np.hanning(length) for length in result.shape]
    result *= windows[0][:, None, None]
    result *= windows[1][None, :, None]
    result *= windows[2][None, None, :]
    return result


def _landmarks(
    volume: np.ndarray,
    *,
    count: int,
    minimum_distance_voxels: int,
) -> np.ndarray:
    filled = np.nan_to_num(volume, nan=float(np.nanmedian(volume)))
    feature = ndimage.gaussian_filter(filled, 0.8) - ndimage.gaussian_filter(filled, 4.0)
    threshold = float(np.percentile(feature, 99.8))
    points = peak_local_max(
        feature,
        min_distance=max(1, minimum_distance_voxels),
        threshold_abs=threshold,
        exclude_border=max(2, minimum_distance_voxels),
        num_peaks=count,
    )
    return np.asarray(points, dtype=float)


def landmark_repeatability(
    reference: np.ndarray,
    moving_aligned: np.ndarray,
    *,
    spacing_um: float,
    count: int,
    minimum_distance_um: float,
    match_radius_um: float,
) -> dict[str, Any]:
    minimum_distance_voxels = max(1, int(round(minimum_distance_um / spacing_um)))
    reference_points = _landmarks(
        reference, count=count, minimum_distance_voxels=minimum_distance_voxels
    )
    moving_points = _landmarks(
        moving_aligned, count=count, minimum_distance_voxels=minimum_distance_voxels
    )
    if not len(reference_points) or not len(moving_points):
        return {
            "reference_landmark_count": int(len(reference_points)),
            "moving_landmark_count": int(len(moving_points)),
            "matched_landmark_count": 0,
            "landmark_error_median_um": float("nan"),
            "landmark_error_p95_um": float("nan"),
        }
    distances, _ = cKDTree(moving_points * spacing_um).query(
        reference_points * spacing_um, k=1
    )
    accepted = distances <= match_radius_um
    matched = distances[accepted]
    return {
        "reference_landmark_count": int(len(reference_points)),
        "moving_landmark_count": int(len(moving_points)),
        "matched_landmark_count": int(matched.size),
        "landmark_error_median_um": float(np.median(matched)) if matched.size else float("nan"),
        "landmark_error_p95_um": float(np.percentile(matched, 95)) if matched.size else float("nan"),
    }


def compare_pair(
    first: dict[str, Any],
    second: dict[str, Any],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    repeat = config["repeat_comparison"]
    spacing = float(repeat["target_spacing_um"])
    z_um, y_um, x_um, geometry = common_physical_grid(
        first["source"], second["source"], target_spacing_um=spacing
    )
    rows = []
    for method in METHODS:
        first_volume = resample_to_physical_grid(
            tifffile.imread(first["volumes"][method]), first["source"], z_um, y_um, x_um
        )
        second_volume = resample_to_physical_grid(
            tifffile.imread(second["volumes"][method]), second["source"], z_um, y_um, x_um
        )
        shift_voxels, error, _ = phase_cross_correlation(
            _standardize_for_alignment(first_volume),
            _standardize_for_alignment(second_volume),
            upsample_factor=int(repeat["registration_upsample_factor"]),
            normalization=None,
        )
        shift_um = np.asarray(shift_voxels, dtype=float) * spacing
        magnitude_um = float(np.linalg.norm(shift_um))
        within_limit = magnitude_um <= float(repeat["max_3d_shift_um"])
        moving_aligned = ndimage.shift(
            second_volume,
            shift=shift_voxels,
            order=1,
            mode="constant",
            cval=np.nan,
            prefilter=False,
        )
        row = {
            "animal_id": first["source"]["animal_id"],
            "date": first["source"]["date"],
            "z_step_um": first["source"]["z_step_um"],
            "first_analysis_id": first["analysis_id"],
            "second_analysis_id": second["analysis_id"],
            "first_scan_id": first["source"]["scan_id"],
            "second_scan_id": second["source"]["scan_id"],
            "method": method,
            **geometry,
            "grid_shape_z": int(z_um.size),
            "grid_shape_y": int(y_um.size),
            "grid_shape_x": int(x_um.size),
            "ncc_before_evaluation_alignment": normalized_correlation(
                first_volume, second_volume
            ),
            "ncc_after_evaluation_alignment": normalized_correlation(
                first_volume, moving_aligned
            ),
            "evaluation_shift_z_um": float(shift_um[0]),
            "evaluation_shift_y_um": float(shift_um[1]),
            "evaluation_shift_x_um": float(shift_um[2]),
            "evaluation_shift_magnitude_um": magnitude_um,
            "evaluation_shift_within_limit": bool(within_limit),
            "evaluation_registration_error": float(error),
            **landmark_repeatability(
                first_volume,
                moving_aligned,
                spacing_um=spacing,
                count=int(repeat["landmark_count"]),
                minimum_distance_um=float(repeat["landmark_min_distance_um"]),
                match_radius_um=float(repeat["landmark_match_radius_um"]),
            ),
        }
        rows.append(row)
    return rows, geometry


def save_repeat_figure(
    run_dir: Path,
    rows: Sequence[dict[str, Any]],
    style: dict[str, Any],
    profile_name: str,
) -> dict[str, Path]:
    configure_matplotlib(style, profile_name)
    pair_keys = list(
        dict.fromkeys((row["first_scan_id"], row["second_scan_id"]) for row in rows)
    )
    colors = ["#0072B2", "#E69F00", "#009E73", "#D55E00"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), constrained_layout=True)
    x = np.arange(len(METHODS))
    for pair_index, pair in enumerate(pair_keys):
        selected = [
            row for row in rows if (row["first_scan_id"], row["second_scan_id"]) == pair
        ]
        by_method = {row["method"]: row for row in selected}
        label = f"{pair[0]}–{pair[1]}"
        axes[0].plot(
            x,
            [by_method[method]["ncc_after_evaluation_alignment"] for method in METHODS],
            marker="o",
            linewidth=1,
            color=colors[pair_index % len(colors)],
            label=label,
        )
        axes[1].plot(
            x,
            [by_method[method]["landmark_error_median_um"] for method in METHODS],
            marker="o",
            linewidth=1,
            color=colors[pair_index % len(colors)],
            label=label,
        )
    for axis in axes:
        axis.set_xticks(x, [method.split("_", 1)[0] for method in METHODS])
        axis.set_xlabel("reconstruction method")
    axes[0].set_ylabel("3-D NCC after evaluation alignment")
    axes[1].set_ylabel("nearest-landmark error (µm)")
    axes[0].legend(frameon=False, fontsize=6)
    fig.suptitle("repeat-stack consistency; evaluation alignment only")
    figure_dir = run_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    svg = figure_dir / "fig_bench2p_xy_registration_repeatability.svg"
    png = figure_dir / "fig_bench2p_xy_registration_repeatability.png"
    fig.savefig(svg, bbox_inches="tight", facecolor="white")
    fig.savefig(png, dpi=style["output"]["png_dpi"], bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {"svg": svg, "png": png}


def build_repeat_run(
    *,
    input_runs: Sequence[Path],
    config_path: Path,
    style_path: Path,
    output_root: Path,
) -> Path:
    config = read_json(config_path)
    style = read_json(style_path)
    loaded = [_load_run(path.resolve()) for path in input_runs]
    groups: dict[tuple[str, str, float], list[dict[str, Any]]] = {}
    for item in loaded:
        source = item["source"]
        key = (source["animal_id"], source["date"], float(source["z_step_um"]))
        groups.setdefault(key, []).append(item)
    pairs = [
        pair
        for items in groups.values()
        for pair in itertools.combinations(sorted(items, key=lambda value: value["source"]["scan_id"]), 2)
    ]
    if not pairs:
        raise ValueError("No same-animal/date/Z-step repeat pairs were supplied")
    identity = {
        "inputs": [item["analysis_id"] for item in loaded],
        "config": config["repeat_comparison"],
    }
    created_at = datetime.now(timezone.utc)
    analysis_id = (
        f"bench2p_zstack_xyreg_repeat__{created_at.strftime('%Y%m%dT%H%M%SZ')}__"
        f"{stable_digest(identity)}"
    )
    final_dir = output_root.resolve() / analysis_id
    incomplete_dir = output_root.resolve() / f".{analysis_id}.incomplete"
    (incomplete_dir / "tables").mkdir(parents=True)
    try:
        rows: list[dict[str, Any]] = []
        pair_geometry = []
        for first, second in pairs:
            pair_rows, geometry = compare_pair(first, second, config)
            rows.extend(pair_rows)
            pair_geometry.append(
                {
                    "first_scan_id": first["source"]["scan_id"],
                    "second_scan_id": second["source"]["scan_id"],
                    **geometry,
                }
            )
            print(
                f"compared {first['source']['scan_id']} to {second['source']['scan_id']}",
                flush=True,
            )
        table_path = incomplete_dir / "tables" / "repeat_stack_consistency.csv"
        _write_rows(table_path, rows)
        _write_rows(incomplete_dir / "tables" / "repeat_pair_geometry.csv", pair_geometry)
        figure_paths = save_repeat_figure(
            incomplete_dir, rows, style, str(config["figure_profile"])
        )
        write_json(
            incomplete_dir / "config.snapshot.json",
            {
                **config,
                "analysis_id": analysis_id,
                "created_at_utc": created_at.isoformat(),
                "input_analysis_ids": [item["analysis_id"] for item in loaded],
                "evaluation_scope": EVALUATION_SCOPE,
            },
        )
        shutil.move(str(incomplete_dir), str(final_dir))
        table_path = final_dir / "tables" / table_path.name
        figure_paths = {
            key: final_dir / "figures" / path.name for key, path in figure_paths.items()
        }
    except BaseException:
        if incomplete_dir.exists():
            write_json(
                incomplete_dir / "INCOMPLETE.json",
                {"analysis_id": analysis_id, "status": "incomplete"},
            )
        raise
    code = [
        {"path": str(path), "sha256": sha256_file(REPO_ROOT / path)}
        for path in (RUNNER_PATH, MODULE_PATH, REGISTRATION_MODULE_PATH)
    ]
    input_records = [
        {
            "analysis_id": item["analysis_id"],
            "manifest_path": repo_relative(item["run_dir"] / "analysis_manifest.json"),
            "manifest_sha256": sha256_file(item["run_dir"] / "analysis_manifest.json"),
            "qc_status": item["qc_status"],
        }
        for item in loaded
    ]
    write_json(
        final_dir / "figures" / "fig_bench2p_xy_registration_repeatability.json",
        {
            "schema_version": "labgraph.figure.v1",
            "figure_id": "fig_bench2p_xy_registration_repeatability",
            "analysis_id": analysis_id,
            "created_at_utc": created_at.isoformat(),
            "evaluation_scope": EVALUATION_SCOPE,
            "plot_data": {
                "path": repo_relative(table_path),
                "sha256": sha256_file(table_path),
            },
            "inputs": input_records,
            "code": code,
            "outputs": {
                key: {"path": repo_relative(path), "sha256": sha256_file(path)}
                for key, path in figure_paths.items()
            },
            "limits": [
                SCOPE_STATEMENT,
                EVALUATION_SCOPE,
                "Nearest bright landmarks are automated intensity peaks, not validated cells.",
            ],
        },
    )
    (final_dir / "figures" / "fig_bench2p_xy_registration_repeatability.md").write_text(
        "\n".join(
            [
                "# fig_bench2p_xy_registration_repeatability",
                "",
                "Pairwise 3-D NCC and nearest bright-landmark consistency after an",
                "evaluation-only alignment on a shared 2 µm physical grid.",
                "",
                f"- {SCOPE_STATEMENT}.",
                f"- {EVALUATION_SCOPE}.",
                "- Landmark peaks are review targets, not cells or soma morphology.",
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
                "Repeat-stack consistency benchmark for A-D reconstructions.",
                "",
                f"**Scope:** {EVALUATION_SCOPE}.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    output_paths = sorted(
        path for path in final_dir.rglob("*") if path.is_file() and path.name != "analysis_manifest.json"
    )
    write_json(
        final_dir / "analysis_manifest.json",
        {
            "schema_version": "labgraph.analysis_manifest.v1",
            "analysis_id": analysis_id,
            "title": "bench2p XY-registration repeat-stack consistency",
            "question": "Does within-plane registration improve repeat-stack consistency over filter-only averaging?",
            "created_at_utc": created_at.isoformat(),
            "analysis_state": config["analysis_state"],
            "scope_statement": SCOPE_STATEMENT,
            "evaluation_scope": EVALUATION_SCOPE,
            "analysis_unit": "same-animal same-day overlapping stack pair",
            "n": {"input_runs": len(loaded), "repeat_pairs": len(pairs)},
            "inputs": input_records,
            "code": code,
            "git": git_state(),
            "configuration": {
                "path": repo_relative(final_dir / "config.snapshot.json"),
                "sha256": sha256_file(final_dir / "config.snapshot.json"),
            },
            "statistical_method": "descriptive paired A-D comparison; no inferential test",
            "unsupported_interpretations": [
                "The evaluation-only 3-D alignment is not an axial-motion correction applied to raw frames.",
                "Differences between 1 and 2 µm acquisitions are not isolated Z-step effects.",
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
        },
    )
    return final_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-run", action="append", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--style", type=Path, default=DEFAULT_STYLE)
    parser.add_argument(
        "--output-root", type=Path, default=REPO_ROOT / "datasets" / "analysis_runs"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_dir = build_repeat_run(
        input_runs=args.input_run,
        config_path=args.config,
        style_path=args.style,
        output_root=args.output_root,
    )
    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
