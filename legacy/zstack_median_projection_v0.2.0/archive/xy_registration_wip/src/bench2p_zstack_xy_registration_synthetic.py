"""Ground-truth checks for the Bench2p within-plane XY registration baseline."""

from __future__ import annotations

import argparse
import csv
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

from .bench2p_zstack import (
    DEFAULT_STYLE,
    REPO_ROOT,
    collect_environment,
    configure_matplotlib,
    read_json,
    repo_relative,
    sha256_file,
    stable_digest,
    write_json,
)
from .bench2p_zstack_xy_registration import (
    DEFAULT_CONFIG,
    SCOPE_STATEMENT,
    process_plane_methods,
)


MODULE_PATH = Path(
    "src/labgraph_ops/workflows/bench2p_zstack_xy_registration_synthetic.py"
)
RUNNER_PATH = Path("scripts/validate_bench2p_zstack_xy_registration_synthetic.py")
REGISTRATION_MODULE_PATH = Path(
    "src/labgraph_ops/workflows/bench2p_zstack_xy_registration.py"
)


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def synthetic_reference(shape: tuple[int, int], seed: int) -> np.ndarray:
    """Build a deterministic, nonperiodic fluorescence-like reference image."""
    rng = np.random.default_rng(seed)
    yy, xx = np.indices(shape, dtype=np.float64)
    image = np.zeros(shape, dtype=np.float64)
    for _ in range(28):
        center_y = rng.uniform(12, shape[0] - 12)
        center_x = rng.uniform(12, shape[1] - 12)
        sigma_y = rng.uniform(1.4, 5.5)
        sigma_x = rng.uniform(1.4, 5.5)
        amplitude = rng.uniform(0.3, 1.0)
        image += amplitude * np.exp(
            -0.5
            * (
                ((yy - center_y) / sigma_y) ** 2
                + ((xx - center_x) / sigma_x) ** 2
            )
        )
    texture = ndimage.gaussian_filter(rng.normal(size=shape), sigma=2.2)
    texture -= np.min(texture)
    texture /= max(float(np.max(texture)), np.finfo(float).eps)
    image += 0.12 * texture
    image -= np.min(image)
    image /= max(float(np.max(image)), np.finfo(float).eps)
    return np.asarray(150.0 + 1800.0 * image, dtype=np.float32)


def shifted_frames(
    reference: np.ndarray,
    motions_yx_px: np.ndarray,
    brightness_scales: np.ndarray,
    offsets: np.ndarray,
    *,
    noise_sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    frames = np.empty((motions_yx_px.shape[0], *reference.shape), dtype=np.float32)
    for index, (motion, scale, offset) in enumerate(
        zip(motions_yx_px, brightness_scales, offsets, strict=True)
    ):
        frame = ndimage.shift(
            reference,
            shift=tuple(float(value) for value in motion),
            order=3,
            mode="reflect",
            prefilter=True,
        )
        frame = float(scale) * frame + float(offset)
        frame += rng.normal(0.0, noise_sigma, size=reference.shape)
        frames[index] = frame
    return frames


def _centered_shift_error(
    recovered_yx: np.ndarray,
    true_correction_yx: np.ndarray,
    comparison_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    recovered_centered = recovered_yx - np.median(
        recovered_yx[comparison_mask], axis=0
    )
    truth_centered = true_correction_yx - np.median(
        true_correction_yx[comparison_mask], axis=0
    )
    error = recovered_centered - truth_centered
    return recovered_centered, truth_centered, error


def evaluate_synthetic_registration(
    config: dict[str, Any], *, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Exercise known motion, brightness-only, outlier, and low-SNR cases."""
    rng = np.random.default_rng(seed)
    reference = synthetic_reference((160, 160), seed)
    reference_std = float(np.std(reference))
    frame_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    case_specs: list[dict[str, Any]] = []
    count = 48
    known_motion = rng.uniform(-2.0, 2.0, size=(count, 2))
    known_outlier = np.zeros(count, dtype=bool)
    known_outlier[[7, 31]] = True
    known_motion[7] = (10.0, -9.0)
    known_motion[31] = (-11.0, 10.0)
    case_specs.append(
        {
            "case": "known_subpixel_motion_with_outliers",
            "motions": known_motion,
            "brightness": rng.uniform(0.65, 1.40, size=count),
            "offsets": rng.uniform(-80.0, 80.0, size=count),
            "noise_sigma": 0.04 * reference_std,
            "outlier": known_outlier,
        }
    )
    count = 36
    case_specs.append(
        {
            "case": "brightness_only_no_motion",
            "motions": np.zeros((count, 2), dtype=float),
            "brightness": np.linspace(0.45, 1.65, count),
            "offsets": np.linspace(-120.0, 120.0, count),
            "noise_sigma": 0.02 * reference_std,
            "outlier": np.zeros(count, dtype=bool),
        }
    )
    count = 36
    case_specs.append(
        {
            "case": "low_snr_no_motion",
            "motions": np.zeros((count, 2), dtype=float),
            "brightness": np.full(count, 0.08),
            "offsets": np.zeros(count),
            "noise_sigma": 1.0 * reference_std,
            "outlier": np.zeros(count, dtype=bool),
        }
    )

    figure_payload: dict[str, Any] = {}
    for case_spec in case_specs:
        frames = shifted_frames(
            reference,
            case_spec["motions"],
            case_spec["brightness"],
            case_spec["offsets"],
            noise_sigma=float(case_spec["noise_sigma"]),
            rng=rng,
        )
        _, registration_rows, plane_summary = process_plane_methods(
            frames,
            pixel_size_x_um=1.0,
            pixel_size_y_um=1.0,
            config=config,
        )
        recovered = np.asarray(
            [[row["dy_px"], row["dx_px"]] for row in registration_rows],
            dtype=float,
        )
        truth = -np.asarray(case_spec["motions"], dtype=float)
        outlier = np.asarray(case_spec["outlier"], dtype=bool)
        comparison_mask = ~outlier
        recovered_centered, truth_centered, error = _centered_shift_error(
            recovered, truth, comparison_mask
        )
        retained = np.asarray(
            [row["registered_quality_retained"] for row in registration_rows],
            dtype=bool,
        )
        error_magnitude = np.hypot(error[:, 0], error[:, 1])
        recovered_magnitude = np.hypot(
            recovered_centered[:, 0], recovered_centered[:, 1]
        )
        case_name = str(case_spec["case"])
        for index, registration_row in enumerate(registration_rows):
            frame_rows.append(
                {
                    "case": case_name,
                    "frame_index": index,
                    "injected_motion_dy_px": float(case_spec["motions"][index, 0]),
                    "injected_motion_dx_px": float(case_spec["motions"][index, 1]),
                    "true_centered_correction_dy_px": float(truth_centered[index, 0]),
                    "true_centered_correction_dx_px": float(truth_centered[index, 1]),
                    "recovered_centered_dy_px": float(recovered_centered[index, 0]),
                    "recovered_centered_dx_px": float(recovered_centered[index, 1]),
                    "recovery_error_px": float(error_magnitude[index]),
                    "brightness_scale": float(case_spec["brightness"][index]),
                    "brightness_offset": float(case_spec["offsets"][index]),
                    "injected_large_motion_outlier": bool(outlier[index]),
                    "registered_quality_retained": bool(retained[index]),
                    "reference_correlation_before": registration_row[
                        "reference_correlation_before"
                    ],
                    "reference_correlation_after": registration_row[
                        "reference_correlation_after"
                    ],
                    "registration_error": registration_row["registration_error"],
                }
            )
        rmse = float(np.sqrt(np.mean(error[comparison_mask] ** 2)))
        summary_rows.append(
            {
                "case": case_name,
                "frame_count": int(frames.shape[0]),
                "non_outlier_recovery_rmse_px": rmse,
                "non_outlier_recovery_error_p95_px": float(
                    np.percentile(error_magnitude[comparison_mask], 95)
                ),
                "recovered_shift_magnitude_p95_px": float(
                    np.percentile(recovered_magnitude, 95)
                ),
                "retained_fraction": float(retained.mean()),
                "large_outliers_injected": int(outlier.sum()),
                "large_outliers_rejected": int(np.sum(outlier & ~retained)),
                "correlation_before_mean": plane_summary["correlation_before_mean"],
                "correlation_after_mean": plane_summary["correlation_after_mean"],
                "keep_floor_relaxed": plane_summary["registered_keep_floor_relaxed"],
            }
        )
        figure_payload[case_name] = {
            "recovered": recovered_centered,
            "truth": truth_centered,
            "retained": retained,
            "error_magnitude": error_magnitude,
        }

    by_case = {row["case"]: row for row in summary_rows}
    known = by_case["known_subpixel_motion_with_outliers"]
    brightness = by_case["brightness_only_no_motion"]
    low_snr = by_case["low_snr_no_motion"]
    gates = {
        "known_shift_recovery_rmse_below_0_35_px": bool(
            known["non_outlier_recovery_rmse_px"] <= 0.35
        ),
        "all_large_motion_outliers_rejected": bool(
            known["large_outliers_injected"] == known["large_outliers_rejected"]
        ),
        "brightness_only_false_shift_p95_below_0_5_px": bool(
            brightness["recovered_shift_magnitude_p95_px"] <= 0.5
        ),
        "brightness_only_retention_at_least_90_percent": bool(
            brightness["retained_fraction"] >= 0.9
        ),
    }
    interpretation = {
        "scope_statement": SCOPE_STATEMENT,
        "gates": gates,
        "overall_status": "pass" if all(gates.values()) else "needs_review",
        "low_snr_no_motion_false_shift_p95_px": low_snr[
            "recovered_shift_magnitude_p95_px"
        ],
        "low_snr_note": (
            "The low-SNR zero-motion case is descriptive, not evidence of real motion. "
            "Any recovered displacement is a false estimate and motivates quality rejection."
        ),
    }
    return frame_rows, summary_rows, {"figure": figure_payload, "interpretation": interpretation}


def save_synthetic_figure(
    path: Path,
    payload: dict[str, Any],
    summary_rows: list[dict[str, Any]],
    style: dict[str, Any],
    profile_name: str,
) -> None:
    configure_matplotlib(style, profile_name)
    figure, axes = plt.subplots(2, 2, figsize=(8.2, 6.5), constrained_layout=True)
    known = payload["known_subpixel_motion_with_outliers"]
    retained = known["retained"]
    for dimension, (axis, label) in enumerate(
        zip(axes[0], ("dy", "dx"), strict=True)
    ):
        axis.scatter(
            known["truth"][:, dimension],
            known["recovered"][:, dimension],
            c=np.where(retained, "#3569a8", "#c84b31"),
            s=14,
            alpha=0.8,
        )
        bounds = np.asarray(axis.get_xlim() + axis.get_ylim(), dtype=float)
        low, high = float(np.min(bounds)), float(np.max(bounds))
        axis.plot([low, high], [low, high], color="0.25", lw=0.8, ls="--")
        axis.set(xlabel=f"true centered {label} correction (px)", ylabel="recovered (px)")
    cases = [row["case"] for row in summary_rows]
    p95 = [row["recovered_shift_magnitude_p95_px"] for row in summary_rows]
    axes[1, 0].bar(range(len(cases)), p95, color="#5c87b2")
    axes[1, 0].set_xticks(range(len(cases)), ["known", "brightness", "low SNR"])
    axes[1, 0].set_ylabel("recovered shift magnitude p95 (px)")
    retention = [row["retained_fraction"] for row in summary_rows]
    axes[1, 1].bar(range(len(cases)), retention, color="#6b9d63")
    axes[1, 1].set_xticks(range(len(cases)), ["known", "brightness", "low SNR"])
    axes[1, 1].set_ylabel("D retained fraction")
    axes[1, 1].set_ylim(0, 1.02)
    figure.suptitle("Synthetic validation: within-plane rigid XY registration")
    figure.savefig(path, dpi=int(style["profiles"][profile_name]["dpi"]))
    plt.close(figure)


def build_synthetic_validation_run(
    *,
    config_path: Path,
    style_path: Path,
    output_root: Path,
    seed: int | None = None,
) -> Path:
    config = read_json(config_path)
    if config.get("scope_statement") != SCOPE_STATEMENT:
        raise ValueError(f"Config must declare: {SCOPE_STATEMENT}")
    style = read_json(style_path)
    resolved_seed = int(config.get("random_seed", 0) if seed is None else seed)
    created_at = datetime.now(timezone.utc)
    identity = {"config": config, "seed": resolved_seed, "type": "synthetic"}
    analysis_id = (
        f"bench2p_zstack_xyreg_synthetic__{created_at.strftime('%Y%m%dT%H%M%SZ')}__"
        f"{stable_digest(identity)}"
    )
    final_dir = output_root.resolve() / analysis_id
    incomplete_dir = output_root.resolve() / f".{analysis_id}.incomplete"
    if final_dir.exists() or incomplete_dir.exists():
        raise FileExistsError(f"Run target already exists for {analysis_id}")
    (incomplete_dir / "tables").mkdir(parents=True)
    (incomplete_dir / "figures").mkdir()
    try:
        frame_rows, summary_rows, payload = evaluate_synthetic_registration(
            config, seed=resolved_seed
        )
        _write_rows(incomplete_dir / "tables" / "synthetic_frame_recovery.csv", frame_rows)
        _write_rows(incomplete_dir / "tables" / "synthetic_case_summary.csv", summary_rows)
        figure_path = incomplete_dir / "figures" / "fig_synthetic_xy_registration.png"
        save_synthetic_figure(
            figure_path,
            payload["figure"],
            summary_rows,
            style,
            str(config["figure_profile"]),
        )
        snapshot = {
            **config,
            "analysis_id": analysis_id,
            "created_at_utc": created_at.isoformat(),
            "synthetic_random_seed": resolved_seed,
            "config_source": repo_relative(config_path),
            "figure_style_source": repo_relative(style_path),
        }
        write_json(incomplete_dir / "config.snapshot.json", snapshot)
        write_json(incomplete_dir / "environment.json", collect_environment())
        write_json(incomplete_dir / "qc.json", payload["interpretation"])
        shutil.move(str(incomplete_dir), str(final_dir))
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
    outputs = [
        final_dir / "tables" / "synthetic_frame_recovery.csv",
        final_dir / "tables" / "synthetic_case_summary.csv",
        final_dir / "figures" / "fig_synthetic_xy_registration.png",
        final_dir / "qc.json",
    ]
    write_json(
        final_dir / "manifest.json",
        {
            "schema_version": "labgraph.bench2p_zstack_xyreg_synthetic.v1",
            "analysis_id": analysis_id,
            "created_at_utc": created_at.isoformat(),
            "scope_statement": SCOPE_STATEMENT,
            "synthetic_ground_truth": True,
            "random_seed": resolved_seed,
            "config": {
                "path": repo_relative(final_dir / "config.snapshot.json"),
                "sha256": sha256_file(final_dir / "config.snapshot.json"),
            },
            "code": code,
            "outputs": [
                {"path": repo_relative(path), "sha256": sha256_file(path)}
                for path in outputs
            ],
            "limitations": [
                SCOPE_STATEMENT,
                "Synthetic texture and noise do not reproduce the full optical or biological process.",
                "No Z displacement is estimated or corrected.",
            ],
        },
    )
    return final_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--style", type=Path, default=DEFAULT_STYLE)
    parser.add_argument(
        "--output-root", type=Path, default=REPO_ROOT / "datasets" / "analysis_runs"
    )
    parser.add_argument("--seed", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_dir = build_synthetic_validation_run(
        config_path=args.config,
        style_path=args.style,
        output_root=args.output_root,
        seed=args.seed,
    )
    print(json.dumps({"analysis_run": str(run_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
