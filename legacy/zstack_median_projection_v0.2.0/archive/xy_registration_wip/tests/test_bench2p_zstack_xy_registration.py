from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
from scipy import ndimage

from labgraph_ops.workflows.bench2p_zstack_xy_registration import (
    METHOD_D,
    maximum_contiguous_true,
    page_z_grouping_qc,
    process_plane_methods,
    z_geometry_qc,
)


ROOT = Path(__file__).resolve().parents[1]


def config_for_small_images() -> dict:
    config = json.loads(
        (ROOT / "config" / "bench2p_zstack_xy_registration.json").read_text()
    )
    config["registration"]["crop_border_px"] = 4
    return config


def gaussian_reference(size: int = 64) -> np.ndarray:
    y, x = np.mgrid[:size, :size]
    image = np.exp(-((x - 31.0) ** 2 + (y - 28.0) ** 2) / (2 * 3.5**2))
    image += 0.65 * np.exp(-((x - 18.0) ** 2 + (y - 43.0) ** 2) / (2 * 2.3**2))
    return (image * 2000).astype(np.float32)


def test_z_geometry_accepts_regular_one_and_two_micron_spacing() -> None:
    for step in (1.0, 2.0):
        result = z_geometry_qc(
            np.arange(100) * step + 0.992,
            step,
            absolute_tolerance_um=1e-6,
            require_increasing=True,
        )
        assert result["status"] == "pass"
        assert result["median_diff_zs_um"] == step


def test_z_geometry_rejects_irregular_or_reversed_coordinates() -> None:
    irregular = z_geometry_qc(
        [0, 1, 2.2, 3.2],
        1.0,
        absolute_tolerance_um=1e-6,
        require_increasing=True,
    )
    reversed_result = z_geometry_qc(
        [3, 2, 1, 0],
        1.0,
        absolute_tolerance_um=1e-6,
        require_increasing=True,
    )
    assert irregular["status"] == "fail"
    assert reversed_result["status"] == "fail"


def test_page_z_grouping_requires_plane_major_contiguous_blocks() -> None:
    passing = page_z_grouping_qc(
        [0, 0, 0, 1, 1, 1],
        n_slices=2,
        frames_per_slice=3,
        absolute_tolerance_um=1e-6,
    )
    interleaved = page_z_grouping_qc(
        [0, 1, 0, 1, 0, 1],
        n_slices=2,
        frames_per_slice=3,
        absolute_tolerance_um=1e-6,
    )
    assert passing["status"] == "pass"
    assert interleaved["status"] == "fail"


def test_known_subpixel_shifts_are_recovered_and_improve_correlation() -> None:
    rng = np.random.default_rng(2)
    reference = gaussian_reference()
    applied = np.asarray([(0.0, 0.0), (1.2, -0.7), (-2.1, 1.4), (0.6, 2.2)])
    frames = np.stack(
        [
            ndimage.shift(reference * scale + offset, shift, order=3)
            + rng.normal(0, 10, reference.shape)
            for shift, scale, offset in zip(
                applied,
                [1.0, 1.2, 0.8, 1.5],
                [0.0, 80.0, -40.0, 120.0],
                strict=True,
            )
        ]
    ).astype(np.float32)
    _, rows, summary = process_plane_methods(
        frames,
        pixel_size_x_um=1.0,
        pixel_size_y_um=1.0,
        config=config_for_small_images(),
    )
    estimated = np.asarray([(row["dy_px"], row["dx_px"]) for row in rows])
    expected_pairwise = -(applied - applied[0])
    estimated_pairwise = estimated - estimated[0]
    assert np.max(np.abs(estimated_pairwise - expected_pairwise)) < 0.35
    assert summary["correlation_after_mean"] > summary["correlation_before_mean"]


def test_brightness_change_without_motion_is_not_rejected() -> None:
    reference = gaussian_reference()
    frames = np.stack(
        [
            reference * scale + offset
            for scale, offset in [(1, 0), (1.5, 200), (0.7, -100), (1.1, 50)]
        ]
    ).astype(np.float32)
    _, rows, _ = process_plane_methods(
        frames,
        pixel_size_x_um=1.0,
        pixel_size_y_um=1.0,
        config=config_for_small_images(),
    )
    assert all(row["registered_quality_retained"] for row in rows)
    assert max(row["shift_magnitude_px"] for row in rows) < 0.35


def test_large_motion_outlier_is_rejected_after_registration() -> None:
    rng = np.random.default_rng(5)
    reference = gaussian_reference()
    applied = [(0, 0), (0.4, -0.3), (-0.5, 0.2), (13.0, -11.0)]
    frames = np.stack(
        [ndimage.shift(reference, shift, order=3) + rng.normal(0, 8, reference.shape) for shift in applied]
    ).astype(np.float32)
    images, rows, _ = process_plane_methods(
        frames,
        pixel_size_x_um=1.0,
        pixel_size_y_um=1.0,
        config=config_for_small_images(),
    )
    assert images[METHOD_D].shape == reference.shape
    assert rows[-1]["registered_quality_retained"] is False
    assert "shift_exceeds_max" in rows[-1]["registered_rejection_reasons"]


def test_low_snr_static_frames_do_not_produce_large_median_shift() -> None:
    rng = np.random.default_rng(7)
    weak_reference = gaussian_reference() * 0.08
    frames = np.stack(
        [weak_reference + rng.normal(0, 20, weak_reference.shape) for _ in range(12)]
    ).astype(np.float32)
    _, rows, _ = process_plane_methods(
        frames,
        pixel_size_x_um=1.0,
        pixel_size_y_um=1.0,
        config=config_for_small_images(),
    )
    magnitudes = np.asarray([row["shift_magnitude_px"] for row in rows])
    assert np.median(magnitudes) < 1.0


def test_maximum_contiguous_true() -> None:
    assert maximum_contiguous_true([False, True, True, False, True]) == 2
