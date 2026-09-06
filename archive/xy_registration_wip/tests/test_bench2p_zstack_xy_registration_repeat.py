from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import ndimage
import tifffile

from labgraph_ops.workflows.bench2p_zstack_xy_registration import METHODS
from labgraph_ops.workflows.bench2p_zstack_xy_registration_repeat import (
    common_physical_grid,
    compare_pair,
    resample_to_physical_grid,
)


ROOT = Path(__file__).resolve().parents[1]


def source(scan_id: str, z_start: float = 0.0) -> dict:
    return {
        "animal_id": "ROS-0001",
        "date": "2026-01-01",
        "scan_id": scan_id,
        "z_step_um": 1.0,
        "z_positions_um": (np.arange(16) + z_start).tolist(),
        "fov_y_um": 32.0,
        "fov_x_um": 32.0,
        "pixel_size_y_um": 1.0,
        "pixel_size_x_um": 1.0,
    }


def test_common_grid_uses_only_overlapping_physical_z_range() -> None:
    z, y, x, geometry = common_physical_grid(
        source("a", 0.0), source("b", 3.0), target_spacing_um=2.0
    )
    assert z[0] == 3.0
    assert z[-1] <= 15.0
    assert geometry["common_z_start_scanimage_um"] == 3.0
    assert y[1] - y[0] == x[1] - x[0] == 2.0


def test_resample_uses_scanimage_z_origin() -> None:
    volume = np.broadcast_to(np.arange(16)[:, None, None], (16, 32, 32)).astype(np.float32)
    z, y, x, _ = common_physical_grid(
        source("a", 0.0), source("b", 3.0), target_spacing_um=2.0
    )
    sampled = resample_to_physical_grid(volume, source("a", 0.0), z, y, x)
    assert np.allclose(sampled[:, 0, 0], z)


def test_repeat_pair_recovers_known_3d_shift(tmp_path: Path) -> None:
    z, y, x = np.mgrid[:16, :32, :32]
    reference = (
        np.exp(-((z - 6) ** 2 + (y - 12) ** 2 + (x - 14) ** 2) / (2 * 2.0**2))
        + 0.7
        * np.exp(-((z - 10) ** 2 + (y - 23) ** 2 + (x - 21) ** 2) / (2 * 1.5**2))
    ).astype(np.float32)
    applied = np.asarray((1.2, -2.0, 1.5))
    moving = ndimage.shift(reference, applied, order=3, mode="constant")
    first_paths = {}
    second_paths = {}
    for method in METHODS:
        first_path = tmp_path / f"first_{method}.tif"
        second_path = tmp_path / f"second_{method}.tif"
        tifffile.imwrite(first_path, reference)
        tifffile.imwrite(second_path, moving)
        first_paths[method] = first_path
        second_paths[method] = second_path
    first = {
        "analysis_id": "first",
        "source": source("first"),
        "volumes": first_paths,
    }
    second = {
        "analysis_id": "second",
        "source": source("second"),
        "volumes": second_paths,
    }
    config = json.loads(
        (ROOT / "config" / "bench2p_zstack_xy_registration.json").read_text()
    )
    config["repeat_comparison"].update(
        {
            "target_spacing_um": 1.0,
            "landmark_min_distance_um": 3.0,
            "landmark_match_radius_um": 6.0,
        }
    )
    rows, _ = compare_pair(first, second, config)
    row = rows[0]
    estimated = np.asarray(
        [
            row["evaluation_shift_z_um"],
            row["evaluation_shift_y_um"],
            row["evaluation_shift_x_um"],
        ]
    )
    assert np.max(np.abs(estimated + applied)) < 0.45
    assert row["ncc_after_evaluation_alignment"] > row["ncc_before_evaluation_alignment"]
