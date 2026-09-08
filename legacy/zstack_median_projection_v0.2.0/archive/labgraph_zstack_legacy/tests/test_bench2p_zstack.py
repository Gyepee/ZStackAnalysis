from __future__ import annotations

import numpy as np

from labgraph_ops.workflows.bench2p_zstack import (
    StackMetadata,
    collapse_z_positions,
    estimate_rigid_shift,
    make_projections,
    merge_adjacent_stacks,
)


def stack_metadata(scan_id: str, z_positions: tuple[float, ...]) -> StackMetadata:
    return StackMetadata(
        source_path=f"/{scan_id}.tif",
        session_dir=f"/JJ_ROS-0001_2026-01-01_{scan_id}_sess1",
        animal_id="ROS-0001",
        date="2026-01-01",
        scan_id=scan_id,
        session_id="sess1",
        setup="bench2p",
        scanimage_config_path="Bench2P.cfg",
        scanimage_user_path="Bench2P.usr",
        scanimage_version="2022.1.0",
        stack_mode="slow",
        stack_enabled=True,
        n_slices=len(z_positions),
        frames_per_slice=2,
        expected_pages=len(z_positions) * 2,
        height_px=16,
        width_px=16,
        dtype="int16",
        z_step_um=1.0,
        z_positions_um=z_positions,
        fov_x_um=16.0,
        fov_y_um=16.0,
        pixel_size_x_um=1.0,
        pixel_size_y_um=1.0,
        frame_rate_hz=30.0,
        size_bytes=1,
        mtime_ns=1,
    )


def test_collapse_repeated_z_positions() -> None:
    assert collapse_z_positions([0, 0, 1, 1, 2, 2], 2) == (0.0, 1.0, 2.0)


def test_estimate_rigid_shift() -> None:
    reference = np.zeros((32, 32), dtype=np.float32)
    reference[9:15, 11:19] = 1
    moving = np.roll(np.roll(reference, 3, axis=0), -2, axis=1)
    dy, dx, _ = estimate_rigid_shift(reference, moving, upsample_factor=1)
    assert np.allclose((dy, dx), (-3, 2), atol=0.01)


def test_merge_adjacent_stacks_uses_one_overlap_plane() -> None:
    first = np.zeros((3, 16, 16), dtype=np.float32)
    second = np.zeros((3, 16, 16), dtype=np.float32)
    first[:, 5:9, 6:10] = np.asarray([1, 2, 3], dtype=np.float32)[:, None, None]
    second[:, 5:9, 6:10] = np.asarray([3, 4, 5], dtype=np.float32)[:, None, None]
    merged, z_positions, details = merge_adjacent_stacks(
        first,
        stack_metadata("scanA", (0.0, 1.0, 2.0)),
        second,
        stack_metadata("scanB", (2.01, 3.01, 4.01)),
        align_boundary_xy=True,
        upsample_factor=1,
        overlap_tolerance_um=0.25,
        max_shift_px=5.0,
    )
    assert merged.shape == (5, 16, 16)
    assert np.isclose(z_positions[2], 2.005)
    assert np.isclose(merged[2, 6, 7], 3.0)
    assert details["output_slice_count"] == 5


def test_make_projections_returns_finite_views() -> None:
    volume = np.zeros((8, 24, 24), dtype=np.float32)
    for index in range(8):
        volume[index, 7 + index // 2 : 11 + index // 2, 8:13] = index + 1
    projections, metadata = make_projections(
        volume,
        pixel_size_xy_um=1.2,
        z_step_um=1.0,
        low_percentile=1.0,
        high_percentile=99.8,
        gamma=0.75,
        azimuth_deg=-20,
        elevation_deg=28,
    )
    assert set(projections) == {"axial", "side", "oblique"}
    assert np.isfinite(projections["axial"]).all() and projections["axial"].max() > 0
    assert np.isfinite(projections["side"]).all() and projections["side"].max() > 0
    assert np.nanmax(projections["oblique"]) > 0
    assert np.isnan(projections["oblique"]).any()
    assert metadata["projection_method"].startswith("maximum_intensity")
    assert metadata["yaw_reshape"] is True
    assert metadata["oblique_background"].startswith("transparent_outside")
