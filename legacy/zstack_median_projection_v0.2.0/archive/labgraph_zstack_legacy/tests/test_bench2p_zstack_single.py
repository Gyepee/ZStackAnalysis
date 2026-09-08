from __future__ import annotations

import numpy as np
from scipy import ndimage

from labgraph_ops.workflows.bench2p_zstack_single import apply_cross_plane_registration


def synthetic_drifting_volume(
    n_slices: int, cumulative_shifts: list[tuple[float, float]]
) -> np.ndarray:
    base = np.zeros((48, 48), dtype=np.float32)
    base[18:30, 20:28] = 1.0
    volume = np.empty((n_slices, 48, 48), dtype=np.float32)
    for index in range(n_slices):
        dy, dx = cumulative_shifts[index]
        volume[index] = ndimage.shift(base, shift=(dy, dx), order=1, mode="nearest")
    return volume


def test_apply_cross_plane_registration_recovers_known_drift() -> None:
    n_slices = 6
    # Injected content drift per plane, relative to plane 0. The registration's
    # cumulative correction should be the negative of this (it undoes the drift).
    true_drift = [(0.0, 0.0), (1.0, -0.5), (2.0, -1.0), (3.0, -1.5), (4.0, -2.0), (5.0, -2.5)]
    volume = synthetic_drifting_volume(n_slices, true_drift)
    z_positions = tuple(float(i) for i in range(n_slices))

    corrected, rows = apply_cross_plane_registration(
        volume,
        z_positions,
        mad_k=3.0,
        max_step_shift_px=10.0,
        upsample_factor=10,
    )

    assert len(rows) == n_slices
    for index, row in enumerate(rows):
        assert row["step_applied"] == (index != 0)
        expected_dy, expected_dx = -true_drift[index][0], -true_drift[index][1]
        assert abs(row["cumulative_shift_y_px"] - expected_dy) < 0.15
        assert abs(row["cumulative_shift_x_px"] - expected_dx) < 0.15

    reference = corrected[0]
    for plane in corrected[1:]:
        assert np.corrcoef(reference.ravel(), plane.ravel())[0, 1] > 0.95


def test_apply_cross_plane_registration_rejects_decorrelated_step() -> None:
    rng = np.random.default_rng(0)
    n_slices = 5
    volume = np.zeros((n_slices, 32, 32), dtype=np.float32)
    volume[:, 10:22, 12:20] = 1.0
    # Replace one plane with pure noise: adjacent-plane correlation collapses,
    # so the candidate shift for that step must not be trusted.
    volume[2] = rng.normal(size=(32, 32)).astype(np.float32)

    corrected, rows = apply_cross_plane_registration(
        volume,
        tuple(float(i) for i in range(n_slices)),
        mad_k=3.0,
        max_step_shift_px=10.0,
        upsample_factor=10,
    )

    assert rows[2]["step_applied"] is False
    assert rows[2]["cumulative_shift_y_px"] == rows[1]["cumulative_shift_y_px"]
    assert rows[2]["cumulative_shift_x_px"] == rows[1]["cumulative_shift_x_px"]


def test_apply_cross_plane_registration_rejects_oversized_shift() -> None:
    base = np.zeros((32, 32), dtype=np.float32)
    base[8:24, 8:24] = 1.0
    volume = np.stack(
        [
            base,
            base,
            ndimage.shift(base, shift=(20.0, 0.0), order=1, mode="nearest"),
        ]
    ).astype(np.float32)

    _, rows = apply_cross_plane_registration(
        volume,
        (0.0, 1.0, 2.0),
        mad_k=3.0,
        max_step_shift_px=10.0,
        upsample_factor=1,
    )

    assert rows[2]["step_applied"] is False
    assert rows[2]["cumulative_shift_y_px"] == 0.0
