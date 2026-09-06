from __future__ import annotations

import numpy as np

from zstack_analysis.pipeline import (
    median_plane,
    physical_projections,
    resolve_source_channel,
    source_page_index,
    validate_z_geometry,
)
from zstack_analysis.scanimage_io import normalize_saved_channels


def test_median_plane_resists_minority_motion_outlier() -> None:
    reference = np.zeros((12, 12), dtype=np.float32)
    reference[4:8, 4:8] = 10
    frames = np.repeat(reference[None], 9, axis=0)
    frames[-2:] = np.roll(reference, shift=4, axis=1)
    np.testing.assert_array_equal(median_plane(frames), reference)


def test_mean_projection_is_not_maximum_projection() -> None:
    volume = np.zeros((4, 3, 2), dtype=np.float32)
    volume[0] = 8
    projections = physical_projections(volume)
    np.testing.assert_allclose(projections["mean_xy_over_z"], 2)
    np.testing.assert_allclose(projections["max_xy_over_z"], 8)


def test_regular_one_and_two_micron_z_geometry_pass() -> None:
    for step in (1.0, 2.0):
        result = validate_z_geometry(
            np.arange(100) * step,
            step,
            tolerance_um=1e-6,
            require_increasing=True,
        )
        assert result["status"] == "pass"


def test_irregular_z_geometry_fails() -> None:
    result = validate_z_geometry(
        (0.0, 1.0, 2.2),
        1.0,
        tolerance_um=1e-6,
        require_increasing=True,
    )
    assert result["status"] == "fail"


def test_saved_channel_normalization_handles_scalar_and_sequence() -> None:
    assert normalize_saved_channels(3) == (3,)
    assert normalize_saved_channels([3, 4]) == (3, 4)


def test_multichannel_selection_is_explicit_and_uses_ztc_page_order() -> None:
    assert resolve_source_channel((3,), None) == (3, 0)
    assert resolve_source_channel((3, 4), 4) == (4, 1)
    assert source_page_index(17, 1, 2) == 35


def test_multichannel_selection_without_channel_fails() -> None:
    try:
        resolve_source_channel((3, 4), None)
    except ValueError as error:
        assert "specify --channel" in str(error)
    else:
        raise AssertionError("Expected multichannel input to require a channel")
