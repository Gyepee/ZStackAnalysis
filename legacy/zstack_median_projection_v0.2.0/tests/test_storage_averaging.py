from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import tifffile

from zstack_analysis.pipeline import reconstruct_median_volume
from zstack_analysis.scanimage_io import derive_storage_layout


def test_scanimage_log_average_layout_is_explicit() -> None:
    assert derive_storage_layout(60, 1, False) == (60, "raw_frame")
    assert derive_storage_layout(60, 60, False) == (1, "logged_mean")
    assert derive_storage_layout(60, 60, True) == (1, "logged_sum")


def test_one_logged_mean_per_plane_reconstructs_without_claiming_raw_frames(
    tmp_path: Path,
) -> None:
    source = tmp_path / "logged_means.tif"
    first = np.arange(20, dtype=np.int16).reshape(4, 5)
    second = first + 100
    with tifffile.TiffWriter(source) as writer:
        writer.write(first)
        writer.write(second)

    metadata = SimpleNamespace(
        n_slices=2,
        height_px=4,
        width_px=5,
        frames_per_slice=60,
        stored_frames_per_slice=1,
        log_average_factor=60,
        storage_aggregation="logged_mean",
        expected_tiff_pages=2,
        saved_channels=(3,),
        frame_rate_hz=30.0,
        z_positions_um=(0.0, 2.0),
        scan_id="scan-test",
    )
    config = {
        "aggregation": {"discard_initial_stored_images_per_plane": 0},
        "within_plane_qc": {
            "correlation_stride_px": 1,
            "descriptive_low_correlation_mad_k": 3.0,
        },
    }

    volume, stored_rows, plane_rows, page_count = reconstruct_median_volume(
        metadata,
        source,
        config,
        source_channel=3,
        channel_position=0,
    )

    np.testing.assert_array_equal(volume[0], first)
    np.testing.assert_array_equal(volume[1], second)
    assert page_count == 2
    assert stored_rows[0]["acquisition_frame_start_index"] == 0
    assert stored_rows[0]["acquisition_frame_end_index"] == 59
    assert stored_rows[1]["acquisition_frame_start_index"] == 60
    assert stored_rows[1]["acquisition_frame_end_index"] == 119
    assert plane_rows[0]["stored_images_total"] == 1
    assert plane_rows[0]["source_storage_aggregation"] == "logged_mean"
