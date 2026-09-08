from pathlib import Path

import numpy as np
import tifffile

from labgraph_ops.workflows.bench2p_zstack_sideview import (
    alpha_composite_volume,
    attenuated_mip,
    cuboid_edge_mask,
    detect_bright_candidates,
    make_cuboid_oblique_render,
    normalize_volume_for_display,
    physical_spacing_from_ome,
    local_oblique_view,
    resolve_saturation_offsets,
    slab_half_width,
)


def test_normalize_volume_uses_one_global_range() -> None:
    volume = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    normalized, metadata = normalize_volume_for_display(
        volume,
        low_percentile=0,
        high_percentile=100,
        asinh_strength=0,
    )
    assert normalized.shape == volume.shape
    assert normalized[0, 0, 0] == 0
    assert normalized[-1, -1, -1] == 1
    assert metadata["low_intensity"] == 0
    assert metadata["high_intensity"] == 23


def test_slab_half_width_is_odd_and_near_requested_thickness() -> None:
    half_width = slab_half_width(6.0, 1.2)
    voxel_count = 2 * half_width + 1
    assert voxel_count % 2 == 1
    assert voxel_count == 5


def test_attenuated_mip_prefers_nearer_equal_signal() -> None:
    volume = np.zeros((3, 5, 3), dtype=np.float32)
    volume[1, 1, 1] = 1
    volume[1, 3, 1] = 1
    rendered = attenuated_mip(volume, axis=1, threshold=0.1, attenuation=1.0)
    assert rendered[1, 1] == 1


def test_cuboid_render_has_all_edges_and_real_margin() -> None:
    volume = np.zeros((12, 24, 30), dtype=np.float32)
    volume[3:9, 8:16, 10:20] = 0.8
    edge_volume = cuboid_edge_mask(volume.shape, thickness_px=1)
    assert edge_volume[:, 0, 0].all()
    assert edge_volume[0, :, 0].all()
    assert edge_volume[0, 0, :].all()
    composite = alpha_composite_volume(
        volume, axis=0, threshold=0.2, opacity=0.15
    )
    assert composite.shape == volume.shape[1:]
    assert composite.max() > 0
    layers, metadata = make_cuboid_oblique_render(
        volume,
        (2.0, 1.0, 1.0),
        {
            "azimuth_deg": -35.0,
            "elevation_deg": 32.0,
            "max_dimension_px": 64,
            "signal_threshold": 0.2,
            "opacity": 0.15,
            "edge_thickness_px": 2,
            "edge_dilation_px": 1,
            "padding_fraction": 0.12,
        },
    )
    assert layers["render"].shape == layers["support"].shape == layers["edges"].shape
    assert np.isnan(layers["render"][0, 0])
    assert layers["edges"].any()
    assert metadata["wireframe"] == "all_12_physical_acquisition_cuboid_edges"
    assert metadata["padding_px"] >= 12


def test_local_oblique_masks_outside_rotated_cuboid() -> None:
    volume = np.ones((5, 12, 14), dtype=np.float32)
    rendered = local_oblique_view(
        volume,
        (1.0, 1.0, 1.0),
        {
            "azimuth_deg": -25.0,
            "elevation_deg": 20.0,
            "signal_threshold": 0.08,
            "attenuation": 0.12,
        },
    )
    assert np.nanmax(rendered) > 0
    assert np.isnan(rendered).any()


def test_saturation_offsets_scale_older_sampling_grid() -> None:
    offsets, rule = resolve_saturation_offsets([0, 25, 50, 75, 99], 60)
    assert offsets == [0, 15, 30, 45, 59]
    assert rule == "normalized_frame_positions_per_slice"


def test_saturation_offsets_keep_valid_indices() -> None:
    offsets, rule = resolve_saturation_offsets([0, 15, 30, 45, 59], 60)
    assert offsets == [0, 15, 30, 45, 59]
    assert rule == "fixed_local_frame_indices_per_slice"


def test_detect_bright_candidates_returns_physical_coordinates() -> None:
    volume = np.zeros((25, 40, 40), dtype=np.float32)
    volume[12, 20, 22] = 100
    config = {
        "background_sigma_um": [4, 4, 4],
        "feature_sigma_um": [1, 1, 1],
        "threshold_percentile": 99.0,
        "minimum_distance_voxels": 3,
        "exclude_border_voxels": [3, 3, 3],
        "candidate_count": 1,
    }
    candidates, feature = detect_bright_candidates(volume, (1.0, 2.0, 2.0), config)
    assert feature.shape == volume.shape
    assert len(candidates) == 1
    candidate = candidates[0]
    assert (candidate.z_index, candidate.y_index, candidate.x_index) == (12, 20, 22)
    assert (candidate.z_um, candidate.y_um, candidate.x_um) == (12.0, 40.0, 44.0)


def test_physical_spacing_from_ome(tmp_path: Path) -> None:
    path = tmp_path / "volume.ome.tif"
    tifffile.imwrite(
        path,
        np.zeros((3, 4, 5), dtype=np.float32),
        ome=True,
        metadata={
            "axes": "ZYX",
            "PhysicalSizeZ": 1.0,
            "PhysicalSizeZUnit": "µm",
            "PhysicalSizeY": 1.2,
            "PhysicalSizeYUnit": "µm",
            "PhysicalSizeX": 1.3,
            "PhysicalSizeXUnit": "µm",
        },
    )
    assert physical_spacing_from_ome(path) == (1.0, 1.2, 1.3)
