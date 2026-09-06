from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from zstack_analysis.version import PIPELINE_ID, PIPELINE_STAGE, PIPELINE_VERSION
from zstack_analysis.workflow import (
    DEFAULT_ANALYSIS_SPEC,
    DEFAULT_CONFIG,
    DEFAULT_FIGURE_STYLE,
    attach_identity,
    configure_matplotlib,
    create_analysis_id,
    resolve_acquisition_context,
    save_projection_figures,
    validate_acquisition_spec,
    validate_analysis_spec,
    validate_figure_style,
)


def _metadata(source: Path) -> SimpleNamespace:
    return SimpleNamespace(
        animal_id="ROS-2335",
        date="2026-09-01",
        scan_id="scanABC",
        session_id="sessABC",
        n_slices=3,
        frames_per_slice=4,
        width_px=8,
        height_px=6,
        pixel_size_x_um=1.0,
        pixel_size_y_um=1.5,
        z_positions_um=(0.0, 2.0, 4.0),
        z_step_um=2.0,
        saved_channels=(3,),
        source_path=str(source),
    )


def test_pipeline_identity_is_fixed_to_initial_development_version() -> None:
    assert PIPELINE_ID == "zstack_median_projection"
    assert PIPELINE_VERSION == "0.1.0"
    assert PIPELINE_STAGE == "initial_development"


def test_default_specs_match_running_pipeline() -> None:
    analysis_spec = json.loads(DEFAULT_ANALYSIS_SPEC.read_text())
    config = json.loads(DEFAULT_CONFIG.read_text())
    style = json.loads(DEFAULT_FIGURE_STYLE.read_text())
    validate_analysis_spec(analysis_spec)
    validate_figure_style(style, config["figure"]["style_profile"])


def test_acquisition_declaration_must_match_tiff_identity(tmp_path: Path) -> None:
    source = tmp_path / "scanABC_JJ_ROS-2335_00001.tif"
    source.write_bytes(b"test")
    declaration_path = tmp_path / "zstack_acquisition.json"
    declaration = {
        "schema_version": "zstack_analysis.acquisition.v1",
        "data_kind": "zstack",
        "animal_id": "ROS-2335",
        "acquisition_date": "2026-09-01",
        "scan_id": "scanABC",
        "session_id": "sessABC",
        "source_tiff": source.name,
        "source_channel": 3,
        "surgery_date": "2026-08-28",
        "post_surgery_day": 4,
        "acquisition_purpose": "deep FOV recovery",
    }
    declaration_path.write_text(json.dumps(declaration))
    result = validate_acquisition_spec(
        declaration, declaration_path, source, _metadata(source)
    )
    assert result["computed_post_surgery_day"] == 4
    declaration["animal_id"] = "ROS-2338"
    try:
        validate_acquisition_spec(declaration, declaration_path, source, _metadata(source))
    except ValueError as error:
        assert "conflicts" in str(error)
    else:
        raise AssertionError("Expected conflicting acquisition identity to fail")


def test_legacy_scan_without_declaration_is_explicitly_incomplete(tmp_path: Path) -> None:
    source = tmp_path / "scanABC_JJ_ROS-2335_00001.tif"
    source.write_bytes(b"test")
    context = resolve_acquisition_context(source, _metadata(source), None)
    assert context["status"] == "legacy_metadata_incomplete"
    assert context["post_surgery_day"] is None


def test_analysis_id_and_table_rows_carry_stable_identity() -> None:
    metadata = _metadata(Path("source.tif"))
    created_at = datetime(2026, 9, 6, 12, 34, 56, tzinfo=timezone.utc)
    analysis_id = create_analysis_id(
        metadata,
        3,
        created_at,
        {"source_sha256": "abc", "config_sha256": "def"},
    )
    assert "zstack_median_projection-v0-1-0" in analysis_id
    assert "ROS-2335_2026-09-01_scanABC_sessABC_ch3" in analysis_id
    rows = attach_identity(
        [{"plane_index": 0}],
        {
            "analysis_id": analysis_id,
            "animal_id": "ROS-2335",
            "acquisition_date": "2026-09-01",
            "scan_id": "scanABC",
        },
    )
    assert rows[0]["analysis_id"] == analysis_id
    assert rows[0]["scan_id"] == "scanABC"


def test_each_figure_exports_svg_png_and_exact_arrays(tmp_path: Path) -> None:
    source = tmp_path / "source.tif"
    source.write_bytes(b"test")
    metadata = _metadata(source)
    config = json.loads(DEFAULT_CONFIG.read_text())
    style = json.loads(DEFAULT_FIGURE_STYLE.read_text())
    configure_matplotlib(style, "qc")
    (tmp_path / "figures").mkdir()
    volume = np.arange(3 * 6 * 8, dtype=np.float32).reshape(3, 6, 8)
    _, rows, descriptors = save_projection_figures(
        tmp_path,
        volume,
        metadata,
        config,
        style,
        "analysis-test-12345678",
        3,
        4,
        "QC pass",
    )
    assert len(rows) == 6
    assert len(descriptors) == 3
    arrays = np.load(tmp_path / "figures" / "projection_source_arrays_float32.npz")
    assert "representative_median_plane" in arrays.files
    for descriptor in descriptors:
        assert descriptor["paths"]["svg"].exists()
        assert descriptor["paths"]["png"].exists()
