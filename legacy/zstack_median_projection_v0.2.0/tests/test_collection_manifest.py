from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from zstack_analysis.collection_manifest import resolve_collection_manifest


@dataclass(frozen=True)
class Metadata:
    animal_id: str = "ROS-2338"
    scan_id: str = "scanABC"
    setup: str = "bench2p"


def test_collection_manifest_can_correct_path_animal_identity(tmp_path: Path) -> None:
    source = tmp_path / "scanABC_JJ_ROS-2338_02031.tif"
    source.write_bytes(b"test")
    manifest = {
        "schema_version": "1.0",
        "setup_type": "bench2p",
        "animal_id": "ROS-2335",
        "scan_id": "ABC",
        "acquisition_type": "zstack_imaging",
        "collected_files": {"name": source.name, "bytes": 4},
        "z_stack": {"enabled": True},
    }
    (tmp_path / "collection_manifest.json").write_text(json.dumps(manifest))

    resolved, context = resolve_collection_manifest(source, Metadata())

    assert resolved.animal_id == "ROS-2335"
    assert context["animal_id_corrected"] is True
    assert context["path_animal_id"] == "ROS-2338"


def test_manifest_must_uniquely_declare_selected_tiff(tmp_path: Path) -> None:
    source = tmp_path / "scanABC_JJ_ROS-2338_02031.tif"
    source.write_bytes(b"test")
    manifest = {
        "schema_version": "1.0",
        "setup_type": "bench2p",
        "animal_id": "ROS-2338",
        "scan_id": "ABC",
        "acquisition_type": "zstack_imaging",
        "collected_files": {"name": "other.tif", "bytes": 4},
    }
    (tmp_path / "collection_manifest.json").write_text(json.dumps(manifest))

    try:
        resolve_collection_manifest(source, Metadata())
    except ValueError as error:
        assert "not uniquely declared" in str(error)
    else:
        raise AssertionError("Expected undeclared TIFF admission to fail")
