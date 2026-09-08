"""Collection-manifest admission and identity reconciliation."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import replace
from pathlib import Path
from typing import Any


ANIMAL_RE = re.compile(r"^[A-Z]+-\d+$")
MANIFEST_NAME = "collection_manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_collection_manifest(
    source_tiff: Path, metadata: Any
) -> tuple[Any, dict[str, Any]]:
    """Validate a colocated manifest and apply its declared animal identity."""
    path = source_tiff.parent / MANIFEST_NAME
    if not path.exists():
        return metadata, {
            "status": "absent",
            "path": None,
            "sha256": None,
            "identity_source": "session_path",
            "path_animal_id": metadata.animal_id,
            "declared_animal_id": None,
            "animal_id_corrected": False,
        }

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "1.0":
        raise ValueError(f"Unsupported collection manifest schema: {path}")
    if payload.get("acquisition_type") != "zstack_imaging":
        raise ValueError(f"Collection manifest is not zstack_imaging: {path}")
    if payload.get("setup_type") != metadata.setup:
        raise ValueError("Collection manifest setup_type conflicts with TIFF metadata")
    declared_scan = str(payload.get("scan_id", ""))
    if declared_scan.removeprefix("scan") != metadata.scan_id.removeprefix("scan"):
        raise ValueError("Collection manifest scan_id conflicts with session identity")

    value = payload.get("collected_files")
    entries = value if isinstance(value, list) else [value]
    matches = [item for item in entries if isinstance(item, dict) and item.get("name") == source_tiff.name]
    if len(matches) != 1:
        raise ValueError("Selected TIFF is not uniquely declared in collection_manifest.json")
    if int(matches[0].get("bytes", -1)) != source_tiff.stat().st_size:
        raise ValueError("Collection manifest byte count conflicts with selected TIFF")

    animal_id = str(payload.get("animal_id", ""))
    if not ANIMAL_RE.fullmatch(animal_id):
        raise ValueError(f"Invalid collection manifest animal_id: {animal_id!r}")
    path_animal_id = metadata.animal_id
    resolved = replace(metadata, animal_id=animal_id)
    return resolved, {
        "status": "pass",
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "identity_source": "collection_manifest",
        "path_animal_id": path_animal_id,
        "declared_animal_id": animal_id,
        "animal_id_corrected": animal_id != path_animal_id,
        "selected_file": matches[0],
        "z_stack": payload.get("z_stack"),
    }
