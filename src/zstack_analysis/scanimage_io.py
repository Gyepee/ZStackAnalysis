"""ScanImage metadata and OME-TIFF helpers owned by ZStackAnalysis."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tifffile


SESSION_RE = re.compile(
    r"^(?P<operator>[^_]+)_(?P<animal>[A-Z]+-\d+)_(?P<date>\d{4}-\d{2}-\d{2})_"
    r"(?P<scan>scan[^_]+)_(?P<session>sess[^_]+)$"
)


@dataclass(frozen=True)
class StackMetadata:
    source_path: str
    session_dir: str
    animal_id: str
    date: str
    scan_id: str
    session_id: str
    setup: str
    scanimage_config_path: str
    scanimage_user_path: str
    scanimage_version: str
    stack_mode: str
    stack_enabled: bool
    saved_channels: tuple[int, ...]
    n_slices: int
    frames_per_slice: int
    expected_pages: int
    expected_tiff_pages: int
    height_px: int
    width_px: int
    dtype: str
    z_step_um: float
    z_positions_um: tuple[float, ...]
    fov_x_um: float
    fov_y_um: float
    pixel_size_x_um: float
    pixel_size_y_um: float
    frame_rate_hz: float
    size_bytes: int
    mtime_ns: int


def normalize_setup(text: str) -> str:
    compact = text.lower().replace("-", "").replace("_", "")
    for setup in ("bench2p", "mini2p1", "mini2p2"):
        if setup in compact:
            return setup
    return "unknown"


def _float(value: Any, field: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Missing or invalid ScanImage metadata field {field}: {value!r}"
        ) from exc


def _positive_int(value: Any, field: str) -> int:
    result = int(round(_float(value, field)))
    if result <= 0:
        raise ValueError(f"ScanImage metadata field {field} must be positive")
    return result


def normalize_saved_channels(value: Any) -> tuple[int, ...]:
    """Return ScanImage channelSave as an ordered tuple of channel IDs."""
    if isinstance(value, np.ndarray):
        values = value.reshape(-1).tolist()
    elif isinstance(value, (list, tuple)):
        values = list(value)
    else:
        values = [value]
    channels = tuple(
        _positive_int(item, "SI.hChannels.channelSave") for item in values
    )
    if len(set(channels)) != len(channels):
        raise ValueError("SI.hChannels.channelSave contains duplicate channels")
    return channels


def collapse_z_positions(values: Any, frames_per_slice: int) -> tuple[float, ...]:
    if not isinstance(values, (list, tuple, np.ndarray)):
        raise ValueError("SI.hStackManager.zs is not a sequence")
    flat: list[float] = []
    for item in values:
        if isinstance(item, (list, tuple, np.ndarray)):
            if len(item) == 0:
                continue
            item = item[0]
        flat.append(float(item))
    if not flat:
        raise ValueError("SI.hStackManager.zs is empty")
    collapsed = [flat[0]]
    for value in flat[1:]:
        if not np.isclose(value, collapsed[-1], rtol=0.0, atol=1e-6):
            collapsed.append(value)
    if len(collapsed) == 1 and len(flat) >= frames_per_slice:
        collapsed = [flat[index] for index in range(0, len(flat), frames_per_slice)]
    return tuple(collapsed)


def parse_stack_metadata(path: Path) -> StackMetadata:
    """Parse one canonical uploaded ScanImage TIFF without changing it."""
    path = path.resolve()
    session_match = SESSION_RE.match(path.parent.name)
    if not session_match:
        raise ValueError(f"Unexpected session folder name: {path.parent.name}")
    stat = path.stat()
    with tifffile.TiffFile(path) as tif:
        metadata = tif.scanimage_metadata or {}
        frame_data = metadata.get("FrameData", {}) if isinstance(metadata, dict) else {}
        first_page = tif.pages[0]
        height, width = map(int, first_page.shape)
        dtype = str(first_page.dtype)
    config_path = str(frame_data.get("SI.hConfigurationSaver.cfgFilename", ""))
    user_path = str(frame_data.get("SI.hConfigurationSaver.usrFilename", ""))
    saved_channels = normalize_saved_channels(
        frame_data.get("SI.hChannels.channelSave")
    )
    n_slices = _positive_int(
        frame_data.get("SI.hStackManager.actualNumSlices"),
        "SI.hStackManager.actualNumSlices",
    )
    frames_per_slice = _positive_int(
        frame_data.get("SI.hStackManager.framesPerSlice"),
        "SI.hStackManager.framesPerSlice",
    )
    z_positions = collapse_z_positions(
        frame_data.get("SI.hStackManager.zs"), frames_per_slice
    )
    if len(z_positions) != n_slices:
        raise ValueError(
            f"Z-position count {len(z_positions)} does not match actualNumSlices {n_slices}"
        )
    fov = np.asarray(frame_data.get("SI.hRoiManager.imagingFovUm"), dtype=float)
    if fov.shape != (4, 2):
        raise ValueError(f"Unexpected imagingFovUm shape: {fov.shape}")
    fov_x_um = float(np.ptp(fov[:, 0]))
    fov_y_um = float(np.ptp(fov[:, 1]))
    version = ".".join(
        str(frame_data.get(key, ""))
        for key in ("SI.VERSION_MAJOR", "SI.VERSION_MINOR", "SI.VERSION_UPDATE")
    )
    parts = session_match.groupdict()
    return StackMetadata(
        source_path=str(path),
        session_dir=str(path.parent.resolve()),
        animal_id=parts["animal"],
        date=parts["date"],
        scan_id=parts["scan"],
        session_id=parts["session"],
        setup=normalize_setup(" ".join((config_path, user_path, path.parent.name))),
        scanimage_config_path=config_path,
        scanimage_user_path=user_path,
        scanimage_version=version,
        stack_mode=str(frame_data.get("SI.hStackManager.stackMode", "")),
        stack_enabled=bool(frame_data.get("SI.hStackManager.enable", False)),
        saved_channels=saved_channels,
        n_slices=n_slices,
        frames_per_slice=frames_per_slice,
        expected_pages=n_slices * frames_per_slice,
        expected_tiff_pages=n_slices * frames_per_slice * len(saved_channels),
        height_px=height,
        width_px=width,
        dtype=dtype,
        z_step_um=_float(
            frame_data.get("SI.hStackManager.actualStackZStepSize"),
            "SI.hStackManager.actualStackZStepSize",
        ),
        z_positions_um=z_positions,
        fov_x_um=fov_x_um,
        fov_y_um=fov_y_um,
        pixel_size_x_um=fov_x_um / width,
        pixel_size_y_um=fov_y_um / height,
        frame_rate_hz=_float(
            frame_data.get("SI.hRoiManager.scanFrameRate"),
            "SI.hRoiManager.scanFrameRate",
        ),
        size_bytes=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
    )


def normalized_correlation(left: np.ndarray, right: np.ndarray) -> float:
    mask = np.isfinite(left) & np.isfinite(right)
    if int(mask.sum()) < 2:
        return float("nan")
    first = left[mask].astype(np.float64)
    second = right[mask].astype(np.float64)
    first -= first.mean()
    second -= second.mean()
    denominator = np.sqrt(np.sum(first * first) * np.sum(second * second))
    return float(np.sum(first * second) / denominator) if denominator else float("nan")


def write_ome_volume(
    path: Path,
    volume: np.ndarray,
    *,
    pixel_size_x_um: float,
    pixel_size_y_um: float,
    z_step_um: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(
        path,
        np.asarray(volume, dtype=np.float32),
        ome=True,
        bigtiff=volume.nbytes > 4 * 1024**3,
        metadata={
            "axes": "ZYX",
            "PhysicalSizeX": float(pixel_size_x_um),
            "PhysicalSizeXUnit": "µm",
            "PhysicalSizeY": float(pixel_size_y_um),
            "PhysicalSizeYUnit": "µm",
            "PhysicalSizeZ": float(z_step_um),
            "PhysicalSizeZUnit": "µm",
        },
    )
