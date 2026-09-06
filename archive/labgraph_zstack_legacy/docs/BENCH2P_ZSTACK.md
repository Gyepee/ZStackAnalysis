# Bench2p Z-stack workflow

This exploratory workflow reconstructs a static three-dimensional fluorescence
volume from ScanImage bench2p slow stacks acquired as consecutive frames at
each Z position. It does not use Bpod, AUX, DataJoint ingest, suite2p ROI
extraction, or same-day mini2p2 acquisitions.

## Admission checks

- Source folders are discovered only under the explicit `--data-root`.
- Exact scan IDs must be supplied; unrelated bench2p tests are not inferred.
- The TIFF ScanImage configuration must identify `bench2p`.
- `SI.hStackManager.enable` must be true and stack mode must be `slow`.
- TIFF page count must equal `actualNumSlices × framesPerSlice`.
- The two stacks must share animal, date, frame shape, frames per slice, XY
  calibration, and Z step.
- Version 1 expects one overlapping boundary plane. It stops if the boundary Z
  positions or rigid XY alignment exceed configured limits.

The uploaded raw TIFFs are read-only. A failed check stops the run and does not
change DataJoint or the source session folder.

## Computation

1. Read each source slice's consecutive frames.
2. Compute the arithmetic mean in float precision without changing raw values.
3. Estimate descriptive within-slice drift from the first and last small frame
   groups. Version 1 reports this QC but does not motion-correct individual
   frames.
4. Rigidly align the second stack to the first at the overlapping boundary
   plane.
5. Average the two representations of the boundary plane and retain the other
   planes once.
6. Write source means and the combined float32 volume as OME-TIFF with measured
   XY and Z voxel sizes.
7. Produce axial, side, and oblique maximum-intensity projections. The oblique
   panel is made after physical Z-to-XY resampling and two declared rotations.

The display percentile and gamma operations affect only figures, not the
stored mean volume.

## Example

```bash
cd /path/to/LabGraph
PYTHONPATH=src python -m labgraph_ops.workflows.bench2p_zstack_run \
  --data-root /path/to/uploaded/sessions \
  --animal ROS-2335 \
  --date 2026-08-27 \
  --scan scan9G1WRJ8Y \
  --scan scan9G1WRM40
```

Large generated runs are stored under
`datasets/analysis_runs/<analysis_id>/`. Each immutable run contains source and
analysis manifests, exact hashes, configuration/environment snapshots, QC,
slice tables, OME-TIFF volumes, and a complete figure bundle.

## Interpretation boundary

The result is a sequential static Z-stack, not simultaneous or time-resolved
volumetric activity. It is suitable for exploratory anatomy and acquisition QC.
It does not establish cell identity, connectivity, population dynamics, or a
biological condition effect.

## Interactive FOV review boundary

LabGraph owns source verification, reconstruction, physical calibration, and
the provenance-rich OME-TIFF. Interactive pre-acquisition FOV review now lives
in the sibling `gRSC_NavigatorV2` workspace under
`src/grsc_v2/fov_explorer/`; it reads the OME-TIFF in place and does not copy the
volume. Keep the historical LabGraph side-view renderer for completed-run
reproducibility, but do not use its fixed bright-peak panels as an automatic FOV
decision.
