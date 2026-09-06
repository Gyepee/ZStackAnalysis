# Proposal: one acquisition manifest for all raw session folders

The proposed canonical filename is `acquisition_manifest.json`. LabGraph should own
and version the schema. This repository contains a reviewable proposal under
`docs/proposals/`; it is not yet a frozen LabGraph contract.

## Why this is separate from external copy status

The observed `external_copy_status.json` is an upload-worker event log. It records
Windows/UNC source and destination paths, selection notes, and per-item OK/FAIL status.
It does not reliably establish server-side file identity because it lacks byte counts
and content hashes. LabGraph's own documents also currently disagree on whether it is
an ingest gate or provenance-only context.

The integrated acquisition manifest references and hashes that log but independently
records the files actually present on the server. The status reported by the copy
worker and the result of server verification are deliberately separate fields.

## One folder can contain multiple modalities

`components` is an array rather than a single `data_kind`. A behavior + mini2p session
can declare behavior, neural imaging, tracking, and AUX components in one folder. A
Bench2p Z-stack folder declares one `bench2p_zstack` component. Each component points
to stable `asset_id` values in the shared asset inventory.

The common manifest holds only cross-modality identity and provenance. Component
objects may add modality-specific metadata under `metadata`; LabGraph should publish
separate component schemas once the required fields are stable.

## Backfilling existing folders

Backfill should be a two-stage LabGraph workflow:

1. **Dry-run inventory:** discover actual folders, parse folder identity, inventory
   files, read the legacy copy log, classify tentative components, and produce a report
   without writing the source folders.
2. **Approved materialization:** write only manifests whose folder identity, assets,
   component classification, and required metadata are unambiguous. Route all others
   to a review queue with explicit reasons.

Do not turn a legacy copy `DONE` into server verification `pass` automatically. Check
actual files and compute hashes. Do not invent missing surgery dates, protocols, or
modality labels from the schedule alone. Backfilled manifests use
`creation_mode: backfill` and preserve the legacy copy-status JSON unchanged.

Because this materialization writes to shared raw-data folders, the executing agent
must receive an explicit target scope (for example animals and date range) and use the
canonical LabGraph/ResearchDataGovernance workflow. A broad recursive write based only
on folder-name matches is not an acceptable first run.

## Migration for ZStackAnalysis

ZStackAnalysis v0.1.0 currently accepts the narrower `zstack_acquisition.json` while
the universal contract is being reviewed. Once LabGraph freezes
`labgraph.acquisition_manifest.v1`, the next pipeline version should prefer
`acquisition_manifest.json`, resolve exactly one `bench2p_zstack` component and its
TIFF/channel, and retain support for the narrow declaration only as an explicit legacy
adapter. The manifest used by a run must always be snapshotted and hashed.
