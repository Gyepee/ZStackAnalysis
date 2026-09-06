# ZStackAnalysis governance and LabGraph boundary

## Ownership model

LabGraph is the control tower. It owns the canonical rules for data discovery,
database/data-access boundaries, analysis-run provenance, figure handoff, and
downstream registration. ZStackAnalysis is a specialized department. It owns its
runtime reconstruction code, tests, versioned analysis specification, and generated
Z-stack run bundles.

ZStackAnalysis does not import a sibling LabGraph checkout at runtime. Adopted policy
semantics and visual settings are frozen locally, attributed to LabGraph, hashed in
each run, and changed through a new ZStackAnalysis version when behavior changes.

## Scientific scope

The project asks when, after surgery, a candidate FOV has recovered enough—including
at depth—to support an informed mini2p mounting-location decision. Surgery-day blood
and optical disturbance are plausible confounds. Interpretations remain provisional:
brightness, contrast, and visibility can also change with acquisition settings,
illumination, motion, tissue state, and other technical factors.

Version 0.1.0 reconstructs and documents one scan/channel at a time. It does not yet
define or validate the endpoint that decides “optimal day.” The next analysis layer
must compare repeated days within animal before any animal-balanced group summary.

## Data contract

The canonical acquisition-side declaration is `zstack_acquisition.json`, stored next
to the source TIFF. Its stable identifiers must agree with ScanImage metadata. The
pipeline reads sources but never edits them. Source review records live separately in
`review_flags/`, are append-only, and can place a scan on hold without mutating an
already completed run.

Raw TIFF access is intentional here because reconstruction is the analysis boundary,
not because the aligned-dataset rule is being bypassed. The manifest records this as
`input_boundary.kind = raw_scanimage_tiff` and states the purpose.

## Handoff contract

A LabGraph consumer should accept only a completed directory whose
`analysis_manifest.json` and referenced file hashes validate. Registration should be
new-key-only and should not rewrite the run. At minimum, the control tower should
index:

- analysis ID and pipeline ID/version/stage;
- animal, date, scan, session, source channel, and post-surgery day;
- source asset ID/hash and acquisition-declaration hash/status;
- reconstruction and longitudinal-comparison eligibility;
- review flags, supersession, and figure-bundle locations/hashes.

The authoritative numeric source for image panels is the exact float32 NPZ array
bundle identified by each figure JSON, not pixels re-extracted from the PNG or SVG.

## Change policy

- Patch changes that alter outputs require a new pipeline version.
- Changed configurations are independently hashed even within one code version.
- Completed run directories are immutable.
- Corrections create a new run with `supersedes`; the prior run is not edited to add a
  reverse link.
- An unresolved acquisition identity conflict is a hard failure.
- Missing legacy surgery-day metadata is explicit and blocks longitudinal comparison,
  but does not block descriptive reconstruction.
