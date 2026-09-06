# Recommendations for LabGraph as the control tower

This is a handoff for the agent that will reorganize LabGraph. It proposes structure
and interfaces; it does not modify LabGraph itself.

## 1. Make one short constitution canonical

Create a concise top-level control document that answers only: authority boundaries,
canonical data discovery route, immutable run rule, stable identifiers, external/shared
write rules, and which contracts every department must implement. Give the contract a
machine-readable semantic version such as `labgraph.governance.v1`. Longer rationale
belongs in linked guides.

## 2. Separate the document layers

Use a small, non-overlapping information architecture:

1. `GOVERNANCE.md`: authority, ownership, immutability, authorization.
2. `DATA_ACCESS.md`: browsing/discovery APIs, raw vs aligned data, read/write boundary.
3. `ANALYSIS_RUN_CONTRACT.md`: required manifest schema and directory structure.
4. `FIGURE_CONTRACT.md`: figure bundle and style requirements.
5. `PROJECT_INTEGRATION.md`: how a department such as ZStackAnalysis registers outputs.
6. `OPERATIONS.md`: ingest/population/retry procedures and failure handling.

Keep task-specific scientific conventions out of the universal contracts. Link them
from project profiles instead.

## 3. Publish schemas, not only prose

Add versioned JSON Schemas for source/acquisition manifests, aligned datasets,
analysis manifests, review flags, and figure manifests. Supply one minimal valid and
one fully populated example for each. A single `labgraph validate <path>` command
should check schema, hashes, stable IDs, allowed states, figure bundles, and
supersession links without writing shared state.

## 4. Define a department registration interface

Independent repositories should never need to import LabGraph source code to generate
valid outputs. Publish a stable handoff envelope containing:

- department/project and pipeline ID/version;
- immutable analysis ID;
- input asset/dataset IDs and manifest hashes;
- animal/date/scan/session identities;
- eligibility and review state;
- output/figure manifest locations and hashes;
- `supersedes` links.

LabGraph can validate and index this envelope through an explicit new-key-only command.
Validation and registration should be separate operations; registration is a shared
write and must require clear authorization.

## 5. Add a policy/profile registry

Maintain named, versioned profiles for common departments (Z-stack, behavior,
mini2p imaging, aligned neural analysis). Each profile should declare the lawful input
boundary, mandatory metadata, biological/repeated/technical units, relevant QC gates,
and permitted output types. This avoids copying one broad “clean cohort” rule between
scientifically different analyses.

## 6. Make provenance status observable

Provide a read-only dashboard or CLI report that distinguishes:

- discovered source;
- ingested source;
- downstream-populated data;
- aligned dataset;
- analysis-eligible data;
- completed/current/superseded analysis;
- registered figure bundle.

These states should not be collapsed into a single “done” flag. Every row should show
the responsible manifest and reason for `hold`, `needs_review`, or failure.

## 7. Preserve source-side declarations and review events

Allow acquisition folders to carry small source-owned JSON declarations, like
ZStackAnalysis's `zstack_acquisition.json`. LabGraph should ingest or index them by
hash, not rewrite them. Operator observations such as monitor-light contamination
should be append-only review events that affect eligibility in new analyses without
changing old run contents.

## 8. Add cross-repository conformance tests

Keep a tiny fixture repository or contract test kit that every department can run in
CI. It should verify manifest portability, no absolute repository-owned paths, exact
hashes, stable identifiers in tables, editable SVG plus 300 dpi PNG, figure JSON/MD,
declared sample units, and explicit supported/unsupported claims. Contract changes
must be versioned and tested against at least one older fixture.

## Suggested first implementation slice

The highest-leverage first slice is: freeze `labgraph.governance.v1`, publish the
analysis/figure JSON Schemas, and implement a read-only validator. Then define the
new-key-only department registration command. This gives all independent projects a
stable target before the broader documentation is reorganized.
