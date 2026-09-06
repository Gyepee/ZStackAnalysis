# Acquisition-folder JSON contract

Use two separate JSON files when both concepts apply:

- `external_copy_status.json`: transfer/copy provenance—source, destination, copy
  completion, byte count, and source/destination hashes.
- `zstack_acquisition.json`: scientific acquisition identity and purpose—Z-stack,
  Bench2p, animal/date/scan/session, TIFF, surgery timing, and operator/QC notes.

Do not rename a copy-status record to imply that it describes the acquisition. A
completed copy can still be the wrong scan, wrong animal, or a non-Z-stack recording.
Conversely, a correct acquisition declaration does not prove that an external copy is
complete or byte-identical.

For Bench2p Z-stacks, the identifying pair is:

```json
{
  "data_kind": "zstack",
  "acquisition_system": "bench2p"
}
```

The canonical acquisition filename remains `zstack_acquisition.json` so the same
pipeline can later support another acquisition system without changing discovery
rules. If a folder contains multiple independent TIFF acquisitions, use one folder per
acquisition or pass an explicit per-TIFF declaration with `--acquisition-spec`.

Version 0.1.0 automatically discovers and hashes `zstack_acquisition.json`. It treats
`external_copy_status.json` as a separate upstream transfer record; its schema and
admission gate should be adopted from ResearchDataGovernance/LabGraph once that
contract is versioned rather than guessed locally.
