# ZStackAnalysis working rules

- The highest-priority requirement for every analysis run is explicit identity
  and provenance. Record the animal ID, acquisition date, scan ID, session ID,
  authoritative source path, source channel when applicable, and source file
  identity. For multi-input analyses, record every input and its ordered role.
- Explicitly identify the analysis pipeline for every run: pipeline/workflow
  name, analysis state, method statement, code version or code hashes, complete
  resolved configuration, and analysis timestamp. Never describe results only
  as a generic "Z-stack analysis."
- Carry the data identity and pipeline identity into run directories,
  manifests/config snapshots, QC records, figure titles or metadata, and
  comparison tables so that every result can be traced without relying on
  memory or surrounding folder context.
- Do not run or compare data when the animal, date, scan/session, source role,
  channel, or pipeline identity is missing or ambiguous. Resolve the ambiguity
  first. Discover actual uploaded files and read their metadata rather than
  inferring identity from a schedule or filename alone.
- Before comparing runs, verify both the biological/acquisition grouping and
  the pipeline/configuration. Treat outputs from different developing pipeline
  versions as different methods unless equivalence has been explicitly tested.
  State all such differences in the comparison.
- Keep development runs, figures, and experimental methods in this project.
- Do not write development artifacts back into LabGraph.
- LabGraph receives only a deliberately selected, tested final integration.
- The default baseline is per-plane pixelwise median over all usable frames.
- Frame-to-median correlations are descriptive QC; do not silently reject frames.
- Produce mean projections as the primary view and maximum projections only as a
  clearly labeled comparison.
- Never call this workflow axial-motion or 3-D motion correction.
- Preserve ScanImage Z coordinates and physical voxel spacing in outputs.
