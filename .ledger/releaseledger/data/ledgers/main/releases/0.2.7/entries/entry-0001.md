---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0001
release_version: 0.2.7
kind: fixed
summary: Made cleanup and detached-mark ownership scale-aware, with explicit search-limit
  diagnostics
status: accepted
audience: null
scopes: []
source_refs: []
paths:
- rasterregions/_cleanup.py
- rasterregions/_grouping.py
- rasterregions/detect.py
- rasterregions/models.py
- rasterregions/io.py
- docs/spaced-segmentation.md
- docs/diagnostics.md
issues: []
prs: []
sources: []
contributors: []
breaking: false
internal: false
order: 1
---
Hard micro-noise is removed using configurable body-relative area, height, and width limits without proximity-only rescue. Compact dots and elongated strokes use target-local geometry. Clear ownership no longer branches to unassigned, and branch-budget truncation is distinguished from alternative clipping.
