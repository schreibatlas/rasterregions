---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 2
entry_id: entry-0001
release_version: 0.1.0
kind: added
summary: Added deterministic foreground masking and anonymous region detection
status: accepted
audience: null
scopes: []
source_refs:
- tl:task-0003
paths:
- rasterregions/mask.py
- rasterregions/detect.py
- rasterregions/cli.py
issues: []
prs: []
sources: []
contributors: []
breaking: false
internal: false
order: 1
---
The Python API and CLI provide boolean masks, 8-connected components, row-major regions and crops, optional expected-count partitioning, confidence, and JSON/image diagnostics.
