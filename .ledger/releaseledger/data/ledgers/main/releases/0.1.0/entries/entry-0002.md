---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 2
entry_id: entry-0002
release_version: 0.1.0
kind: added
summary: Added ROI-constrained detection and typed isolation decisions
status: accepted
audience: null
scopes: []
source_refs:
- tl:task-0003
paths:
- rasterregions/detect.py
- rasterregions/models.py
- rasterregions/components.py
issues: []
prs: []
sources: []
contributors: []
breaking: false
internal: false
order: 2
---
Detection can analyze a half-open region of interest while returning full-image coordinates. Isolation results expose a public reason type for selected and rejected components.
