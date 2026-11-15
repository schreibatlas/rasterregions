---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 3
entry_id: entry-0003
release_version: 0.1.0
kind: fixed
summary: Fixed bbox tightening and optional padding within supplied bounds
status: accepted
audience: null
scopes: []
source_refs:
- tl:task-0003
paths:
- rasterregions/regions.py
issues: []
prs: []
sources: []
contributors: []
breaking: false
internal: false
order: 3
---
Empty-mask tightening preserves the supplied box, and nonempty results cannot expand beyond it.
