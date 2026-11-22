---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0001
release_version: 0.2.6
kind: changed
summary: Improved spaced segmentation with scale-aware cleanup, context-aware mark
  assignment, and evidence-based splits
status: accepted
audience: null
scopes: []
source_refs:
- git:5c3f657c14e74f4c4ef9daf8f670ad15ac26c65b
paths:
- docs/diagnostics.md
- docs/spaced-segmentation.md
- rasterregions/__init__.py
- rasterregions/_cleanup.py
- rasterregions/_grouping.py
- rasterregions/_partition.py
- rasterregions/_rows.py
- rasterregions/detect.py
- rasterregions/io.py
- rasterregions/models.py
- tests/test_anchor_first_segmentation.py
- tests/test_anchor_split_evidence.py
- tests/test_context_aware_cleanup.py
- tests/test_io.py
- tests/test_micro_noise_cleanup.py
- tests/test_segmentation_search_limits.py
- tests/test_spaced_regions.py
issues: []
prs: []
sources:
- git:5c3f657c14e74f4c4ef9daf8f670ad15ac26c65b
contributors:
- '@holgern'
breaking: false
internal: false
order: 1
---
