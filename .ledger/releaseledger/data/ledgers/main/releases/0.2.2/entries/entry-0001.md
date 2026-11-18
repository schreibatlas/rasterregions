---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0001
release_version: 0.2.2
kind: changed
summary: Changed automatic segmentation to start with body anchors and use local split/merge
  repairs
status: accepted
audience: null
scopes: []
source_refs:
- git:8649a59fce4dd9af58c67e429d859e6e57dabc49
paths:
- benchmarks/README.md
- benchmarks/benchmark_workloads.py
- docs/benchmarks.md
- docs/spaced-segmentation.md
- rasterregions/_grouping.py
- rasterregions/_partition.py
- rasterregions/detect.py
- rasterregions/io.py
- rasterregions/models.py
- tests/test_anchor_first_segmentation.py
- tests/test_io.py
- tests/test_models.py
- tests/test_page_expected_count.py
- tests/test_segmentation_search_limits.py
- tests/test_spaced_region_hypotheses.py
- tests/test_spaced_regions.py
issues: []
prs: []
sources:
- git:8649a59fce4dd9af58c67e429d859e6e57dabc49
contributors:
- '@holgern'
breaking: false
internal: false
order: 1
---
