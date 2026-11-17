# Spaced-region segmentation

`detect_spaced_regions` is an opt-in segmentation strategy for sparse foreground objects whose spacing is useful evidence. It reuses the normal image normalization, mask extraction, cleanup, connected components, crops, and immutable result models. It does not change the defaults of `detect_regions`.

```python
from rasterregions import detect_spaced_regions

result = detect_spaced_regions("page.png")
print(result.status, len(result.regions))
```

Use `row_counts` when a caller has external evidence about counts for individual detected rows. Each entry is either a positive integer for an exact-count partition or `None` to infer that row's count:

```python
result = detect_spaced_regions("page.png", row_counts=[None, 4])
```

The sequence must have one entry per detected row. Automatic inference is deterministic and bounded by `SpacingConfig.max_inferred_count`. Similar-cost alternatives, strong internal valleys in a single connected body, weak boundaries, suspicious width deviations, ambiguous marks, or unassigned ink can make the status `ambiguous`. A row that cannot be partitioned returns diagnostics rather than fabricated cuts.

## Ownership and diagnostics

The API uses body and secondary-mark evidence, smoothed projection valleys, component-center midpoints, and the existing dynamic-programming partitioner. Width priors are estimated from the current row; scale-relative controls live in `DetectionConfig.spacing`:

```python
from rasterregions import DetectionConfig, SpacingConfig, detect_spaced_regions

config = DetectionConfig(
    spacing=SpacingConfig(
        padding_ratio=0.08,
        min_region_width_ratio=0.15,
        max_region_width_ratio=3.5,
    )
)
result = detect_spaced_regions("page.png", config=config)
```

Every `Region.mask` contains only pixels owned by that region. Region masks never overlap. `DetectionResult.unassigned_mask` retains cleaned foreground that was not assigned, including marks with ambiguous ownership, and the union of region masks and the unassigned mask accounts for the cleaned foreground. Rectangular boxes may overlap; use each region's mask when exact ownership matters.

`DetectionResult.diagnostics` includes row envelopes and component IDs, selected and unselected candidate boundaries, valley strength, component crossings, width-prior deviation, body counts, secondary-mark decisions and margins, the best-versus-second-best partition margin, and unassigned-ink fraction. `detection_payload(result)` serializes these details as JSON.

## Distinguish the stages

- **Connected components** are 8-connected foreground pixel sets.
- **Grouped regions** from `detect_regions` attach plausible components using the legacy grouping strategy.
- **Spaced-region segmentation** from `detect_spaced_regions` partitions each row using spacing, projection, body-count, width, and ownership evidence.
- **Expected-count partitioning** through `detect_regions(expected_count=...)` remains a separate legacy behavior. For spaced segmentation, `expected_count` is a page total and `row_counts` optionally provides per-row evidence.

The synthetic example writes a cleaned mask, row and candidate-cut overlays, the selected partition, up to two alternative partitions, unassigned ink, and a JSON hypothesis payload under `build/examples/08_spaced_regions/`:
```bash
python examples/08_spaced_regions.py
```


## Bounded row and page hypotheses

`detect_spaced_regions` accepts `expected_count` as generic page-level count evidence. It generates bounded row-count and partition candidates, then combines them with dynamic programming. A page allocation is returned only when its row counts sum exactly to the requested total. If none is feasible, the result is failed with an explicit warning and all cleaned ink remains unassigned. Existing `DetectionResult` fields and `detect_regions` behavior remain unchanged.

Use `detect_spaced_region_hypotheses` when downstream code needs alternatives and stable normalization frames:

```python
from rasterregions import detect_spaced_region_hypotheses

hypotheses = detect_spaced_region_hypotheses(
    "page.png",
    expected_count=5,
    row_counts=[None, None],
)
selected = hypotheses.selected
best_page = hypotheses.page_hypotheses[0] if hypotheses.page_hypotheses else None
print(hypotheses.truncated, hypotheses.failure_reason)
```

When page hypotheses exist, hypothesis zero matches `selected`. Row and page hypotheses expose costs, boundary evidence, cut-crossing costs, body counts, region widths, frames, secondary-mark assignments, and unassigned-ink fractions. A detached mark can appear assigned to different regions in separate page hypotheses, including an explicit unassigned option. Every page hypothesis owns each cleaned foreground pixel exactly once, either through one region mask or its unassigned mask.

`SpacingConfig` bounds candidate boundaries, row counts, row and page search states, secondary-mark branches, and returned hypotheses. `truncated=True` means at least one configured bound removed alternatives. The selected result and search diagnostics remain available when a bounded search finds no feasible page allocation.

Serialize the aggregate result with `rasterregions.io.spaced_hypotheses_payload`. Masks are JSON-compatible row runs, and region runs are relative to their bounding boxes. `write_spaced_hypotheses` writes the payload plus cleaned-mask, row, candidate-cut, selected-partition, alternative-partition, and unassigned-ink images.

The CLI exposes the same path:

```bash
rasterregions spaced page.png --expected-count 5 --row-counts auto,auto --output build/page-segmentation
```

`--row-counts` accepts comma-separated positive counts or `auto`, with one entry required per detected row. The CLI reports the number of returned page hypotheses and whether search was truncated; `hypotheses.json` includes the limiting budget and detailed alternatives.
