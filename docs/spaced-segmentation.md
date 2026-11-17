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
- **Expected-count partitioning** through `detect_regions(expected_count=...)` remains a separate legacy behavior for callers that already know a count. The new API accepts optional per-row count hints without changing that contract.

The synthetic example writes an input mask, row and component overlays, a candidate-boundary overlay, a final region overlay, and JSON diagnostics under `build/examples/08_spaced_regions/`:

```bash
python examples/08_spaced_regions.py
```
