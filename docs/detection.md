# Region detection

`detect_regions(source, *, expected_count=None, expected_count_scope="row", mode="auto", padding=None, min_confidence=0.3, config=None, roi=None)` returns a `DetectionResult` with the normalized input image, raw and cleaned masks, component and row records, ordered regions, warnings, status, and diagnostics.

## Automatic grouping

Without `expected_count`, the library cleans the foreground mask, groups 8-connected components into rows and bodies, attaches plausible detached satellites, and returns anonymous regions in reading order. Within each row, regions are ordered left to right. Rows are ordered top to bottom. The library does not infer right-to-left order or semantic labels.

## Expected count

When the caller has external evidence for a count, `expected_count=N` requests a projection-based partition using low-ink valleys and component-crossing costs:

```python
result = detect_regions("written-row.png", expected_count=3)
```

The default `expected_count_scope="row"` partitions the best qualifying row. Use `expected_count_scope="page"` to constrain the total across detected rows. A credible partition returns the requested number of regions. If no credible partition is found, or its confidence is below `min_confidence`, status is `failed` and `regions` is empty.

## Regions of interest

Pass a half-open `BBox` to constrain analysis. The result still uses full-image coordinates, and the raw and cleaned masks have the full image dimensions. Pixels outside the ROI are false. Diagnostics and `regions.json` record the ROI.

```python
from rasterregions import BBox, detect_regions

result = detect_regions(
    "worksheet.png",
    roi=BBox(24, 16, 496, 238),
    expected_count=4,
)
```

An ROI must have positive width and height and fit inside the input image. ROI coordinates use the normalized, orientation-corrected image coordinate system described in [Concepts](guide.md).

## Status and confidence

The status is one of `ok`, `ambiguous`, or `failed`:

- Automatic detection is `failed` when it returns no regions. It is `ambiguous` when any returned region has confidence below `min_confidence`; those regions remain available for review.
- Expected-count detection is `failed` when a credible partition cannot be produced. It returns no regions in this case.
- Mask confidence is a separate diagnostic and warning. It does not gate the detection status.

Mask, region, and partition confidence values are deterministic heuristic scores in `[0, 1]`. They are not calibrated probabilities and should not be compared as though they measure the same evidence.

`padding` is a nonnegative number of pixels added around each crop, clipped to image bounds. When omitted, detection chooses a small default based on the median component height. `DetectionConfig` groups cleanup, grouping, mask, and partition parameters.

For connected components and bbox utilities, see [Components and isolation](components.md). For status manifests and output files, see [Diagnostics](diagnostics.md).
