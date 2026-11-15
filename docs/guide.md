# Concepts and input contracts

## Processing boundary

The library's domain-neutral pipeline is:

```text
image -> foreground mask -> components -> rows -> anonymous regions and crops
```

Applications own OCR, glyph identity, document semantics, and domain labels. A `Region` describes geometry and contains an image crop and mask; it does not claim what the foreground depicts.

## Input images

`detect_regions` and `foreground_mask` accept a filesystem path, a Pillow image, or a NumPy array. Arrays may be two-dimensional grayscale or have RGB/RGBA channels. Values must be finite numeric values. Floating-point arrays in `[0, 1]` are scaled to `[0, 255]`; values are clipped to that range during conversion. RGBA inputs are composited over white.

Paths and Pillow images are transposed according to EXIF orientation before processing. Reported boxes, masks, overlays, and crops use the orientation-corrected coordinates. NumPy arrays have no EXIF metadata and retain their supplied orientation.

## Coordinates and arrays

`BBox(x0, y0, x1, y1)` uses half-open pixel coordinates: the left/top edges are included, while the right/bottom edges are excluded. A box has nonnegative origin and positive width and height. Region rows, columns, and reading-order indexes are zero-based. Rows are ordered top to bottom and regions within each row left to right. Right-to-left ordering is not inferred.

Masks use boolean values with `True` for foreground. `MaskResult.mask`, `DetectionResult.clean_mask`, component coordinates, region masks, and isolation masks are copied read-only NumPy snapshots. Result dataclasses are frozen. Pillow images in results are independent mutable copies, so editing a crop does not modify the source or another result.

## Configuration and confidence

`DetectionConfig` groups `CleanupConfig`, `GroupingConfig`, `MaskConfig`, and `PartitionConfig`. Configuration dataclasses validate their values and reject non-finite or out-of-domain settings rather than silently clamping them. See [Masking](masking.md) and [Detection](detection.md) for mode and parameter behavior.

Confidence values are deterministic heuristic scores in `[0, 1]`, not calibrated probabilities. Mask confidence is separate from region and partition confidence and does not gate detection status.

## Current limits

This release is an MVP for sparse images with separated foreground objects. Touching objects, dense prose, rotated layouts, nested regions, and arbitrary document layout require stronger algorithms. Use an expected count only when the caller has external evidence for it.


## Benchmarks

The opt-in benchmark suite provides deterministic dense/sparse component masks and partition rows of 2,000 or 4,000 pixels. Run it from the repository root with `python -m benchmarks.benchmark_workloads --repeat 5 --warmup 1`; see [Opt-in benchmarks](benchmarks.md) for workload definitions. Timings depend on the machine and are not CI assertions. Compare before/after runs under controlled conditions and include representative application workloads.
