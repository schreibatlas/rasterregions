# rasterregions — implementation brief and improvement plan

## Purpose of this brief

This document is intended for a coding agent working on the reconstructed `rasterregions` repository. It describes the current architecture, validates what already works, records concrete failure cases reproduced against the supplied snapshot, and proposes a staged implementation plan that preserves the library's most valuable property: a small, deterministic, dependency-light raster segmentation core built only on NumPy and Pillow.

The supplied Codecrate pack reconstructed cleanly with machine-header and SHA verification. The current test suite passes unchanged: **11 tests passed**.

This brief does **not** recommend turning `rasterregions` into an OCR or general document-layout engine. The existing design boundary is good and should remain:

```text
pixels -> foreground mask -> components -> rows -> anonymous regions/crops
```

Semantic identity, OCR, glyph classification, handwriting-specific logic, training data schemas, and application-specific labels should stay outside this package.

---

## 1. Current repository architecture

### Public API

`rasterregions/__init__.py` exposes:

- `BBox`
- `Component`
- `DetectionResult`
- `IsolationCandidate`
- `IsolationResult`
- `MaskResult`
- `Region`
- `RowBand`
- `components`
- `crop`
- `detect_regions`
- `detect_rows`
- `foreground_mask`
- `isolate_component`
- `tighten_bbox`

The package is typed (`py.typed`) and targets Python 3.10+.

### Modules

#### `mask.py`

Responsibilities:

- convert PIL/NumPy inputs to RGB float arrays
- compute luminance and saturation/chroma
- choose `grayscale` or `color` in `auto`
- Otsu thresholding for grayscale
- fixed paper-relative threshold for `dark`
- simple confidence estimation and warnings

Current mask modes:

```python
"auto" | "grayscale" | "dark" | "color"
```

#### `components.py`

Responsibilities:

- dependency-free 8-connected component extraction using an explicit Python stack
- preserve every component's pixel coordinates (`ys`, `xs`)
- isolate the component nearest a supplied center
- explain rejected isolation candidates

#### `detect.py`

Responsibilities:

- normalize input source types
- remove page artifacts / long rules / tiny noise
- estimate body-component scale
- cluster body components into rows
- group nearby bodies into anonymous regions
- attach detached satellite marks (for example the dot over `i`)
- optionally partition a row into `expected_count=N` using projection valleys and component-crossing penalties
- construct final `Region` objects and statuses

This module contains most segmentation policy and most of the hard-coded heuristics.

#### `regions.py`

Responsibilities:

- tighten a bounding box to actual foreground pixels
- crop a PIL image without resizing

#### `io.py`

Responsibilities:

- serialize detection metadata to JSON
- write a cleaned mask
- write per-region crops
- write an overlay image with region boxes and IDs

#### `models.py`

Frozen dataclasses for geometry and result objects. Masks are copied and made read-only in result models.

#### `cli.py`

Single `detect` command exposing:

- input image
- output directory
- `--expected-count`
- `--mode`
- `--padding`
- `--min-confidence`

---

## 2. What already works well

Do not rewrite these parts without evidence. The current MVP already has several strong properties.

### 2.1 Clear scope

The library returns anonymous geometry and confidence, not semantic labels. This makes it reusable for handwriting, forms, sprites, diagrams, signatures, worksheets, and other sparse-image workflows.

### 2.2 Minimal dependency surface

Runtime dependencies are only:

```text
numpy>=1.24
Pillow>=10
```

This is worth preserving. Avoid adding OpenCV, SciPy, or scikit-image unless there is a demonstrated requirement that cannot be met reasonably with NumPy/Pillow.

### 2.3 Deterministic geometry

`BBox` uses half-open rectangles, crops preserve pixels, and ordering is deterministic. These are useful invariants for downstream tooling.

### 2.4 Useful diagnostic output

`write_detection()` already writes:

```text
regions.json
mask.png
regions.png
crops/
```

That is a good foundation for visual regression testing.

### 2.5 Detached-mark handling

Automatic grouping can attach small nearby components such as dots above a body. This is an important capability for glyph-like and handwriting-like material without introducing glyph semantics.

### 2.6 Exact-count mode is conceptually valuable

The projection-valley dynamic-programming approach gives callers a useful constrained segmentation mode when the expected number of objects is known. This should be strengthened rather than removed.

---

## 3. Validation performed against the supplied snapshot

The reconstructed repository was tested as supplied.

### Existing tests

```text
11 passed
```

### Added demonstration fixtures

The companion example bundle contains deterministic synthetic PNG fixtures and scripts. Current snapshot behavior:

| Fixture | Mode | Current result |
| --- | --- | --- |
| `01_multirow_sparse.png` | automatic | 8 regions, two rows, reading order correct |
| `02_expected_count_three.png` | `expected_count=3` | 3 regions |
| `03_color_on_grid.png` | automatic/color | 3 colored regions; neutral grid ignored |
| `04_dark_ink_gray_paper.png` | `dark` | 3 regions |
| `05_component_isolation.png` | components/isolation | 4 components; target component selected correctly |
| NumPy input example | `expected_count=3` | 3 regions, `source_path=None` as expected |

The bundle also records current edge cases that should become regression tests.

---

## 4. Correctness issues found during inspection

These are not speculative. They were reproduced with the supplied code.

## P0.1 — Otsu quantization mismatch can produce an empty grayscale mask

### Current behavior

`_otsu_threshold()` quantizes luminance through:

```python
np.clip(values, 0, 255).astype(np.uint8)
```

but `foreground_mask()` applies the returned integer threshold to the original float luminance:

```python
mask = luminance <= _otsu_threshold(luminance)
```

A pixel with RGB `(20, 20, 20)` can have float luminance approximately `20.000002`. The histogram truncates that value to bin `20`; Otsu returns threshold `20.0`; the final comparison becomes:

```text
20.000002 <= 20.0  -> False
```

A synthetic image containing only RGB 20 foreground and RGB 255 background therefore produced a **zero foreground mask** in `grayscale` mode.

### Required fix

Use the same quantized luminance values for both histogram construction and threshold application.

Recommended minimal implementation:

```python
luminance = _luminance(rgb)
luminance_u8 = np.clip(luminance, 0.0, 255.0).astype(np.uint8)
threshold = _otsu_threshold(luminance_u8)
mask = luminance_u8 <= threshold
```

Then keep the original float luminance for confidence/contrast calculations.

Prefer changing `_otsu_threshold()` to make its input contract explicit, for example `_otsu_threshold_u8(values: np.ndarray)`, or centralize quantization in a helper.

### Regression test

Add:

```python
def test_grayscale_otsu_uses_consistent_quantization() -> None:
    image = np.full((40, 40, 3), 255, dtype=np.uint8)
    image[10:30, 18:22] = 20
    result = foreground_mask(image, mode="grayscale")
    assert result.mask[10:30, 18:22].all()
    assert not result.mask[0, 0]
```

---

## P0.2 — `auto` color selection can silently discard black content

### Current behavior

`_select_method()` switches to `color` when enough saturated pixels exist. `color` mode then uses only:

```python
colored = (saturation >= 0.20) & (maximum < 250.0)
```

On a page containing black + blue + black shapes, `auto` selected `color` and detected only the blue shape. The two black shapes disappeared from the mask.

Observed fixture:

```text
mixed-ink auto -> 1 region
mixed-ink dark -> 3 regions
```

### Why this matters

A small colored annotation, stamp, correction, or mark can change the segmentation mode for the whole page and cause neutral dark foreground to vanish. This is a high-risk silent failure.

### Recommended solution

Do not make `auto` an exclusive choice between grayscale and color masks. Build candidate masks and support a hybrid foreground mask.

A dependency-light first version can be:

```python
color_mask = colored
paper = percentile(luminance, 95)
neutral_dark = (saturation < 0.20) & (luminance <= paper - neutral_dark_delta)
```

Then:

```python
hybrid = color_mask | neutral_dark
```

The neutral-dark threshold must be conservative enough to reject worksheet rules and gray grids. A starting delta of 48–64 luminance points is more appropriate than reusing the broad `dark` threshold directly.

### API choice

Preferred:

```python
MaskMethod = Literal["grayscale", "dark", "color", "hybrid"]
```

`mode="color"` should remain color-only. Only `auto` should select `hybrid` when both saturated and sufficiently dark neutral content are present.

If avoiding a new public `MaskMethod` value is mandatory, keep the internal hybrid behavior but document that `method="color"` may include dark neutral foreground. This is less clear and is not preferred.

### Regression tests

Add mixed-content tests asserting:

- black + colored objects survive `auto`
- a light gray guide line is still excluded
- pure colored input still uses color/hybrid successfully
- pure black input remains grayscale/dark as appropriate

---

## P0.3 — `expected_count=1` has zero partition confidence by construction

### Current behavior

`_partition_row()` computes confidence only for **internal boundaries**:

```python
confidences = []
for boundary in boundaries[1:-1]:
    ...
return boundaries, min(confidences, default=0.0)
```

For `expected_count=1`, there are no internal boundaries, so confidence is always `0.0`. With default `min_confidence=0.3`, a valid single-region request fails.

Observed result on a simple small black rectangle:

```text
expected_count=1 -> failed
best partition confidence is 0.000, below required 0.300
```

### Required fix

A one-segment partition with actual foreground should not receive zero confidence merely because no cut was needed.

Minimal compatibility fix:

```python
partition_confidence = min(confidences, default=1.0)
```

Better fix: combine boundary confidence and segment plausibility into a score. For `N=1`, use only segment plausibility.

### Regression test

Add a valid one-region fixture and assert:

```python
result = detect_regions(image, expected_count=1)
assert result.status == "ok"
assert len(result.regions) == 1
```

---

## P0.4 — page-artifact detection can delete legitimate large connected foreground

### Current behavior

`_is_page_artifact()` ends with:

```python
return box_area / (width * height) > 0.2
```

This classifies any component whose **bounding box** covers more than 20% of the page as an artifact, regardless of component fill ratio, border contact, or geometry.

A synthetic row of three outlined objects connected by thin bridges was removed as:

```text
Removed component 0: page border, rule, or oversized artifact.
```

Then `expected_count=3` had nothing left to partition.

A large but legitimate isolated filled rectangle can also be rejected if its bounding box crosses the same threshold.

### Required fix

Replace unconditional bounding-box coverage rejection with geometry-aware rules.

Recommended features:

```text
component_area / page_area
bbox_area / page_area
component_area / bbox_area        # fill ratio
bbox aspect ratio
border contact
horizontal/vertical span
```

Suggested policy:

- remove border-touching components only when they also span a very large fraction of width/height or page coverage
- remove very long thin horizontal/vertical rules using aspect ratio + span
- do **not** reject a non-border component solely because its bbox exceeds 20% of page area
- optionally use fill ratio to distinguish sparse connected content from filled page artifacts

Example decision sketch:

```python
bbox_fraction = box_area / page_area
area_fraction = component.area / page_area
fill_ratio = component.area / box_area

if touches_border and (component.width > 0.8 * width or component.height > 0.8 * height):
    return True
if horizontal_rule or vertical_rule:
    return True
if touches_border and bbox_fraction > 0.20 and fill_ratio < 0.15:
    return True
if area_fraction > 0.50:
    return True
return False
```

Exact constants should be test-driven rather than copied blindly from this sketch.

### Regression tests

Add cases for:

- page border removed
- long horizontal rule removed
- long vertical rule removed
- large legitimate symbol retained
- thin-connected multi-object row retained

---

## 5. Expected-count semantics should be made explicit

The current implementation does **not** interpret `expected_count=N` as a total count over the whole page. It tries to partition each detected row independently into exactly `N` pieces and returns the best qualifying row.

This behavior is visible in `_expected_count_regions()` and can be useful, but the parameter name alone does not communicate the scope clearly.

### Recommended compatibility-preserving API

Keep current behavior as the default, but make it explicit:

```python
def detect_regions(
    source,
    *,
    expected_count: int | None = None,
    expected_count_scope: Literal["row", "page"] = "row",
    ...,
) -> DetectionResult:
```

Alternatively introduce a less ambiguous new argument:

```python
expected_row_count: int | None = None
```

and deprecate ambiguous use of `expected_count` later.

### Page-wide count mode

A page-wide exact count is feasible without new dependencies.

For every row `r`, compute feasible partitions for `k = 1..N` and their costs/confidences. Then solve a small DP that allocates a total of `N` regions across rows.

State:

```text
dp[row_index][used_count] = best accumulated cost/confidence
```

Transition:

```text
choose k regions for current row
used_count + k <= expected_count
```

Finally:

- concatenate row regions in reading order
- renumber `region_id`, `column_index`, and global `reading_order`
- aggregate confidence conservatively (minimum or weighted geometric mean)

This gives callers both useful semantics:

```text
row scope  -> "this row should contain N objects"
page scope -> "this page should contain N objects total"
```

---

## 6. Improve diagnostics before adding many more heuristics

At present, many internal decisions are reduced to strings in `warnings`. That makes algorithm tuning harder than necessary.

### Proposed model additions

Add frozen dataclasses such as:

```python
@dataclass(frozen=True)
class ComponentDecision:
    component_index: int
    bbox: BBox
    area: int
    action: Literal["kept", "removed"]
    reason: str | None = None

@dataclass(frozen=True)
class PartitionBoundary:
    x: int
    projection: float
    crossing_components: int
    confidence: float

@dataclass(frozen=True)
class DetectionDiagnostics:
    image_size: tuple[int, int]
    mask_threshold: float | None
    raw_component_count: int
    cleaned_component_count: int
    component_decisions: tuple[ComponentDecision, ...]
    partition_boundaries: tuple[PartitionBoundary, ...] = ()
```

Then add:

```python
DetectionResult.diagnostics: DetectionDiagnostics | None = None
```

### JSON diagnostics

Extend `regions.json` with:

```json
{
  "rasterregions_version": "...",
  "image": {"width": 520, "height": 260},
  "parameters": {
    "mode": "auto",
    "padding": 3,
    "min_confidence": 0.3,
    "expected_count": null,
    "expected_count_scope": "row"
  },
  "mask": {
    "method": "grayscale",
    "threshold": 20,
    "confidence": 1.0,
    "foreground_fraction": 0.018
  },
  "cleanup": [...],
  "rows": [...],
  "regions": [...]
}
```

Keep old keys intact where possible so this remains a compatible additive change.

---

## 7. Centralize heuristic configuration

`detect.py` currently embeds many numeric heuristics directly in helper functions. Examples include component area ratios, median-height factors, satellite distance limits, grouping-gap factors, rule aspect-ratio thresholds, and confidence constants.

This makes the implementation difficult to tune systematically.

### Recommended approach

Introduce internal/public configuration dataclasses without making the simple API verbose.

For example:

```python
@dataclass(frozen=True)
class CleanupConfig:
    tiny_area_ratio: float = 0.008
    rule_aspect_ratio: float = 12.0
    rule_min_span_ratio: float = 0.20

@dataclass(frozen=True)
class GroupingConfig:
    body_min_area_ratio: float = 0.18
    body_min_height_ratio: float = 0.25
    row_center_gap_ratio: float = 0.72
    body_join_gap_ratio: float = 0.12
    satellite_max_area_ratio: float = 0.60
    satellite_max_horizontal_gap_ratio: float = 0.40
    satellite_max_vertical_gap_ratio: float = 1.25
```

Then:

```python
DEFAULT_CLEANUP_CONFIG = CleanupConfig()
DEFAULT_GROUPING_CONFIG = GroupingConfig()
```

The public `detect_regions()` can optionally accept a single `config=` object while preserving all current keyword arguments.

Benefits:

- heuristics become named and reviewable
- tests can target thresholds directly
- applications can tune unusual data without forking internals
- future optimization can run parameter sweeps

Do not expose every low-level knob on the CLI initially. A config object in Python is enough.

---

## 8. Masking improvements after the P0 fixes

Once correctness is fixed, improve robustness incrementally.

### 8.1 Candidate-mask scoring

Instead of only deciding method from presence of saturated pixels, build candidate masks and score them.

Candidate methods:

- grayscale Otsu
- dark relative-to-paper
- color saturation
- hybrid color + neutral-dark

Useful scoring signals:

- foreground fraction plausibility
- foreground/background luminance contrast
- median saturation of selected foreground
- number and distribution of connected components
- border occupancy
- fraction of image covered by very long thin structures

Keep scoring deterministic.

### 8.2 Optional adaptive/local thresholding

Global Otsu is vulnerable to strong illumination gradients. A local threshold can be implemented dependency-free using tiles or an integral-image mean.

Prefer a simple new mode rather than silently changing old `grayscale` behavior:

```python
mode="adaptive"
```

A practical first algorithm:

1. compute luminance
2. calculate local mean over a configurable window using integral images
3. foreground where `luminance <= local_mean - offset`
4. apply conservative global floor/ceiling

Avoid morphology by default; it changes topology and can merge objects.

### 8.3 Alpha handling tests

The current alpha compositing behavior (transparent pixels composited onto white) is sensible. Add explicit tests so it remains stable.

---

## 9. Connected-component implementation and performance

The current implementation is easy to understand but uses a Python stack and checks up to 9 neighboring pixels for every foreground pixel.

Indicative local benchmark from this snapshot:

```text
256x256 dense foreground   ~0.16 s
512x512 dense foreground   ~0.49 s
768x768 dense foreground   ~1.07 s
```

Sparse pages are much faster, so this is not a P0 problem for the MVP. Do not optimize before correctness work.

### Future optimization option

If real workloads show this is a bottleneck, replace the flood fill with a two-pass union-find / run-length connected-component labeler using only NumPy + Python.

Design goal:

- retain 8-connectivity
- retain deterministic component ordering
- retain exact `ys`/`xs` or provide an equivalent label map
- keep the public `Component` API stable

A label-map representation could also reduce repeated per-component coordinate storage in some workflows, but this is a larger internal redesign and should be benchmark-driven.

---

## 10. Region confidence should become evidence-based

Automatic region confidence is currently essentially binary heuristic output:

```text
0.92 for sufficiently separated groups
0.58 otherwise
```

Expected-count confidence is based mainly on cut valleys and crossing penalties.

This is usable as a coarse indicator but should not be interpreted as calibrated probability.

### Recommended improvement

Compute confidence from explicit normalized features and expose them in diagnostics.

Possible automatic-region features:

- left/right separation normalized by median body height
- satellite attachment ambiguity
- component-size consistency
- row assignment margin
- artifact-cleanup proximity

Possible partition features:

- projection valley depth
- number of components crossed by each boundary
- segment foreground balance
- number of primary bodies per segment
- narrow-segment penalty

The first goal is **monotonic interpretability**, not statistical calibration.

Document confidence as a heuristic score in `[0, 1]` unless/until calibrated against labeled data.

---

## 11. I/O and CLI improvements

These are P2 after correctness and diagnostics.

### JSON reproducibility

Add:

- package version
- source image dimensions
- effective padding
- mask mode requested and method selected
- mask threshold when applicable
- foreground fraction
- detection parameters
- row metadata
- structured cleanup decisions

### CLI

Potential additive flags:

```text
--json                 print detection payload to stdout
--expected-count-scope row|page
--no-crops
--no-overlay
--no-mask
```

Do not add batch-directory traversal until there is a concrete use case. It is easy for callers to loop over files in Python or shell.

---

## 12. Test strategy

### Keep existing tests

All current tests should remain green.

### Add correctness regressions first

Recommended test names:

```text
test_grayscale_otsu_uses_consistent_quantization
test_auto_preserves_black_and_colored_foreground
test_expected_count_one_can_succeed
test_large_legitimate_component_is_not_page_artifact
test_page_border_is_removed
test_long_rule_is_removed
test_touching_connected_content_survives_cleanup
```

### Add example fixtures as visual regression inputs

The companion bundle contains deterministic assets. Copy selected fixtures under `tests/fixtures/` if binary test data is acceptable, or recreate them in test helper functions with Pillow.

Prefer generating small test images in code when possible; reserve committed PNGs for cases where visual inspection itself is useful.

### Add invariant/property-style tests without a new dependency

Examples:

- every region bbox is inside image bounds
- `region.image.size == bbox size`
- `region.mask.shape == bbox size`
- regions are returned in monotonic reading order
- padding never expands outside image bounds
- `clean_mask` shape equals raw mask shape
- identical input produces identical payload
- NumPy float `[0,1]` and equivalent uint8 input produce equivalent geometry

### Golden outputs

For a small stable set, compare normalized JSON payloads rather than PNG bytes. PNG encoder metadata can complicate byte-for-byte image comparisons.

---

## 13. Proposed implementation sequence

### Phase 1 — regression tests for reproduced failures

No production changes yet.

Add tests for:

1. Otsu float/uint8 mismatch
2. mixed black + color auto mask
3. `expected_count=1`
4. oversized legitimate connected component

Acceptance: tests fail on current snapshot for the intended reason.

### Phase 2 — mask correctness

Files:

```text
rasterregions/mask.py
tests/test_mask.py
```

Implement:

- consistent Otsu quantization
- hybrid auto masking or another tested strategy that preserves black + colored content
- optional threshold metadata internally if diagnostics work is done concurrently

Acceptance:

- all old tests pass
- new mask regressions pass
- `03_color_on_grid.png` still ignores neutral grid lines
- `07_mixed_black_color.png` detects all three objects in `auto`

### Phase 3 — artifact cleanup correctness

Files:

```text
rasterregions/detect.py
tests/test_detect.py
```

Implement geometry-aware `_is_page_artifact()`.

Acceptance:

- page border and long-rule tests still pass
- connected/touching fixture is not deleted before partitioning
- large legitimate foreground survives cleanup

### Phase 4 — exact-count fixes

Files:

```text
rasterregions/detect.py
tests/test_detect.py
README.md
```

Implement:

- nonzero meaningful confidence for `expected_count=1`
- explicitly document current row-scoped semantics
- optionally add page-wide scope via DP

Acceptance:

- one-region case succeeds at default confidence
- existing `expected_count=3` test still passes
- wrong expected count still fails cleanly

### Phase 5 — structured diagnostics and config

Files likely touched:

```text
rasterregions/models.py
rasterregions/mask.py
rasterregions/detect.py
rasterregions/io.py
rasterregions/__init__.py
tests/test_io.py
tests/test_public_api.py
README.md
```

Keep additions backward-compatible.

### Phase 6 — examples and docs integration

Move or adapt the supplied example bundle into repository `examples/`.

Recommended structure:

```text
examples/
  README.md
  generate_assets.py
  example_01_auto_regions.py
  example_02_expected_count.py
  example_03_color_mask.py
  example_04_dark_paper.py
  example_05_component_isolation.py
  example_06_numpy_input.py
  example_07_current_edge_cases.py
  assets/
```

Do not ship generated `outputs/` in the Python wheel unless intentionally desired; they are better for repository documentation or CI artifacts.

### Phase 7 — performance only if needed

Benchmark real target scans. Optimize connected components only if it is significant relative to mask creation, I/O, and downstream processing.

---

## 14. Recommended first patch set

A coding agent should start with a narrow correctness PR rather than a broad refactor.

### Patch A

**Goal:** fix the four reproduced correctness failures without redesigning the API.

Changes:

1. make Otsu quantization consistent
2. make `auto` preserve saturated + dark-neutral content
3. set sensible confidence for one-part partitions
4. make page-artifact filtering geometry-aware
5. add focused regression tests
6. update README limitations/behavior

Avoid in Patch A:

- new third-party dependencies
- complete confidence redesign
- component algorithm rewrite
- new CLI command hierarchy
- OCR/layout semantics

This yields the highest correctness gain with the lowest compatibility risk.

---

## 15. Concrete acceptance matrix for Patch A

Run the repository tests and the companion examples.

Expected after the patch:

| Case | Current | Target |
| --- | --- | --- |
| existing test suite | 11/11 pass | all pass |
| multi-row sparse | 8 regions | still 8 |
| expected count 3 | 3 regions | still 3 |
| color on gray grid | 3 colored regions | still 3, grid excluded |
| dark gray RGB 20 ink in grayscale | empty mask | ink retained |
| black + blue + black in auto | 1 region | 3 regions |
| expected count 1 | failed at default confidence | succeeds |
| thin-connected 3 objects | deleted as artifact | survives cleanup; partition can attempt/succeed |

Do not declare the patch complete if it fixes a new edge case by regressing the existing color-grid behavior.

---

## 16. Compatibility constraints

Maintain these behaviors unless explicitly versioned:

- `detect_regions()` accepts path, PIL image, grayscale NumPy, RGB NumPy, RGBA NumPy
- NumPy float arrays in `[0,1]` remain supported
- region IDs remain deterministic (`region-000`, etc.)
- bboxes remain half-open
- `Region.image` and `Region.mask` match bbox dimensions exactly
- existing public imports remain valid
- `write_detection()` continues producing current core files
- CLI exit code remains nonzero for failed detection

Additive model fields should have defaults where possible.

---

## 17. Scope boundaries to preserve

Do not add the following to `rasterregions` unless the project explicitly changes scope:

- OCR
- glyph names
- handwriting identity
- language-specific text direction inference
- semantic form-field labels
- ML model inference
- training pipelines
- full document-layout hierarchy

A future extension for rotated/skewed sparse pages can still fit the package, but it should remain geometric/pixel-level and optional.

---

## 18. Deliverables accompanying this brief

The example archive includes:

```text
README.md
generate_assets.py
run_all_examples.py
example_01_auto_regions.py
example_02_expected_count.py
example_03_color_mask.py
example_04_dark_paper.py
example_05_component_isolation.py
example_06_numpy_input.py
example_07_current_edge_cases.py
assets/*.png
outputs/*
run_log.txt
```

`outputs/gallery.png` provides a compact visual check of the main successful capabilities. The edge-case script prints reproducible current failures and should be treated as a starting point for regression tests.

---

## Final recommendation

Keep the overall architecture. It is appropriately small and the public boundary is sound. The next release should focus on **segmentation correctness and explainability**, not feature breadth.

The highest-value work is:

1. fix mask quantization correctness
2. make automatic masking safe for mixed neutral + colored foreground
3. correct one-region exact-count confidence
4. stop removing legitimate large connected content as a generic page artifact
5. expose structured diagnostics and named heuristic configuration
6. only then expand exact-count semantics and performance

That sequence improves reliability while retaining the dependency-light design that makes `rasterregions` useful as a reusable primitive.
