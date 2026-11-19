# Continuous foreground evidence

`foreground_mask` remains the ordinary segmentation API. It returns a `MaskResult` with a boolean foreground mask and the existing extraction metadata. Call `foreground_evidence` when a downstream geometric algorithm also needs the continuous evidence that produced that classification:

```python
from rasterregions import foreground_evidence

evidence = foreground_evidence("page.png", mode="auto")
mask = evidence.mask
field = evidence.decision_field
```

`ForegroundEvidence` is an immutable snapshot. Its mask is a read-only Boolean array, and its decision field is a read-only `float32` array with the same two-dimensional shape. `foreground_mask` does not allocate a decision field.

## Decision-field contract

The field is a signed implicit representation of the same classification, not a probability or coverage map:

- Positive values mean foreground.
- Negative values mean background.
- Zero is the classification boundary and belongs to foreground, so `evidence.mask` equals `evidence.decision_field >= 0`.
- The field is finite and sampled at the pixel centers.
- Magnitude is evidence strength. It is neither calibrated probability nor Euclidean distance.

For grayscale and dark modes, the field is based on the original floating-point luminance and the selected scalar threshold. Adaptive mode uses its actual per-pixel threshold map. Color mode takes the minimum of normalized saturation and value margins. Hybrid mode takes the maximum of color and neutral-dark evidence, matching its union classification. Auto mode returns the field for the method selected by the existing scoring and tie-breaking rules.

The deterministic color margin scales are derived from `MaskConfig`: saturation uses `max(saturation_threshold, 1 - saturation_threshold)`, and channel value uses `max(color_max_value, 255 - color_max_value)`. Intensity margins use the relevant configured transition scale, with a minimum scale of one intensity level. These choices normalize magnitudes without changing the zero crossing. Values whose sign conflicts with the historical mask decision are moved to the nearest representable `float32` value on the matching side of zero.

## Pixel coordinates

For array index `[row, column]`, the continuous coordinates are `x = column` and `y = row`. An integer coordinate identifies the center of that pixel. RasterRegions adds no half-pixel offset. A zero crossing between neighboring pixel centers can therefore be localized at a fractional coordinate, such as `x = 23.87`, by bilinearly interpolating the field.

## Visualize a fractional crossing

This example writes a grayscale visualization of the signed field and marks an interpolated crossing on a horizontal scanline. It needs only the package's existing Pillow dependency:

```python
import numpy as np
from PIL import Image, ImageDraw

from rasterregions import foreground_evidence

image = np.full((32, 32), 255, dtype=np.uint8)
image[8:24, 8:24] = 0
evidence = foreground_evidence(image, mode="dark")

row = 16
field = evidence.decision_field
crossings = np.flatnonzero((field[row, :-1] >= 0) & (field[row, 1:] < 0))
left = int(crossings[-1])  # right edge of the dark rectangle
v0, v1 = field[row, left], field[row, left + 1]
x_boundary = left + float(v0 / (v0 - v1))
print(f"Interpolated crossing: x={x_boundary:.2f}, y={row}")

scale = 8
display = np.clip(127.5 + field * (127.5 / 8.0), 0, 255).astype(np.uint8)
preview = Image.fromarray(display).convert("RGB").resize(
    (display.shape[1] * scale, display.shape[0] * scale), Image.Resampling.NEAREST
)
marker_x = round((x_boundary + 0.5) * scale)
ImageDraw.Draw(preview).line(
    (marker_x, 0, marker_x, preview.height - 1), fill=(255, 0, 0), width=1
)
preview.save("foreground-evidence.png")
```

The visualization clips only its display values. The returned decision field itself is not blurred or clipped near zero, so interpolation continues to use the segmentation decision.

## Release note

> RasterRegions can now expose the signed continuous decision field underlying foreground segmentation. Existing boolean-mask APIs are unchanged. The new field is intended for consumers that need subpixel boundary localization.
