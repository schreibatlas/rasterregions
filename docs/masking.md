# Foreground masking

`foreground_mask(image, mode="auto", *, config=None, adaptive_window_size=None, adaptive_offset=None)` returns a `MaskResult` containing a two-dimensional boolean mask, confidence, selected method, warnings, and an optional threshold. `True` denotes foreground. The result owns a read-only mask snapshot.

## Modes

- `auto` deterministically scores grayscale, dark-relative, color, and hybrid candidates, then selects the highest-scoring candidate.
- `grayscale` quantizes luminance and selects a global threshold with Otsu's method.
- `dark` selects pixels darker than a threshold relative to the estimated paper level.
- `color` selects colored pixels only. It does not include neutral dark marks.
- `hybrid` combines colored pixels with neutral dark pixels.
- `adaptive` compares luminance with a local mean and is intended for uneven illumination.

The selected `method` can differ from the requested `mode` when `mode="auto"`. `MaskResult.confidence` is a deterministic heuristic score from zero to one, not a calibrated probability. Warnings describe empty masks, excessive foreground, or low separation confidence. Mask confidence does not determine `DetectionResult.status`.

## Configuration

Use `MaskConfig` to configure threshold and confidence behavior. For adaptive thresholding, `adaptive_window_size` must be an odd integer of at least three. The direct keyword arguments override the corresponding values in `config`:

```python
from rasterregions import DetectionConfig, MaskConfig, detect_regions

config = DetectionConfig(mask=MaskConfig(adaptive_window_size=21, adaptive_offset=8.0))
result = detect_regions("shaded-page.png", mode="adaptive", config=config)
```

`MaskConfig` is also accepted directly by `foreground_mask`:

```python
from rasterregions import MaskConfig, foreground_mask

mask = foreground_mask("page.png", config=MaskConfig())
```

For supported image values, EXIF handling, and mask ownership, see [Concepts](guide.md). For the image-to-region pipeline, see [Detection](detection.md).
