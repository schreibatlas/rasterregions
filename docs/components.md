# Components, isolation, and crops

## Connected components

`components(mask)` labels `True` pixels as 8-connected components. Components are returned in row-major discovery order and have zero-based indexes, a half-open `BBox`, area, and read-only `ys` and `xs` arrays of foreground pixel coordinates. The function does not run detection cleanup. Supply a two-dimensional mask; nonzero values are interpreted as foreground.

## Anchor-based isolation

`isolate_component(mask, center=(x, y), competing_centers=())` selects the component with a foreground pixel nearest to the anchor and returns a mask containing only that component. Distances are Euclidean in pixel coordinates. The result includes a record for every component:

- `nearest-to-center`: the selected component.
- `nearest-to-competing-center`: a rejected component closer to one of the supplied competing centers than to the reviewed center.
- `not-nearest-component`: a rejected component not selected by either rule.

`IsolationReason` is the public literal type for these values. The selection is a proposal; an application decides whether to accept it. The returned mask is a read-only snapshot. With no foreground components, the result mask is empty and the candidate list is empty.

## Tighten a bbox and crop

`BBox(x0, y0, x1, y1)` uses half-open coordinates. `tighten_bbox(mask, bbox, pad=0)` finds foreground pixels inside the supplied bbox and tightens around them. Optional padding is clipped to the original bbox, so the result never expands beyond the supplied bounds. If the bbox has no foreground pixels, it is returned unchanged.

`crop(image, bbox)` returns an independent Pillow image crop without resizing. Both helpers reject boxes that extend beyond their input dimensions.

```python
from rasterregions import BBox, crop, isolate_component, tighten_bbox

bbox = tighten_bbox(mask, BBox(10, 10, 80, 70), pad=3)
proposal = isolate_component(mask, center=(42.0, 35.0))
region_image = crop(image, bbox)
```

See [Concepts](guide.md) for coordinate conventions and [Diagnostics](diagnostics.md) for component-index namespaces in serialized output.
