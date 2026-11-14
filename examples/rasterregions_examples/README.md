# rasterregions example bundle

These examples exercise the rasterregions API with reproducible synthetic fixtures.

## Setup

From the reconstructed repository:

```bash
python -m pip install -e .
```

Or run without installation by setting `PYTHONPATH` to the repository root.

## Run everything

```bash
python examples/rasterregions_examples/run_all_examples.py
```

Each example writes diagnostics under `outputs/`. The bundle also contains the exact PNG inputs under `assets/`.

## Examples

- `example_01_auto_regions.py` — automatic multi-row region detection, reading order, page-border/rule/noise cleanup, and diagnostics.
- `example_02_expected_count.py` — exact `expected_count=3` partitioning.
- `example_03_color_mask.py` — automatic colored-ink masking while neutral gray grid lines are ignored.
- `example_04_dark_paper.py` — dark ink on shaded gray paper.
- `example_05_component_isolation.py` — 8-connected components and anchor-based isolation.
- `example_06_numpy_input.py` — direct NumPy input.
- `example_07_current_edge_cases.py` — regression assertions for mixed ink, Otsu quantization, connected cleanup, and one-region exact count.

Running the scripts refreshes outputs to match current behavior. They are repository-side visual artifacts and are not included in the Python wheel.
