# rasterregions example bundle

These examples exercise the Python API with synthetic fixtures that can be regenerated from source.

## Setup and run

Install the package from the repository root, then run:

```bash
python -m pip install -e .
python examples/rasterregions_examples/run_all_examples.py
```

The runner regenerates all eight input assets before executing the scripts. Example outputs are written under the ignored `build/rasterregions_examples/` directory and are not committed evidence.

## Examples

- `example_01_auto_regions.py`: automatic multi-row detection, reading order, cleanup, and diagnostics.
- `example_02_expected_count.py`: exact `expected_count=3` partitioning.
- `example_03_color_mask.py`: colored-ink masking while gray worksheet guides are excluded.
- `example_04_dark_paper.py`: dark ink on shaded paper.
- `example_05_component_isolation.py`: 8-connected components and anchor-based isolation.
- `example_06_numpy_input.py`: direct NumPy image input.
- `example_07_current_edge_cases.py`: sparse mixed ink, Otsu quantization, touching components, and one-region partitioning.
