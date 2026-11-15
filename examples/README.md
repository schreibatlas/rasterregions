# Examples

The repository's examples are flat under `examples/`. From the repository root, install the editable package and run the canonical bundle:

```bash
python -m pip install -e .
python examples/run_all.py
```

`run_all.py` regenerates all eight source-controlled input fixtures, then runs the numbered scripts in explicit order. It includes `00_quickstart.py`, which generates its own small synthetic image. Detection artifacts are written under the ignored `build/examples/` directory and are disposable. The Python package wheel does not include examples.

## Scripts

- `00_quickstart.py`: generate a small synthetic row, detect three regions, and write diagnostics.
- `01_auto_regions.py`: multi-row detection, cleanup, and reading order.
- `02_expected_count.py`: expected-count partitioning.
- `03_color_mask.py`: color foreground with gray worksheet guides.
- `04_dark_paper.py`: dark ink on shaded paper.
- `05_component_isolation.py`: connected components and anchor isolation.
- `06_numpy_input.py`: pass a NumPy image directly.
- `07_edge_cases.py`: mixed content, grayscale quantization, touching components, and one-region partitioning.

Fixture generation can also be run by itself with `python examples/generate_assets.py`. The documentation figures are regenerated or checked with `python docs/generate_visual_examples.py` and `python docs/generate_visual_examples.py --check`; see [the visual examples page](../docs/examples.md).
