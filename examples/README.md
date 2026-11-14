# Examples

The `rasterregions_examples/` directory contains standalone scripts and a deterministic synthetic-asset generator. From the repository root, run the complete bundle with:

```bash
python examples/rasterregions_examples/run_all_examples.py
```

The runner regenerates its input assets and writes diagnostics under the ignored `build/rasterregions_examples/` directory. Generated run logs and outputs are disposable, not committed evidence, and are excluded from the Python wheel.

See `rasterregions_examples/README.md` for descriptions of the fixtures and scripts.
