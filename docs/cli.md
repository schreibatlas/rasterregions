# Command-line interface

The `rasterregions` command provides detection and artifact writing:

```bash
rasterregions detect page.png --output build/page
```

The output directory contains a JSON manifest, cleaned mask, overlay, and region crops by default. Use `--help` for the complete option list and `--version` to print the installed package version.

## Detection options

```text
rasterregions detect IMAGE --output DIR
  --expected-count N
  --expected-count-scope {row,page}
  --mode {auto,grayscale,dark,color,hybrid,adaptive}
  --padding PIXELS
  --min-confidence SCORE
  --json
  --no-crops
  --no-overlay
  --no-mask
```

`--expected-count-scope` defaults to `row`. Confidence and status semantics are described in [Detection](detection.md). The Python API supports an ROI through `detect_regions(..., roi=BBox(...))`; there is no ROI CLI option in this release.

`--json` prints the JSON-compatible detection payload to stdout. Artifact files are still written to `--output`. The manifest has `schema_version: 1`; see [Diagnostics](diagnostics.md).

The command exits with status `0` for `ok` or `ambiguous` detections and `2` for `failed` detection. A failed detection still writes its manifest and selected artifacts so that the failure can be reviewed.
