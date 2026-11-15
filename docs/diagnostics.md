# Diagnostics and output files

## Python diagnostics

`DetectionResult.diagnostics` records image dimensions, selected mask method and confidence, cleanup decisions, parameters, optional ROI, and expected-count partition boundaries. `DetectionResult.warnings` combines mask and detection warnings. Arrays in mask and component records are read-only snapshots; Pillow image snapshots are independent and mutable.

`detection_payload(result)` returns the JSON-compatible payload without writing files. The current manifest has `schema_version: 1` and includes:

- package version, source filename, status, requested and detected counts;
- image dimensions and effective parameters, including an optional ROI;
- mask method, threshold, confidence, foreground fraction, and warnings;
- cleanup decisions and their component bboxes;
- rows, partition boundaries, warnings, and region geometry/confidence.

Component indexes in cleanup decisions, rows, and regions use the explicitly named `raw_mask` namespace. Removed components may therefore leave gaps in indexes for retained components.

## Write artifacts

```python
from rasterregions.io import write_detection

manifest_path = write_detection(result, "build/page")
```

By default, the output directory contains:

```text
regions.json
mask.png
regions.png
crops/
  region-000.png
  region-001.png
```

Use `include_crops=False`, `include_overlay=False`, or `include_mask=False` to omit those artifacts. `regions.json` is written last as the completion marker. Reruns remove stale managed crop files and disabled mask/overlay files, while preserving unrelated files in the output directory. Image files and the manifest are replaced atomically.

## Manifest compatibility

The Python API and JSON schema version are separate compatibility contracts. See [Compatibility policy](compatibility.md) for the pre-1.0 API policy and the rules for changing `schema_version`.
