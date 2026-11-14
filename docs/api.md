# API reference

The package-level API re-exports the principal image, detection, configuration, and model types.

```{eval-rst}
.. automodule:: rasterregions
   :members:
   :undoc-members:
```

## Output helpers

```{eval-rst}
.. autofunction:: rasterregions.io.detection_payload

.. autofunction:: rasterregions.io.write_detection
```

`DetectionResult`, `MaskResult`, `Region`, and `DetectionDiagnostics` are frozen data models. Image snapshots returned in the result are independent Pillow images and remain mutable by the caller. Component coordinate arrays are copied and read-only.
