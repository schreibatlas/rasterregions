# Compatibility and schema policy

`rasterregions` is a pre-1.0 package. For the `0.1.x` release line, the documented public names, coordinate conventions, result semantics, and manifest schema are the supported consumer contract. Applications that require this line should constrain dependencies to `rasterregions>=0.1,<0.2`. A later `0.y` release may introduce incompatible API changes and must document them in its release notes.

Additive optional API parameters and additive manifest fields may be introduced within a compatible release line. Consumers should ignore unknown JSON object keys and must inspect `schema_version` rather than infer a payload version from the package version.

The `regions.json` payload currently uses `schema_version: 1`. Changes that alter the meaning or structure of existing fields, coordinate conventions, status values, or component-index namespace require a schema-version increment. Additive fields that preserve the existing field meanings do not. A schema change does not by itself imply that Python API compatibility changed, and vice versa.

Coordinates are half-open pixel coordinates in the normalized input orientation. Boolean masks use `True` for foreground. The result dataclasses are frozen, NumPy arrays are read-only snapshots, and Pillow image snapshots are independent mutable copies. See [Concepts](guide.md) and [Diagnostics](diagnostics.md) for the detailed contracts.


Package versions are derived from Git tags with `setuptools-scm`. A tagged checkout such as `v0.1.0` builds as `0.1.0`; a source archive without Git metadata uses the configured development fallback `0.1.0.dev0`.
