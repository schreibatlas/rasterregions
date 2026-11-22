# Manual release procedure

This procedure is documentation only. No tag, package upload, or release publication is performed by this task. The repository's CI workflow validates code and build artifacts but does not publish releases.

## Before preparing a release

1. Choose a reviewed commit and confirm the working tree is clean.
2. Run the full test, lint, formatting, typing, docs, examples, and packaging checks described in CI and this documentation.
3. Confirm the version tag does not already exist locally or remotely:

   ```bash
   git tag --list v0.1.0
   git ls-remote --tags origin refs/tags/v0.1.0
   ```

4. Review `CHANGELOG.md`, project metadata, license, and wheel/sdist contents. Verify the `py.typed` marker is present in the wheel.

## Build from the release tag

The project version is dynamic and comes from Git tags through `setuptools-scm`. The tag `v0.1.0` must therefore exist before building the final distributions. A source archive without Git metadata uses the configured development fallback and is not a substitute for building from the release tag.

Planned release records and `Unreleased` changelog sections do not set the package version. Before a release tag exists, setuptools-scm derives a development version from the latest tag and commits; this checkout builds as `0.2.7.dev0` plus Git-local metadata from `v0.2.6`. The final `0.2.7` version is generated from the release tag after the reviewed code is committed. Do not edit the generated `rasterregions/_version.py` directly.


After an explicitly approved release decision, create and check out the annotated tag:

```bash
git tag -a v0.1.0 -m "rasterregions 0.1.0"
git checkout --detach v0.1.0
```

Build both artifacts and validate their metadata:

```bash
rm -rf dist
python -m pip install --upgrade build twine
python -m build --sdist --wheel
python -m twine check dist/*
python scripts/check_artifacts.py dist
```

Install each artifact separately from outside the checkout:

```bash
artifacts="$PWD/dist"
cd /tmp
python -m venv /tmp/rasterregions-wheel
/tmp/rasterregions-wheel/bin/python -m pip install "$artifacts"/*.whl
python -m venv /tmp/rasterregions-sdist
/tmp/rasterregions-sdist/bin/python -m pip install "$artifacts"/*.tar.gz
```

Inspect the contents before upload. The wheel should contain the `rasterregions` package and `rasterregions/py.typed`; the sdist should contain the build inputs and license. Run the following smoke test from `/tmp` in both virtual environments. The version and CLI must report `0.1.0`, and the printed module path must point into the active environment, not the checkout.

```bash
cd /tmp
for environment in /tmp/rasterregions-wheel /tmp/rasterregions-sdist; do
  "$environment/bin/rasterregions" --version
  "$environment/bin/python" - <<'PY'
from PIL import Image, ImageDraw
import rasterregions
from rasterregions import detect_regions, foreground_mask

image = Image.new("RGB", (120, 60), "white")
draw = ImageDraw.Draw(image)
draw.rectangle((15, 10, 35, 40), fill="black")
draw.rectangle((75, 10, 95, 40), fill="black")
mask = foreground_mask(image, mode="dark")
result = detect_regions(image, expected_count=2)
assert rasterregions.__version__ == "0.1.0"
assert mask.mask[20, 20]
assert result.status == "ok" and len(result.regions) == 2
print(rasterregions.__file__)
PY
done
```
## TestPyPI rehearsal

Use the same artifacts built from `v0.1.0`. TestPyPI upload is a publication action and must be explicitly approved:

```bash
python -m twine upload --repository testpypi dist/*
python -m venv /tmp/rasterregions-testpypi
/tmp/rasterregions-testpypi/bin/python -m pip install --index-url https://test.pypi.org/simple/ --no-deps rasterregions==0.1.0
/tmp/rasterregions-testpypi/bin/python -m pip install numpy Pillow
cd /tmp
source /tmp/rasterregions-testpypi/bin/activate
```

Run the smoke test above in the activated TestPyPI environment. Do not use the repository checkout or an editable install.

## Production publication and post-publish smoke test

Only after the TestPyPI rehearsal succeeds and a maintainer explicitly approves publication, push the tag and upload the same validated artifacts to PyPI:

```bash
git push origin v0.1.0
python -m twine upload dist/*
python -m venv /tmp/rasterregions-pypi
/tmp/rasterregions-pypi/bin/python -m pip install rasterregions==0.1.0
cd /tmp
source /tmp/rasterregions-pypi/bin/activate
```

In a fresh environment outside the checkout, install `rasterregions==0.1.0` from PyPI and repeat the version, CLI, import, mask, and detection smoke tests. Check the published project metadata links and release page. Reconcile the release record and changelog only after the maintainer confirms the release shipped.

The package URLs in `pyproject.toml` point to the verified GitHub repository, its documentation source, issues, and changelog. A Read the Docs site and a currently published PyPI project page were not verified, so they are not listed as project URLs.
