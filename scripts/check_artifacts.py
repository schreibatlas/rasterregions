from __future__ import annotations

import sys
import tarfile
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

from packaging.utils import parse_sdist_filename, parse_wheel_filename


def main() -> None:
    dist = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("dist")
    wheels = list(dist.glob("*.whl"))
    sdists = list(dist.glob("*.tar.gz"))
    assert len(wheels) == 1, f"expected one wheel in {dist}, found {len(wheels)}"
    assert len(sdists) == 1, f"expected one sdist in {dist}, found {len(sdists)}"

    wheel_name, wheel_version, _, _ = parse_wheel_filename(wheels[0].name)
    sdist_name, sdist_version = parse_sdist_filename(sdists[0].name)
    assert wheel_name == sdist_name == "rasterregions"
    assert wheel_version == sdist_version, (
        f"artifact versions differ: {wheel_version} != {sdist_version}"
    )

    with ZipFile(wheels[0]) as wheel:
        wheel_paths = set(wheel.namelist())
        metadata_paths = [path for path in wheel_paths if path.endswith(".dist-info/METADATA")]
        assert len(metadata_paths) == 1, f"expected one wheel METADATA, found {metadata_paths}"
        metadata = wheel.read(metadata_paths[0]).decode()
        assert "Name: rasterregions\n" in metadata
        assert f"Version: {wheel_version}\n" in metadata
        assert "Description-Content-Type: text/markdown" in metadata
        assert "Project-URL: Repository, https://github.com/schreibatlas/rasterregions" in metadata
        assert "rasterregions/py.typed" in wheel_paths
        assert not any(PurePosixPath(path).parts[0] == "examples" for path in wheel_paths)

    with tarfile.open(sdists[0], "r:gz") as sdist:
        sdist_paths = [PurePosixPath(path) for path in sdist.getnames()]
        metadata_paths = [
            path
            for path in sdist_paths
            if path.name == "PKG-INFO" and path.parent.name != "rasterregions.egg-info"
        ]
        assert len(metadata_paths) == 1, f"expected one sdist PKG-INFO, found {metadata_paths}"
        metadata = sdist.extractfile(metadata_paths[0].as_posix())
        assert metadata is not None
        metadata_text = metadata.read().decode()
        assert "Name: rasterregions\n" in metadata_text
        assert f"Version: {sdist_version}\n" in metadata_text
        assert any(path.parts[-2:] == ("rasterregions", "py.typed") for path in sdist_paths)
        assert any(path.name == "README.md" for path in sdist_paths)
        assert any(path.name == "LICENSE" for path in sdist_paths)
        assert not any(".ledger" in path.parts for path in sdist_paths)
        assert not any(".github" in path.parts for path in sdist_paths)

    print(f"Verified wheel and sdist for rasterregions {wheel_version}")


if __name__ == "__main__":
    main()
