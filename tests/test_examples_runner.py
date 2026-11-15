import runpy
from pathlib import Path


def test_runner_lists_every_numbered_example_in_order() -> None:
    root = Path(__file__).resolve().parents[1]
    examples = root / "examples"
    runner = runpy.run_path(str(examples / "run_all.py"), run_name="example_runner")
    expected = tuple(path.name for path in sorted(examples.glob("[0-9][0-9]_*.py")))

    assert runner["EXAMPLE_SCRIPTS"] == expected
