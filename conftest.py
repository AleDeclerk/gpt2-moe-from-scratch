"""Test support for the whole course.

Every chapter has two versions of the same module: `exercise.py`, which you
write, and `solution.py`, the reference. The `target` fixture loads one of
them. The environment variable MOE_TARGET selects which one.

    uv run pytest chapters/ch00_tensors                    # tests your code
    MOE_TARGET=solution uv run pytest chapters/ch00_tensors  # tests the reference
"""

import importlib.util
import os
import sys

import pytest

VALID_TARGETS = ("exercise", "solution")


@pytest.fixture(scope="module")
def target(request):
    """Import the module under test from the directory of the test file."""
    name = os.environ.get("MOE_TARGET", "exercise")
    if name not in VALID_TARGETS:
        raise ValueError(f"MOE_TARGET must be one of {VALID_TARGETS}, not {name!r}")

    chapter = request.path.parent
    path = chapter / f"{name}.py"
    if not path.exists():
        pytest.fail(f"{path} does not exist")

    # A unique module name prevents a collision between chapters, because every
    # chapter has a file with the same name.
    unique = f"{chapter.name}__{name}"
    spec = importlib.util.spec_from_file_location(unique, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[unique] = module
    spec.loader.exec_module(module)
    return module
