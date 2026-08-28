"""Soporte de tests para todo el curso.

Cada capítulo tiene dos versiones del mismo módulo: `exercise.py`, que lo
escribís vos, y `solution.py`, que es la referencia. El fixture `target` carga
uno de los dos. La variable de entorno MOE_TARGET elige cuál.

    uv run pytest chapters/ch00_tensors                    # testea tu código
    MOE_TARGET=solution uv run pytest chapters/ch00_tensors  # testea la referencia
"""

import importlib.util
import os
import sys

import pytest

VALID_TARGETS = ("exercise", "solution")


@pytest.fixture(scope="module")
def target(request):
    """Importa el módulo bajo test desde el directorio del archivo de test."""
    name = os.environ.get("MOE_TARGET", "exercise")
    if name not in VALID_TARGETS:
        raise ValueError(f"MOE_TARGET tiene que ser uno de {VALID_TARGETS}, no {name!r}")

    chapter = request.path.parent
    path = chapter / f"{name}.py"
    if not path.exists():
        pytest.fail(f"{path} no existe")

    # Un nombre de módulo único evita la colisión entre capítulos, porque todos
    # tienen un archivo que se llama igual.
    unique = f"{chapter.name}__{name}"
    spec = importlib.util.spec_from_file_location(unique, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[unique] = module
    spec.loader.exec_module(module)
    return module
