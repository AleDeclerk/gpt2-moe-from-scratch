"""Copia el código validado de un capítulo al paquete `gpt2moe`.

El curso arma un solo paquete, capítulo a capítulo. Cuando pasan los tests de
un capítulo, este script copia tu `exercise.py` a `gpt2moe/`. El capítulo que
sigue importa desde el paquete, así que importa tu propio código.

    uv run python scripts/promote.py ch04
    uv run python scripts/promote.py ch04 --from-solution
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHAPTERS = ROOT / "chapters"
PACKAGE = ROOT / "gpt2moe"


def find_chapter(prefix: str) -> Path:
    """Busca el directorio de capítulo que empieza con `prefix`, por ejemplo ch04."""
    matches = sorted(p for p in CHAPTERS.iterdir() if p.is_dir() and p.name.startswith(prefix))
    if not matches:
        available = ", ".join(sorted(p.name for p in CHAPTERS.iterdir() if p.is_dir()))
        sys.exit(f"Ningún capítulo empieza con {prefix!r}. Disponibles: {available}")
    if len(matches) > 1:
        sys.exit(f"{prefix!r} es ambiguo: {', '.join(p.name for p in matches)}")
    return matches[0]


def module_name(chapter: Path) -> str:
    """Devuelve el nombre del módulo destino. ch04_attention pasa a ser attention.py."""
    return chapter.name.split("_", 1)[1]


def tests_pass(chapter: Path) -> bool:
    """Corre los tests de un capítulo contra el código del ejercicio."""
    print(f"Corriendo los tests de {chapter.name} ...")
    env = os.environ.copy()
    env["MOE_TARGET"] = "exercise"
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(chapter), "-q"],
        cwd=ROOT,
        env=env,
    )
    return result.returncode == 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("chapter", help="prefijo del capítulo, por ejemplo ch04")
    parser.add_argument(
        "--from-solution",
        action="store_true",
        help="promueve el código de referencia en lugar del tuyo, y saltea los tests",
    )
    args = parser.parse_args()

    chapter = find_chapter(args.chapter)
    source_name = "solution" if args.from_solution else "exercise"
    source = chapter / f"{source_name}.py"
    if not source.exists():
        sys.exit(f"{source} no existe")

    if not args.from_solution and not tests_pass(chapter):
        sys.exit(
            f"\nLos tests de {chapter.name} fallan, así que no se movió nada.\n"
            f"Corregí {source}, o usá --from-solution para seguir con el código de referencia."
        )

    PACKAGE.mkdir(exist_ok=True)
    destination = PACKAGE / f"{module_name(chapter)}.py"
    header = (
        f'"""Promovido desde chapters/{chapter.name}/{source_name}.py.\n\n'
        f"No edites este archivo. Editá el capítulo, y promovelo de nuevo.\n"
        f'"""\n\n'
    )
    destination.write_text(header + source.read_text())
    print(f"OK. {chapter.name}/{source_name}.py ahora es gpt2moe/{destination.name}")


if __name__ == "__main__":
    main()
