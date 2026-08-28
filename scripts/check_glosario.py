"""Verificar que los enlaces al glosario salieron bien en el sitio construido.

El enlazado automático es la pieza más frágil del sitio: mete etiquetas <a>
adentro del HTML ya armado. Este chequeo mira el resultado real y falla si
algo salió mal.

    uv run python scripts/check_glosario.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "web" / ".next" / "server" / "app"
GLOSARIO = ROOT / "GLOSARIO.md"


def anclas_publicadas() -> set[str]:
    """Leer las anclas reales de la página del glosario ya construida.

    El chequeo no vuelve a calcular los slugs: los lee del HTML que el sitio
    publica. Si los recalculara, estaría comparando el sitio contra una
    segunda implementación que puede tener sus propios errores, y de hecho la
    primera versión de este script fallaba justo por eso, con los acentos.
    """
    pagina = BUILD / "glosario.html"
    if not pagina.exists():
        sys.exit(f"No existe {pagina}. Corré `npm run build` en web/ primero.")
    html = pagina.read_text(encoding="utf-8")
    # Los términos son los <div id="..."> de la lista de definiciones.
    return set(re.findall(r'<div id="([^"]+)"', html))


def paginas() -> list[Path]:
    """Las páginas de capítulo y la intro, que son las que llevan enlaces."""
    return sorted(
        [p for p in BUILD.rglob("*.html") if "chapters" in str(p) or "intro" in str(p)]
    )


def main() -> None:
    problemas: list[str] = []
    anclas = anclas_publicadas()
    print(f"{len(anclas)} términos publicados en el glosario")

    archivos = paginas()
    if not archivos:
        sys.exit(f"No hay HTML construido en {BUILD}. Corré `npm run build` en web/ primero.")

    for pagina in archivos:
        html = pagina.read_text(encoding="utf-8")
        enlaces = re.findall(r'href="/glosario#([^"]+)"', html)

        # 1. Cada término se enlaza una vez por página, no veinte.
        repetidos = {s for s in enlaces if enlaces.count(s) > 1}
        if repetidos:
            problemas.append(f"{pagina.name}: enlaces repetidos para {sorted(repetidos)}")

        # 2. Nunca adentro de un bloque de código.
        for bloque in re.findall(r"<code[^>]*>.*?</code>", html, re.S):
            if "/glosario#" in bloque:
                problemas.append(f"{pagina.name}: hay un enlace al glosario dentro de <code>")
                break

        # 3. Nunca adentro de un título.
        for titulo in re.findall(r"<h[1-6][^>]*>.*?</h[1-6]>", html, re.S):
            if "/glosario#" in titulo:
                problemas.append(f"{pagina.name}: hay un enlace al glosario dentro de un título")
                break

        # 4. Ningún enlace apunta a un ancla que no existe.
        rotos = sorted(set(enlaces) - anclas)
        if rotos:
            problemas.append(f"{pagina.name}: anclas sin término: {sorted(rotos)}")

        print(f"  {pagina.name:<28} {len(set(enlaces))} términos enlazados")

    if problemas:
        print("\nPROBLEMAS:")
        for p in problemas:
            print(f"  - {p}")
        sys.exit(1)
    print("\nOK. Los enlaces al glosario están bien.")


if __name__ == "__main__":
    main()
