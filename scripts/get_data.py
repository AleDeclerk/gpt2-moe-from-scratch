"""Baja el dataset Tiny Shakespeare a `data/`.

El archivo tiene alrededor de 1.1 MB de texto, y es el corpus de entrenamiento
de todo el curso. El capítulo 1 explica qué hace el curso con él.

    uv run python scripts/get_data.py
"""

import sys
import urllib.request
from pathlib import Path

URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/"
    "tinyshakespeare/input.txt"
)
DESTINATION = Path(__file__).resolve().parent.parent / "data" / "tinyshakespeare.txt"


def main() -> None:
    if DESTINATION.exists():
        size = DESTINATION.stat().st_size
        print(f"{DESTINATION} ya existe, con {size} bytes. No hay nada que hacer.")
        return

    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    print(f"Bajando {URL} ...")
    try:
        with urllib.request.urlopen(URL, timeout=30) as response:
            text = response.read().decode("utf-8")
    except Exception as error:  # noqa: BLE001 - el mensaje tiene que quedar legible
        sys.exit(f"Falló la descarga: {error}")

    DESTINATION.write_text(text, encoding="utf-8")
    print(f"OK. {len(text)} caracteres en {DESTINATION}")


if __name__ == "__main__":
    main()
