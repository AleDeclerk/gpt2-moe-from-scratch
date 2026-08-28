"""Download the Tiny Shakespeare dataset into `data/`.

The file has about 1.1 MB of text, and it is the training corpus for the whole
course. Chapter 1 explains what the course does with it.

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
        print(f"{DESTINATION} exists already, with {size} bytes. Nothing to do.")
        return

    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    print(f"Download {URL} ...")
    try:
        with urllib.request.urlopen(URL, timeout=30) as response:
            text = response.read().decode("utf-8")
    except Exception as error:  # noqa: BLE001 - the message must stay readable
        sys.exit(f"The download failed: {error}")

    DESTINATION.write_text(text, encoding="utf-8")
    print(f"OK. {len(text)} characters in {DESTINATION}")


if __name__ == "__main__":
    main()
