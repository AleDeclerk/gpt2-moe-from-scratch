"""Measure the real progress of the course, and write progress.json.

The website reads that file. The numbers come from a test run, so the site
cannot say that a chapter is complete while the tests fail.

    uv run python scripts/sync_progress.py

The run uses MOE_TARGET=exercise, so it measures your code, not the reference.
Tests with the mark `slow` stay out, because they need a network download.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ElementTree
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHAPTERS = ROOT / "chapters"
PACKAGE = ROOT / "gpt2moe"
MANIFEST = CHAPTERS / "manifest.json"
OUTPUT = ROOT / "progress.json"


def run_chapter_tests(chapter: Path) -> list[dict]:
    """Run the tests of one chapter, and get the result of each one.

    Returns:
        A list of dicts with the keys `name` and `passed`.
    """
    env = os.environ.copy()
    env["MOE_TARGET"] = "exercise"

    with tempfile.TemporaryDirectory() as work:
        report = Path(work) / "report.xml"
        subprocess.run(
            [
                sys.executable, "-m", "pytest", str(chapter),
                "-m", "not slow",
                "--junit-xml", str(report),
                "-p", "no:cacheprovider",
                "-q", "--tb=no",
            ],
            cwd=ROOT,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if not report.exists():
            return []

        tree = ElementTree.parse(report)
        results = []
        for case in tree.iter("testcase"):
            # A case without a child element passed. A failure, an error, or a
            # skip adds a child, so the test does not count as green.
            failed = any(
                case.find(tag) is not None
                for tag in ("failure", "error", "skipped")
            )
            results.append({"name": case.get("name", "?"), "passed": not failed})
        return results


def status_of(exists: bool, passed: int, total: int) -> str:
    """Get the state of one chapter from its test results."""
    if not exists:
        return "planned"
    if total == 0 or passed == 0:
        return "todo"
    if passed < total:
        return "in_progress"
    return "done"


def main() -> None:
    manifest = json.loads(MANIFEST.read_text())

    known = {c["dir"] for part in manifest["parts"] for c in part["chapters"]}
    present = {p.name for p in CHAPTERS.iterdir() if p.is_dir() and p.name.startswith("ch")}
    missing = present - known
    if missing:
        sys.exit(
            f"These chapter directories are not in the manifest: {sorted(missing)}.\n"
            f"Add them to {MANIFEST}, so the website can show them."
        )

    parts = []
    tests_total = tests_passed = chapters_done = chapters_total = 0

    for part in manifest["parts"]:
        chapters = []
        for entry in part["chapters"]:
            directory = CHAPTERS / entry["dir"]
            exists = directory.is_dir()
            module = entry["dir"].split("_", 1)[1]
            results = run_chapter_tests(directory) if exists else []
            passed = sum(1 for r in results if r["passed"])

            if exists:
                print(f"  {entry['dir']:<18} {passed}/{len(results)} tests")

            chapters.append({
                **entry,
                "module": module,
                "exists": exists,
                "promoted": (PACKAGE / f"{module}.py").exists(),
                "tests": results,
                "tests_total": len(results),
                "tests_passed": passed,
                "status": status_of(exists, passed, len(results)),
            })

            chapters_total += 1
            tests_total += len(results)
            tests_passed += passed
            if chapters[-1]["status"] == "done":
                chapters_done += 1

        parts.append({**part, "chapters": chapters})

    document = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "course": manifest["course"],
        "repo": manifest["repo"],
        "totals": {
            "chapters_total": chapters_total,
            "chapters_done": chapters_done,
            "chapters_written": sum(1 for p in parts for c in p["chapters"] if c["exists"]),
            "tests_total": tests_total,
            "tests_passed": tests_passed,
        },
        "parts": parts,
    }
    OUTPUT.write_text(json.dumps(document, indent=2) + "\n")
    print(f"\nOK. {tests_passed}/{tests_total} tests, {chapters_done}/{chapters_total} chapters done.")
    print(f"Written to {OUTPUT}")


if __name__ == "__main__":
    main()
