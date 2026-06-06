"""Shared helpers for the extraction scripts (issues #3, #6).

Extracted to remove the verbatim duplication of ``parse_script_text``
between ``extract_text.py`` and ``extract_shared_space.py`` (Rule of
Three: same body, different files).

The function is intentionally tiny (3 filter rules, 1 file read) — it
was promoted here only because two callers needed identical semantics.
A new caller that diverges in filter logic should inline a local helper
or extend this module with a parameter, NOT silently fork the parser.
"""

from __future__ import annotations

from pathlib import Path


def parse_script_text(script_path: Path) -> list[str]:
    """Extract clean Italian sentences from ``script.md``.

    Filters:
      - lines starting with ``###`` (markdown headers)
      - lines containing ``|`` (the GitHub-project marker)
      - empty / whitespace-only lines

    Returns:
        list of paragraph-level sentence strings (whitespace-stripped).
    """
    if not script_path.is_file():
        raise FileNotFoundError(f"script.md not found: {script_path}")
    sentences: list[str] = []
    for raw in script_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("###"):
            continue
        if "|" in line:
            continue
        sentences.append(line)
    return sentences


__all__ = ["parse_script_text"]
