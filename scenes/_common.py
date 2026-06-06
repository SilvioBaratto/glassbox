"""Shared helpers for the 7 Manim scenes.

Extracted from scenes/01..07 to avoid verbatim duplication of the same
helpers across files (Rule of Three: ``_add_italian_caption`` was in
4 files, ``_arrange_horizontally`` in 2, ``load_data`` in 2). Pure helpers
only — no Manim scene logic lives here.

Importable from any scene module as:

    from scenes._common import add_italian_caption, arrange_horizontally
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from manim import DOWN, RIGHT, FadeIn, Text

# --- Constants shared across scenes ---------------------------------------

# White captions used in every scene; the modality-specific colour
# constants stay inside each scene file because they differ per modality.
COLOR_CAPTION = "#FFFFFF"

# Default data dir (one level up from scenes/).
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data"


# --- Pure data helpers (extracted for testability) -------------------------


def load_npy(path: Path) -> np.ndarray:
    """Load a single .npy file — thin wrapper for symmetry with load_json."""
    return np.load(path)


def load_json(path: Path):
    """Load a JSON file — thin wrapper for symmetry with load_npy."""
    import json

    return json.loads(path.read_text())


# --- Manim helpers (act on a fake scene too, so tests can exercise them) --


def arrange_horizontally(mobjects: Sequence, buff: float = 0.3) -> None:
    """Lay out ``mobjects`` left-to-right with the given buffer.

    Works on any mobject exposing ``.next_to(other, direction, buff=)``,
    so the tests' mock mobjects satisfy the duck-type.
    """
    for i in range(1, len(mobjects)):
        mobjects[i].next_to(mobjects[i - 1], RIGHT, buff=buff)


def add_italian_caption(
    scene, text: str, *, color: str = COLOR_CAPTION, font_size: int = 36
) -> None:
    """Add a caption at the bottom edge of ``scene`` and play FadeIn.

    Shared by scenes 02, 03, 04, 05. Scene 06/07 use specialised
    variants (panel-specific positions) and keep their own helpers.

    Uses module-level ``Text`` and ``FadeIn`` so that the test suite's
    ``patch.object(_common, "Text", _Text)`` rebinds the lookup.
    """
    caption = Text(text, font_size=font_size, color=color)
    caption.to_edge(DOWN)
    scene.play(FadeIn(caption, run_time=0.5))


__all__ = [
    "COLOR_CAPTION",
    "DEFAULT_DATA_DIR",
    "load_npy",
    "load_json",
    "arrange_horizontally",
    "add_italian_caption",
    "Text",
    "FadeIn",
]
