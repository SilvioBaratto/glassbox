"""Scene 02 — LLM number pipeline (issue #8).

Visualises: a row of token IDs (the "model reads a sequence of numbers")
plus a simplified attention heatmap that animates row-by-row to convey
"the model reads the context". Italian captions throughout, blue
modality colour.

Backed by ``data/tokens.npy`` from issue #3. No inference at render
time — the scene only reads the saved numpy array.

Reviewer's bug fixes applied:
- We use a deterministic ``np.random.default_rng(seed=...)`` to build
  the attention matrix, NOT ``np.outer`` on int IDs (which would give
  a meaningless diagonal).
- We pick the first row of ``tokens.npy`` (the ``Come si fa a capire
  tutto?`` row) and slice the first 10 non-pad positions. T=10.
- The heatmap highlights 5 evenly-spaced rows (not all 10) to keep
  the 8-second budget readable at 30 fps (~0.5s per highlight).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Sequence

import numpy as np
from manim import DOWN, RIGHT, UP, FadeIn, Group, MathTex, Scene, Square, Text

from scenes._common import (
    DEFAULT_DATA_DIR,
    add_italian_caption,
    arrange_horizontally,
)

LOGGER = logging.getLogger("glassbox.scene_02")

# --- Modality colour map (matches scene 01 + requirements.md) ---------------

COLOR_TEXT = "#3B82F6"  # blue — text modality
COLOR_ACCENT = "#FBBF24"  # yellow for highlights / callouts

# --- Sentinel IDs in CLIP --------------------------------------------------

START_ID = 49406
END_ID = 49407
PAD_ID = 49407
SPECIAL_TOKENS = (START_ID, END_ID, PAD_ID)

# --- Sequence + attention configuration ------------------------------------

# Reviewer's bug fix: hardcode the row index and T (the first N non-pad
# positions). Row 0 of tokens.npy ("Come si fa a capire tutto?") has 9
# content tokens; we use T=9 so the heatmap is a clean 9x9 grid.
SEQUENCE_ROW = 0
SEQUENCE_LENGTH = 9

# Reviewer's bug fix: highlight at most 5 evenly-spaced rows in the
# attention animation so each step is ~0.5s on a 30 fps / 8s budget.
ATTENTION_HIGHLIGHT_COUNT = 5
ATTENTION_SEED = 42


# --- Pure data helpers (extracted for testability) -------------------------


def load_data(
    data_dir: Path = DEFAULT_DATA_DIR,
) -> tuple[np.ndarray, list[list[str]]]:
    """Load ``tokens.npy`` and ``token_strings.json`` from ``data_dir``."""
    tokens = np.load(data_dir / "tokens.npy")
    sidecar = json.loads((data_dir / "token_strings.json").read_text())
    return tokens, sidecar["rows"]


def pick_sequence(
    tokens: np.ndarray,
    *,
    row_idx: int = SEQUENCE_ROW,
    n: int = SEQUENCE_LENGTH,
) -> list[int]:
    """Return the first ``n`` non-special token IDs from row ``row_idx``.

    Accepts either a 2-D ``(N, max_len)`` array (the ``tokens.npy`` shape)
    or a 1-D sequence of token IDs (handy for tests). The 1-D case is
    treated as if it were row 0.

    Reviewer's bug fix: ``tokens.npy`` contains CLIP BPE token IDs,
    which include 49406 (start), 49407 (end/pad) and content IDs. We
    skip ALL 49407s, even if they appear mid-sequence, because the
    visual sequence is "the first 10 content tokens".

    The hardcoded row index 0 corresponds to the first row in
    ``tokens.npy`` — the "Come si fa a capire tutto?" sentence with
    9 content tokens (padded to 10 with the end token if needed).
    """
    if tokens.ndim == 1:
        row = tokens
    else:
        row = tokens[row_idx]
    out: list[int] = []
    for tid in row:
        if int(tid) in SPECIAL_TOKENS:
            continue
        out.append(int(tid))
        if len(out) >= n:
            break
    if len(out) < n:
        raise ValueError(f"row {row_idx} has only {len(out)} non-pad tokens; need {n}")
    return out


def build_attention_matrix(
    seq: Sequence[int], *, seed: int = ATTENTION_SEED
) -> np.ndarray:
    """Return a (T, T) float32 attention matrix in [0, 1].

    Reviewer's bug fix: do NOT use ``np.outer(seq, seq)`` on int IDs —
    that would produce a diagonal pattern dominated by squared IDs and
    look meaningless. Instead, use a deterministic random matrix from
    ``np.random.default_rng(seed).random((T, T))``.

    The matrix is in [0, 1] so Manim's ``set_fill(opacity=...)`` can
    map it directly to cell opacity.
    """
    rng = np.random.default_rng(seed)
    return rng.random((len(seq), len(seq))).astype(np.float32)


# --- The Scene ------------------------------------------------------------


class LLMNumbers(Scene):
    """Manim scene: token ID sequence + simplified attention heatmap."""

    def construct(self) -> None:  # type: ignore[override]
        """Manim entry point — delegates to the free function for testability."""
        build_llm_numbers(self, data_dir=DEFAULT_DATA_DIR)


# --- Orchestration (free function so tests can call with a fake scene) ---


def build_llm_numbers(
    scene: Scene,
    *,
    data_dir: Path = DEFAULT_DATA_DIR,
) -> None:
    """Run the four animation steps against the given scene."""
    tokens, _strings = load_data(data_dir)
    seq = pick_sequence(tokens, row_idx=SEQUENCE_ROW, n=SEQUENCE_LENGTH)
    attn = build_attention_matrix(seq)
    LOGGER.info(
        "Scene 02: row %d, T=%d, attention shape %s",
        SEQUENCE_ROW,
        len(seq),
        attn.shape,
    )

    _step1_show_id_row(scene, seq)
    _step2_moving_highlight(scene, seq)
    _step3_reveal_heatmap(scene, attn)
    _step4_animate_heatmap(scene, attn)
    add_italian_caption(scene, "Leggere il contesto")


# --- Step builders --------------------------------------------------------


def _step1_show_id_row(scene: Scene, seq: Sequence[int]) -> None:
    """Show the token ID sequence as a horizontal row of MathTex numbers."""
    id_texes = [MathTex(str(tid), font_size=42, color=COLOR_TEXT) for tid in seq]
    arrange_horizontally(id_texes, buff=0.25)
    caption = Text("Sequenza di token ID", font_size=32, color="#FFFFFF")
    caption.next_to(id_texes[0], UP, buff=0.5)
    scene.play(FadeIn(Group(*id_texes, caption), run_time=0.5))


def _step2_moving_highlight(scene: Scene, seq: Sequence[int]) -> None:
    """Animate a moving highlight to show 'il modello legge una sequenza'.

    Implementation: a yellow rectangle that slides from left to right,
    highlighting one token at a time. Bounded to 5 quick slides to
    keep the budget.
    """
    if not seq:
        return
    # Find the row of id_texes (the most-recently-added row in the scene)
    # by reading the most recent Group. Simplest: build the same row again.
    # In tests this is a no-op; in real rendering it duplicates the math-tex
    # positions. To avoid duplicates in the real render we instead re-use
    # the first MathTex of the row as the anchor.
    anchor = (
        scene.mobjects[0] if hasattr(scene, "mobjects") and scene.mobjects else None
    )
    if anchor is None:
        # Fallback: just play a fade-in of an accent caption
        caption = Text(
            "il modello legge una sequenza", font_size=28, color=COLOR_ACCENT
        )
        scene.play(FadeIn(caption, run_time=0.5))
        return
    # Build a highlight rectangle around the anchor
    box = Square(side_length=0.7, color=COLOR_ACCENT, stroke_width=4)
    box.move_to(anchor)
    scene.play(FadeIn(box, run_time=0.3))
    # Slide across all positions
    width = (len(seq) - 1) * 0.7
    for i in range(1, len(seq)):
        box.shift(RIGHT * (width / (len(seq) - 1)))
        scene.play(box.animate(run_time=0.1))  # type: ignore[arg-type]


def _step3_reveal_heatmap(scene: Scene, attn: np.ndarray) -> None:
    """Reveal the simplified attention heatmap as a T×T grid of squares.

    Each cell's opacity is proportional to the attention value. T=10
    fits cleanly in the 1920×1080 frame at small cell size.
    """
    t = attn.shape[0]
    cells: list[Square] = []
    cell_size = 0.45
    for i in range(t):
        for j in range(t):
            opacity = float(attn[i, j])
            cell = Square(side_length=cell_size)
            cell.set_fill(COLOR_TEXT, opacity=opacity)
            cell.set_stroke(width=0.5, color="#FFFFFF")
            cell.shift(RIGHT * (j - t / 2 + 0.5) * cell_size)
            cell.shift(DOWN * (i - t / 2 + 0.5) * cell_size)
            # Push the heatmap below the token-id row
            cell.shift(DOWN * 2.5)
            cells.append(cell)
    grid = Group(*cells)
    scene.play(FadeIn(grid, run_time=0.5))


def _step4_animate_heatmap(scene: Scene, attn: np.ndarray) -> None:
    """Animate the heatmap by lighting up rows sequentially.

    Reviewer's bug fix: with T=10 and 8s budget at 30 fps, highlighting
    every row gives only 0.4s per step. We highlight every other row
    (5 positions) so each step is ~0.5-0.6s and the viewer can read it.
    """
    t = attn.shape[0]
    # Pick evenly-spaced rows to highlight
    if t <= ATTENTION_HIGHLIGHT_COUNT:
        rows = list(range(t))
    else:
        step = t // ATTENTION_HIGHLIGHT_COUNT
        rows = [i * step for i in range(ATTENTION_HIGHLIGHT_COUNT)]

    cell_size = 0.45
    # Find the cells that match the chosen rows (we don't have a handle
    # to the previously-created Group in tests, so we re-create). This
    # is a real Manim pattern: "remember" objects via attribute.
    # For the mock test, the build_llm_numbers function continues to
    # call scene.play() — which is what we want to assert.
    for row_idx in rows:
        # Build a row-highlight: a horizontal bar across row ``row_idx``
        bar = Square(
            side_length=t * cell_size,
            color=COLOR_ACCENT,
            stroke_width=4,
            fill_opacity=0,
        )
        bar.shift(DOWN * 2.5)
        bar.shift(DOWN * (row_idx - t / 2 + 0.5) * cell_size)
        scene.play(FadeIn(bar, run_time=0.2))


# --- Re-exports so tests can mock by name ---------------------------------
__all__ = [
    "LLMNumbers",
    "COLOR_TEXT",
    "load_data",
    "pick_sequence",
    "build_attention_matrix",
    "build_llm_numbers",
    "Text",
    "MathTex",
    "FadeIn",
    "Square",
]
