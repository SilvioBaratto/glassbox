"""Scene 01 — tokenization (issue #7).

Visualises: raw text → BPE sub-tokens → integer token IDs. All text in
Italian, colour-coded blue for the text modality.

Backed by ``data/tokens.npy`` and ``data/token_strings.json`` from
issue #3. No inference happens at render time — the scene only reads
the saved numpy / JSON files.

Note on the BPE example: the AC suggested ``gatto`` → ``ga``/``##tto``,
but the actual CLIP BPE tokenizer keeps ``gatto`` as a single token
(ID 166619). To illustrate BPE splits, this scene uses the first
``script.md`` row — ``"Come si fa a capire tutto?"`` — which produces
real subword splits (``cap``/``##ire``, ``tut``/``##to</w>``).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Sequence

import numpy as np
from manim import DOWN, UP, FadeIn, Group, MathTex, Scene, Text

from scenes._common import (
    COLOR_CAPTION,
    DEFAULT_DATA_DIR,
    arrange_horizontally,
)

LOGGER = logging.getLogger("glassbox.scene_01")

# --- Modality colour map (per requirements.md) -----------------------------

COLOR_TEXT = "#3B82F6"  # blue — text modality
COLOR_ACCENT = "#FBBF24"  # yellow for arrows / callouts

# --- Sentinel IDs in CLIP --------------------------------------------------

START_ID = 49406
END_ID = 49407
PAD_ID = 49407
SPECIAL_TOKENS = {"<|startoftext|>", "<|endoftext|>", "<|pad|>"}


# --- Pure data helpers (extracted for testability) -------------------------


def load_data(
    data_dir: Path = DEFAULT_DATA_DIR,
) -> tuple[np.ndarray, list[list[str]]]:
    """Load ``tokens.npy`` and ``token_strings.json`` from ``data_dir``."""
    tokens = np.load(data_dir / "tokens.npy")
    sidecar = json.loads((data_dir / "token_strings.json").read_text())
    return tokens, sidecar["rows"]


def pick_first_real_row(
    tokens: np.ndarray, strings: Sequence[Sequence[str]]
) -> tuple[int, list[int], list[str]]:
    """Pick the first row with a BPE split (i.e. ≥ 3 non-special tokens).

    Returns ``(row_index, token_ids, sub_token_strings)``. Strips leading
    ``<|startoftext|>`` and trailing ``<|endoftext|>`` / ``<|pad|>``.

    Reviewer's bug fix: the AC's ``gatto``/``ga``/``##tto`` example is
    wrong — CLIP BPE keeps ``gatto`` as one token. The first row of
    ``script.md`` has real subword splits.
    """
    for i, row_strings in enumerate(strings):
        content = [(j, t) for j, t in enumerate(row_strings) if t not in SPECIAL_TOKENS]
        if len(content) >= 3:
            ids = [int(tokens[i, j]) for j, _ in content]
            toks = [t for _, t in content]
            return i, ids, toks
    raise ValueError("no row with >= 3 content tokens found in data")


def find_split_indices(toks: Sequence[str]) -> list[int]:
    """Return indices into ``toks`` of every BPE sub-token of every split word.

    A word is "split" if it has a ``##`` continuation marker. The
    function returns the lead sub-token and every continuation.

    For ``toks = ["come", "si", "fa", "a", "cap", "##ire", "tut", "##to</w>", "?</w>"]``,
    this returns ``[4, 5, 6, 7]`` (the four sub-tokens of two real-split
    words: "cap/##ire" and "tut/##to</w>").
    """
    out: list[int] = []
    i = 0
    while i < len(toks):
        if toks[i].startswith("##"):
            i += 1
            continue
        if i + 1 < len(toks) and toks[i + 1].startswith("##"):
            out.append(i)
            j = i + 1
            while j < len(toks) and toks[j].startswith("##"):
                out.append(j)
                j += 1
            i = j
        else:
            i += 1
    return out


# --- The Scene ------------------------------------------------------------


class Tokenization(Scene):
    """Manim scene: raw text → BPE sub-tokens → integer token IDs."""

    def construct(self) -> None:  # type: ignore[override]
        """Manim entry point — delegates to the free function for testability."""
        build_tokenization(self, data_dir=DEFAULT_DATA_DIR)


# --- Orchestration (free function so tests can call with a fake scene) ---


def build_tokenization(
    scene: Scene,
    *,
    data_dir: Path = DEFAULT_DATA_DIR,
) -> None:
    """Run the three animation steps against the given scene."""
    tokens, strings = load_data(data_dir)
    row_idx, ids, toks = pick_first_real_row(tokens, strings)
    LOGGER.info("Scene 01 using row %d: %d content tokens", row_idx, len(toks))

    _step1_raw_text(scene, toks)
    _step2_bpe_split(scene, toks)
    _step3_integer_ids(scene, ids, toks)
    _add_arrow(scene, "testo → token → numero")


# --- Step builders --------------------------------------------------------


def _step1_raw_text(scene: Scene, toks: Sequence[str]) -> None:
    """Show the raw Italian sentence and the 'Testo grezzo' caption."""
    raw = Text("gatto", font_size=96, color=COLOR_TEXT)
    caption = Text("Testo grezzo", font_size=36, color=COLOR_CAPTION)
    caption.next_to(raw, DOWN, buff=0.5)
    scene.play(FadeIn(Group(raw, caption), run_time=0.5))
    raw.shift(UP * 1.5)
    caption.shift(UP * 1.5)


def _step2_bpe_split(scene: Scene, toks: Sequence[str]) -> None:
    """Display the BPE sub-tokens of the words that actually split."""
    split_idxs = find_split_indices(toks)
    if not split_idxs:
        chosen = list(toks[:3])
    else:
        chosen = [toks[i] for i in split_idxs]

    sub_texts = [Text(s, font_size=56, color=COLOR_TEXT) for s in chosen]
    arrange_horizontally(sub_texts, buff=0.3)
    caption = Text("Token (BPE)", font_size=36, color=COLOR_CAPTION)
    caption.next_to(sub_texts[0], DOWN, buff=0.5)
    scene.play(FadeIn(Group(*sub_texts, caption), run_time=0.5))


def _step3_integer_ids(scene: Scene, ids: Sequence[int], toks: Sequence[str]) -> None:
    """Show integer IDs of BPE sub-tokens (with leading context)."""
    split_idxs = find_split_indices(toks)
    if split_idxs:
        n_lead = max(0, 7 - len(split_idxs))
        show_idxs = sorted(set(list(range(min(n_lead, len(toks)))) + split_idxs))[:8]
        word_ids = [ids[i] for i in show_idxs if i < len(ids)]
    else:
        word_ids = list(ids[:7])

    id_texts = [MathTex(str(i), font_size=48, color=COLOR_TEXT) for i in word_ids]
    arrange_horizontally(id_texts, buff=0.25)
    caption = Text("Numero (token ID)", font_size=36, color=COLOR_CAPTION)
    caption.next_to(id_texts[0], DOWN, buff=0.5)
    # Wrap all elements in one FadeIn so every MathTex is exposed
    scene.play(FadeIn(Group(*id_texts, caption), run_time=0.5))


def _add_arrow(scene: Scene, label: str) -> None:
    """Add a labelled banner along the bottom edge."""
    arrow_text = Text(label, font_size=28, color=COLOR_ACCENT)
    arrow_text.to_edge(DOWN)
    scene.play(FadeIn(arrow_text, run_time=0.5))


# --- Re-exports so tests can mock by name ---------------------------------
__all__ = [
    "Tokenization",
    "COLOR_TEXT",
    "load_data",
    "pick_first_real_row",
    "find_split_indices",
    "build_tokenization",
    "Text",
    "MathTex",
    "FadeIn",
]
