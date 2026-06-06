"""Scene 03 — image patch splitting (issue #9).

Visualises: real photo → 14×14 grid of patches → each patch → vector
point. Italian captions throughout, orange modality colour.

Backed by ``data/sample_image_224.npy`` (uint8 HWC displayable image),
``data/patch_embeddings.npy`` (196, 768), and ``data/patch_grid.npy``
(14, 14) from issue #4. No inference at render time.

Reviewer's bug fixes applied:
- Step 1 reads ``sample_image_224.npy`` (uint8 HWC), NOT the
  post-normalisation float tensor — that would render as black.
- Step 2 animates the slicing 2 rows at a time → 7 beats instead of
  14, fitting the AC's 8s budget at 30 fps.
- We use a single ``ImageMobject`` for the base image and overlay
  ``Rectangle`` masks (one per row, 14 total) — NOT 196 individual
  Rectangle mobjects — for fast render.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from manim import (
    LEFT,
    RIGHT,
    FadeIn,
    Group,
    ImageMobject,
    Rectangle,
    Scene,
    Text,
    VGroup,
)

from scenes._common import DEFAULT_DATA_DIR, add_italian_caption

LOGGER = logging.getLogger("glassbox.scene_03")

# --- Modality colour map (matches requirements.md) -------------------------

COLOR_IMAGE = "#F97316"  # orange — image modality
COLOR_ACCENT = "#FBBF24"  # yellow for callouts
COLOR_GRID = "#1F2937"  # dark slate — grid line color

N_PATCHES_PER_SIDE = 14
N_PATCHES_TOTAL = 14 * 14  # 196
EMBED_DIM = 768
BAR_DIMS = 32  # first 32 dims of each patch embedding (simplified visual)

# Reviewer's bug fix: 2 rows at a time → 7 beats instead of 14.
SLICING_ROWS_PER_BEAT = 2
SLICING_BEATS = N_PATCHES_PER_SIDE // SLICING_ROWS_PER_BEAT  # 7


# --- Pure data helpers (extracted for testability) -------------------------


def load_image_for_display(data_dir: Path = DEFAULT_DATA_DIR) -> np.ndarray:
    """Load ``sample_image_224.npy`` (uint8 HWC 224×224×3) for display.

    Reviewer's bug fix: the post-normalisation CHW float tensor would
    render as black in Manim. We use the resized-but-not-normalised
    uint8 image that issue #4 saves alongside.
    """
    return np.load(data_dir / "sample_image_224.npy")


def load_patch_embeddings(data_dir: Path = DEFAULT_DATA_DIR) -> np.ndarray:
    """Load ``patch_embeddings.npy`` (196, 768) float32."""
    return np.load(data_dir / "patch_embeddings.npy")


def load_patch_grid(data_dir: Path = DEFAULT_DATA_DIR) -> np.ndarray:
    """Load ``patch_grid.npy`` (14, 14) int64."""
    return np.load(data_dir / "patch_grid.npy")


def normalise_first_32_dims(vec: np.ndarray) -> np.ndarray:
    """Normalise the first 32 dims of a single patch embedding to [0, 1].

    Min-max scaling: smallest non-zero value → 0, largest → 1. Used by
    the bar-chart snippet visualisation in step 3.
    """
    head = vec[:BAR_DIMS].astype(np.float32)
    mn, mx = float(head.min()), float(head.max())
    if mx - mn < 1e-9:
        return np.zeros(BAR_DIMS, dtype=np.float32)
    return ((head - mn) / (mx - mn)).astype(np.float32)


def build_dot_grid_layout(
    cols: int = N_PATCHES_PER_SIDE,
    rows: int = N_PATCHES_PER_SIDE,
    cell: float = 0.18,
) -> list[tuple[float, float]]:
    """Return (x, y) positions for a cols×rows dot grid centred at origin.

    Used by step 4 to lay out the 196 patch-embedding dots in a
    14×14 grid on the right side of the frame.
    """
    out: list[tuple[float, float]] = []
    for r in range(rows):
        for c in range(cols):
            x = (c - (cols - 1) / 2) * cell
            y = ((rows - 1) / 2 - r) * cell
            out.append((x, y))
    return out


# --- The Scene ------------------------------------------------------------


class ImagePatches(Scene):
    """Manim scene: real photo → 14×14 patches → vector dots."""

    def construct(self) -> None:  # type: ignore[override]
        """Manim entry point — delegates to the free function for testability."""
        build_image_patches(self, data_dir=DEFAULT_DATA_DIR)


# --- Orchestration (free function so tests can call with a fake scene) ---


def build_image_patches(
    scene: Scene,
    *,
    data_dir: Path = DEFAULT_DATA_DIR,
) -> None:
    """Run the four animation steps against the given scene."""
    image = load_image_for_display(data_dir)
    patches = load_patch_embeddings(data_dir)

    _step1_show_full_image(scene, image)
    _step2_slice_into_grid(scene, image)
    _step3_vector_bar_snippet(scene, patches)
    _step4_dot_grid(scene, patches)
    add_italian_caption(scene, "Ogni quadrato diventa un vettore")


# --- Step builders --------------------------------------------------------


def _step1_show_full_image(scene: Scene, image: np.ndarray) -> None:
    """Show the original sample image full-frame.

    Reviewer's bug fix: use ``sample_image_224.npy`` (uint8 HWC), not
    the post-normalisation float tensor.
    """
    img_mob = ImageMobject(image)
    img_mob.height = 5.0  # scale to fit 1080p height
    scene.play(FadeIn(img_mob, run_time=0.5))
    img_mob.shift(LEFT * 3.0)  # move left to make room for step 4 grid


def _step2_slice_into_grid(scene: Scene, image: np.ndarray) -> None:
    """Animate the 14×14 grid slicing, 2 rows at a time (reviewer's fix).

    Strategy: a single ImageMobject for the base + 14 Rectangle overlays
    (one per row). Animate the rectangles in groups of 2.
    """
    # The base image is already in the scene from step 1; we add 14
    # transparent row-overlays that fade in 2 at a time.
    row_rects: list[Rectangle] = []
    cell_h = 5.0 / N_PATCHES_PER_SIDE
    base_y = -2.5  # bottom of the image
    for r in range(N_PATCHES_PER_SIDE):
        rect = Rectangle(
            width=5.0,
            height=cell_h,
            stroke_width=2,
            stroke_color=COLOR_IMAGE,
            fill_color=COLOR_IMAGE,
            fill_opacity=0.25,
        )
        rect.move_to(np.array([0, base_y + (r + 0.5) * cell_h, 0]))
        rect.shift(LEFT * 3.0)
        row_rects.append(rect)

    # Animate in groups of 2 rows
    for beat in range(SLICING_BEATS):
        i = beat * SLICING_ROWS_PER_BEAT
        group = Group(*row_rects[i : i + SLICING_ROWS_PER_BEAT])
        scene.play(FadeIn(group, run_time=0.4))


def _step3_vector_bar_snippet(scene: Scene, patches: np.ndarray) -> None:
    """Show a 32-dim bar chart snippet of the first patch's embedding.

    Visualisation: 32 thin vertical bars whose heights correspond to
    the normalised first 32 dims of patch 0. Sits in the centre of the
    frame.
    """
    normed = normalise_first_32_dims(patches[0])
    bar_w = 0.06
    bars: list[Rectangle] = []
    for i, h in enumerate(normed):
        bar_h = max(0.05, float(h) * 1.2)
        bar = Rectangle(
            width=bar_w,
            height=bar_h,
            fill_color=COLOR_IMAGE,
            fill_opacity=0.9,
            stroke_width=0,
        )
        bar.move_to(np.array([(i - 16) * bar_w, bar_h / 2, 0]))
        bar.shift(RIGHT * 3.0)  # right side of the frame
        bars.append(bar)
    group = VGroup(*bars)
    scene.play(FadeIn(group, run_time=0.5))


def _step4_dot_grid(scene: Scene, patches: np.ndarray) -> None:
    """Place 196 dots in a 14×14 grid on the right side.

    Color: uniform orange tint (per AC, "color-coded by patch position
    OR a uniform orange tint"). We use uniform orange for simplicity.
    """
    from manim import Dot

    positions = build_dot_grid_layout()
    dots: list[Dot] = []
    for i, (x, y) in enumerate(positions):
        d = Dot(radius=0.04, color=COLOR_IMAGE, fill_opacity=0.85)
        d.move_to(np.array([x + 3.0, y, 0]))  # shift to the right side
        dots.append(d)
    grid = VGroup(*dots)
    scene.play(FadeIn(grid, run_time=0.5))


# --- Re-exports so tests can mock by name ---------------------------------
__all__ = [
    "ImagePatches",
    "COLOR_IMAGE",
    "load_image_for_display",
    "load_patch_embeddings",
    "load_patch_grid",
    "normalise_first_32_dims",
    "build_dot_grid_layout",
    "build_image_patches",
    "Text",
    "ImageMobject",
    "Rectangle",
    "FadeIn",
    "Group",
    "VGroup",
]
