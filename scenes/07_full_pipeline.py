"""Scene 07 — full pipeline (issue #12, the capstone).

End-to-end recap: photo + caption → tokenize/patch → embed → shared
space → model output. Italian captions throughout, modality colours.

Reuses the data from issues #3, #4, #6 — no new extraction. The 3D
scatter from scene 05 is flattened to 2D here (pca_coords_3d[:, :2])
to keep the recap flat and 2D.

Per the reviewer's bug fix on issue #6: only 6 points (5 text + 1
image) are shown — the visualisation is intentionally sparse.

Per the reviewer's bookkeeping note: this issue is numbered #12 in
state.json but is scene 07. Implementation order is
#11 (shared space) → #13 (two-path) → #12 (this, the capstone). The
narrative climax is at the end.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np
from manim import (
    DOWN,
    RIGHT,
    UP,
    Arrow,
    FadeIn,
    Group,
    ImageMobject,
    Rectangle,
    Scene,
    Text,
    VGroup,
)

LOGGER = logging.getLogger("glassbox.scene_07")

# --- Modality colour map (consistent across all scenes) -------------------

COLOR_TEXT = "#3B82F6"  # blue — text
COLOR_IMAGE = "#F97316"  # orange — image
COLOR_AUDIO = "#10B981"  # green — audio
COLOR_CAPTION = "#FFFFFF"  # white captions
COLOR_ACCENT = "#FBBF24"  # yellow for callouts

# --- Data paths (overridable for tests) ------------------------------------

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# --- Visual configuration --------------------------------------------------

# The opening caption is hardcoded so the scene is self-contained.
OPENING_CAPTION = "Un gatto sul divano"
MODEL_OUTPUT_TEXT = "Un gatto sul divano"
FINAL_CAPTION = "Capisce tutto insieme"

# Token sequence to display (use the first row of tokens.npy)
ROW_IDX = 0

# Number of mel-style patches to show in the recap (small for visual)
PATCH_RECAP_COUNT = 4  # 2x2 grid
TOKEN_RECAP_COUNT = 6  # 6 BPE sub-tokens

# Box dimensions for the "shared space" step
SHARED_BOX_WIDTH = 4.0
SHARED_BOX_HEIGHT = 2.5


# --- Pure data helpers (extracted for testability) -------------------------


def load_image(data_dir: Path = DEFAULT_DATA_DIR) -> np.ndarray:
    """Load the displayable image (uint8 HWC 224×224×3)."""
    return np.load(data_dir / "sample_image_224.npy")


def load_tokens(data_dir: Path = DEFAULT_DATA_DIR) -> np.ndarray:
    """Load the (N, 77) int64 token IDs."""
    return np.load(data_dir / "tokens.npy")


def load_pca_2d(data_dir: Path = DEFAULT_DATA_DIR) -> np.ndarray:
    """Load the (N, 3) PCA coords and slice to (N, 2) for the 2D recap."""
    pca_3d = np.load(data_dir / "pca_coords_3d.npy")
    return pca_3d[:, :2].astype(np.float32)


def pick_first_content_row(
    data_dir: Path = DEFAULT_DATA_DIR, *, row_idx: int = ROW_IDX
) -> np.ndarray:
    """Return the first row of tokens.npy, stripped of CLIP specials.

    Reviewer's bug fix (from issue #3): CLIP special tokens are 49406
    (start) and 49407 (end/pad). They are removed so the visual token
    row shows only content IDs.
    """
    tokens = load_tokens(data_dir)
    row = tokens[row_idx]
    return np.asarray(
        [int(t) for t in row if int(t) not in (49406, 49407)],
        dtype=np.int64,
    )


# --- The Scene ------------------------------------------------------------


class FullPipeline(Scene):
    """Manim scene: end-to-end recap of the multimodal pipeline."""

    def construct(self) -> None:  # type: ignore[override]
        """Manim entry point — delegates to the free function for testability."""
        build_full_pipeline(self, data_dir=DEFAULT_DATA_DIR)


# --- Orchestration (free function so tests can call with a fake scene) ---


def build_full_pipeline(
    scene: Scene,
    *,
    data_dir: Path = DEFAULT_DATA_DIR,
) -> None:
    """Run the four-step recap against the given scene."""
    image = load_image(data_dir)
    token_ids = pick_first_content_row(data_dir)
    pca_2d = load_pca_2d(data_dir)
    LOGGER.info(
        "Scene 07: image=%s, tokens=%d, pca_2d=%s",
        image.shape,
        len(token_ids),
        pca_2d.shape,
    )

    _step1_photo_and_caption(scene, image)
    _step2_morph_to_patches_and_tokens(scene, image, [int(t) for t in token_ids])
    _step3_shared_space_box(scene, pca_2d)
    _step4_model_output(scene)
    _add_final_caption(scene, FINAL_CAPTION)


# --- Step builders --------------------------------------------------------


def _step1_photo_and_caption(scene: Scene, image: np.ndarray) -> None:
    """Photo on the left, Italian caption on the right."""
    img_mob = ImageMobject(image)
    img_mob.height = 3.5
    img_mob.shift(np.array([-3.5, 0.5, 0.0]))
    caption = Text(OPENING_CAPTION, font_size=42, color=COLOR_TEXT)
    caption.shift(np.array([3.0, 0.5, 0.0]))
    arrow = Arrow(
        img_mob.get_right(),
        caption.get_left(),
        buff=0.2,
        color=COLOR_ACCENT,
        stroke_width=4,
    )
    arrow_label = Text("Caption", font_size=20, color=COLOR_ACCENT)
    arrow_label.next_to(arrow, UP, buff=0.1)
    scene.play(FadeIn(Group(img_mob, caption, arrow, arrow_label), run_time=0.5))


def _step2_morph_to_patches_and_tokens(
    scene: Scene, image: np.ndarray, token_ids: Sequence[int]
) -> None:
    """Photo becomes a 2x2 patch grid; caption becomes a BPE token row."""
    # Replace the photo with a 2x2 grid of small squares
    patch_squares: list[Rectangle] = []
    cell = 0.5
    for r in range(2):
        for c in range(2):
            sq = Rectangle(
                width=cell,
                height=cell,
                fill_color=COLOR_IMAGE,
                fill_opacity=0.8,
                stroke_width=1,
                stroke_color="#FFFFFF",
            )
            sq.move_to(np.array([-4.0 + c * cell, 0.5 + r * cell, 0.0]))
            patch_squares.append(sq)
    patches = VGroup(*patch_squares)
    arrow_patch = Arrow(
        np.array([-3.0, 0.5, 0.0]),
        np.array([-0.5, 0.5, 0.0]),
        buff=0.2,
        color=COLOR_ACCENT,
        stroke_width=4,
    )
    label_patch = Text("Patch", font_size=20, color=COLOR_ACCENT)
    label_patch.next_to(arrow_patch, UP, buff=0.1)

    # Replace the caption with a row of token IDs
    token_texes = [
        Text(str(tid), font_size=28, color=COLOR_TEXT)
        for tid in token_ids[:TOKEN_RECAP_COUNT]
    ]
    for i in range(1, len(token_texes)):
        token_texes[i].next_to(token_texes[i - 1], RIGHT, buff=0.15)
    tokens_group = VGroup(*token_texes)
    tokens_group.move_to(np.array([2.5, 0.5, 0]))
    arrow_token = Arrow(
        np.array([0.0, 0.5, 0.0]),
        np.array([1.5, 0.5, 0.0]),
        buff=0.2,
        color=COLOR_ACCENT,
        stroke_width=4,
    )
    label_token = Text("Token", font_size=20, color=COLOR_ACCENT)
    label_token.next_to(arrow_token, UP, buff=0.1)

    scene.play(
        FadeIn(
            Group(
                patches,
                tokens_group,
                arrow_patch,
                label_patch,
                arrow_token,
                label_token,
            ),
            run_time=0.5,
        )
    )


def _step3_shared_space_box(scene: Scene, pca_2d: np.ndarray) -> None:
    """Both streams flow into a 2D 'shared space' box (PCA scatter)."""
    # The shared space box (a rectangle framing the dot scatter)
    box = Rectangle(
        width=SHARED_BOX_WIDTH,
        height=SHARED_BOX_HEIGHT,
        stroke_width=3,
        stroke_color="#FFFFFF",
        fill_opacity=0,
    )
    box.shift(DOWN * 1.0)

    # Dots from the 2D PCA scatter
    from manim import Dot

    text_idxs = list(range(5))  # first 5 are text
    x_scale, y_scale = 0.8, 0.5
    dots: list[Dot] = []
    for i, (x, y) in enumerate(pca_2d):
        color = COLOR_TEXT if i in text_idxs else COLOR_IMAGE
        radius = 0.08 if i in text_idxs else 0.12
        dot = Dot(
            point=np.array([x * x_scale, y * y_scale - 1.0, 0]),
            color=color,
            radius=radius,
            fill_opacity=0.85,
        )
        dots.append(dot)
    dot_group = VGroup(*dots)

    # Arrows from the patches+tokens above into the box
    arrow_left = Arrow(
        np.array([-3.0, 0.5, 0.0]),
        np.array([-2.0, -1.0, 0.0]),
        buff=0.2,
        color=COLOR_ACCENT,
        stroke_width=4,
    )
    arrow_right = Arrow(
        np.array([1.5, 0.5, 0.0]),
        np.array([2.0, -1.0, 0.0]),
        buff=0.2,
        color=COLOR_ACCENT,
        stroke_width=4,
    )
    label_box = Text("Spazio condiviso", font_size=24, color=COLOR_CAPTION)
    label_box.next_to(box, UP, buff=0.2)

    scene.play(
        FadeIn(Group(box, dot_group, arrow_left, arrow_right, label_box), run_time=0.5)
    )


def _step4_model_output(scene: Scene) -> None:
    """A 'Modello' rectangle reads the unified stream; output text appears."""
    # The model rectangle below the shared space
    model_box = Rectangle(
        width=3.0,
        height=1.0,
        fill_color="#1F2937",
        fill_opacity=0.9,
        stroke_width=2,
        stroke_color="#FFFFFF",
    )
    model_box.shift(DOWN * 3.5)
    model_label = Text("Modello", font_size=28, color=COLOR_CAPTION)
    model_label.move_to(model_box.get_center())

    # Arrow from shared space to model
    arrow_to_model = Arrow(
        np.array([0.0, -2.3, 0.0]),
        np.array([0.0, -3.0, 0.0]),
        buff=0.1,
        color=COLOR_ACCENT,
        stroke_width=4,
    )

    # Output text (the model "reads" the stream and produces this)
    output_text = Text(MODEL_OUTPUT_TEXT, font_size=36, color=COLOR_TEXT)
    output_text.next_to(model_box, DOWN, buff=0.5)

    arrow_response = Arrow(
        np.array([0.0, -3.5, 0.0]),
        np.array([0.0, -3.0, 0.0]),
        buff=0.1,
        color=COLOR_ACCENT,
        stroke_width=4,
    )
    label_response = Text("Risposta", font_size=20, color=COLOR_ACCENT)
    label_response.next_to(arrow_response, RIGHT, buff=0.1)

    scene.play(
        FadeIn(
            Group(
                model_box,
                model_label,
                arrow_to_model,
                output_text,
                arrow_response,
                label_response,
            ),
            run_time=0.5,
        )
    )


def _add_final_caption(scene: Scene, text: str) -> None:
    """Final caption at the bottom edge: 'Capisce tutto insieme'."""
    caption = Text(text, font_size=32, color=COLOR_CAPTION)
    caption.to_edge(DOWN)
    scene.play(FadeIn(caption, run_time=0.5))


# --- Re-exports so tests can mock by name ---------------------------------
__all__ = [
    "FullPipeline",
    "COLOR_TEXT",
    "COLOR_IMAGE",
    "COLOR_AUDIO",
    "load_image",
    "load_tokens",
    "load_pca_2d",
    "pick_first_content_row",
    "build_full_pipeline",
    "Text",
    "ImageMobject",
    "Rectangle",
    "Arrow",
    "FadeIn",
    "Group",
    "VGroup",
]
