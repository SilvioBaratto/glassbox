"""Scene 06 — two-path comparison (issue #13, the rhetorical pivot).

Side-by-side comparison: "Traduttore separato" (image → bottleneck →
LLM, lossy) vs "Spazio condiviso" (native multimodal, direct). A red
warning sign marks the lossy path.

Reuses ``data/shared_text_embeds.npy``, ``data/shared_image_embeds.npy``,
and ``data/pca_coords_3d.npy`` from issue #6 — no new extraction.

Per the AC, the right panel can reuse the 3D scatter from scene 05 OR
a 2D simplification. We use the 2D simplification
(``pca_coords_3d[:, :2]``) to keep the recap flat.

Per state.json: this issue is numbered #13 but the scene is 06 in the
file. Implementation order: #11 (shared space) → #13 (this, scene 06) →
#12 (scene 07 capstone).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from manim import (
    Arrow,
    Cross,
    FadeIn,
    Group,
    ImageMobject,
    Line,
    Rectangle,
    Scene,
    Text,
    VGroup,
)

# Dot is in manim top-level
from manim import Dot

LOGGER = logging.getLogger("glassbox.scene_06")

# --- Modality colour map (consistent across all scenes) -------------------

COLOR_TEXT = "#3B82F6"  # blue — text
COLOR_IMAGE = "#F97316"  # orange — image
COLOR_WARNING = "#EF4444"  # red — bottleneck warning
COLOR_CAPTION = "#FFFFFF"  # white captions
COLOR_ACCENT = "#FBBF24"  # yellow for callouts
COLOR_BOTTLENECK = "#374151"  # dark slate — narrow pipe

# --- Data paths (overridable for tests) ------------------------------------

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# --- Visual configuration --------------------------------------------------

# Two halves of the screen (Manim units, frame is 14 wide × 8 tall)
PANEL_LEFT_X = -3.5
PANEL_RIGHT_X = 3.5
PANEL_Y = 0.5

# Bottleneck dimensions (the "narrow pipe")
BOTTLENECK_WIDTH = 0.3
BOTTLENECK_HEIGHT = 2.0

# PCA scaling for the right-panel scatter
PCA_X_SCALE = 0.8
PCA_Y_SCALE = 0.5


# --- Pure data helpers (extracted for testability) -------------------------


def load_pca_2d(data_dir: Path = DEFAULT_DATA_DIR) -> np.ndarray:
    """Load ``pca_coords_3d.npy`` and slice to (N, 2)."""
    pca_3d = np.load(data_dir / "pca_coords_3d.npy")
    return pca_3d[:, :2].astype(np.float32)


def load_shared_text_embeds(data_dir: Path = DEFAULT_DATA_DIR) -> np.ndarray:
    """Load ``shared_text_embeds.npy`` (N_text, 512)."""
    return np.load(data_dir / "shared_text_embeds.npy")


def load_shared_image_embeds(data_dir: Path = DEFAULT_DATA_DIR) -> np.ndarray:
    """Load ``shared_image_embeds.npy`` (1, 512)."""
    return np.load(data_dir / "shared_image_embeds.npy")


def pca_to_2d_dots(
    coords: np.ndarray, *, x_scale: float = PCA_X_SCALE, y_scale: float = PCA_Y_SCALE
) -> np.ndarray:
    """Return (N, 2) array of (x, y) dot positions from PCA coords."""
    out = np.zeros((coords.shape[0], 2), dtype=np.float32)
    out[:, 0] = coords[:, 0] * x_scale
    out[:, 1] = coords[:, 1] * y_scale
    return out


def panel_split_offsets() -> tuple[float, float]:
    """Return (left_x, right_x) for the two-panel split."""
    return (PANEL_LEFT_X, PANEL_RIGHT_X)


# --- The Scene ------------------------------------------------------------


class TwoPathComparison(Scene):
    """Manim scene: side-by-side translator vs shared-space comparison."""

    def construct(self) -> None:  # type: ignore[override]
        """Manim entry point — delegates to the free function for testability."""
        build_two_path_comparison(self, data_dir=DEFAULT_DATA_DIR)


# --- Orchestration (free function so tests can call with a fake scene) ---


def build_two_path_comparison(
    scene: Scene,
    *,
    data_dir: Path = DEFAULT_DATA_DIR,
) -> None:
    """Build the two-panel comparison against the given scene."""
    pca_2d = load_pca_2d(data_dir)
    LOGGER.info("Scene 06: pca_2d shape=%s", pca_2d.shape)

    _step_left_panel(scene)
    _step_right_panel(scene, pca_2d)
    _step_warning_overlay(scene)
    _add_captions(scene)


# --- Step builders --------------------------------------------------------


def _step_left_panel(scene: Scene) -> None:
    """Left panel: 'Traduttore separato' — image → bottleneck → LLM.

    The image is represented by a small ImageMobject (we don't have
    ImageMobject in tests, but for the recap we use a Rectangle
    placeholder when no image is available).
    """
    # Panel title
    left_title = Text("Traduttore separato", font_size=36, color=COLOR_TEXT)
    left_title.move_to(np.array([PANEL_LEFT_X, PANEL_Y + 1.8, 0.0]))

    # Image proxy (orange square representing the input image)
    img_proxy = Rectangle(
        width=1.2,
        height=1.2,
        fill_color=COLOR_IMAGE,
        fill_opacity=0.85,
        stroke_width=1,
        stroke_color="#FFFFFF",
    )
    img_proxy.move_to(np.array([PANEL_LEFT_X - 2.0, PANEL_Y, 0.0]))

    # Arrow from image to bottleneck
    arrow1 = Arrow(
        np.array([PANEL_LEFT_X - 1.3, PANEL_Y, 0.0]),
        np.array([PANEL_LEFT_X - 0.5, PANEL_Y, 0.0]),
        buff=0.0,
        color=COLOR_ACCENT,
        stroke_width=4,
    )

    # Bottleneck (narrow pipe)
    bottleneck = Rectangle(
        width=BOTTLENECK_WIDTH,
        height=BOTTLENECK_HEIGHT,
        fill_color=COLOR_BOTTLENECK,
        fill_opacity=0.9,
        stroke_width=2,
        stroke_color="#FFFFFF",
    )
    bottleneck.move_to(np.array([PANEL_LEFT_X, PANEL_Y, 0.0]))

    # Arrow from bottleneck to LLM
    arrow2 = Arrow(
        np.array([PANEL_LEFT_X + 0.2, PANEL_Y, 0.0]),
        np.array([PANEL_LEFT_X + 1.0, PANEL_Y, 0.0]),
        buff=0.0,
        color=COLOR_ACCENT,
        stroke_width=4,
    )

    # LLM box (faded because of lossy compression)
    llm_box = Rectangle(
        width=1.5,
        height=1.0,
        fill_color=COLOR_BOTTLENECK,
        fill_opacity=0.7,
        stroke_width=2,
        stroke_color="#FFFFFF",
    )
    llm_box.move_to(np.array([PANEL_LEFT_X + 1.8, PANEL_Y, 0.0]))
    llm_label = Text("LLM", font_size=24, color=COLOR_CAPTION)
    llm_label.move_to(np.array([PANEL_LEFT_X + 1.8, PANEL_Y, 0.0]))

    panel = Group(left_title, img_proxy, arrow1, bottleneck, arrow2, llm_box, llm_label)
    scene.play(FadeIn(panel, run_time=0.7))


def _step_right_panel(scene: Scene, pca_2d: np.ndarray) -> None:
    """Right panel: 'Spazio condiviso' — image + text both feed directly."""
    right_title = Text("Spazio condiviso", font_size=36, color=COLOR_TEXT)
    right_title.move_to(np.array([PANEL_RIGHT_X, PANEL_Y + 1.8, 0.0]))

    # Image and text both feed the same box (no translator)
    img_proxy = Rectangle(
        width=1.0,
        height=1.0,
        fill_color=COLOR_IMAGE,
        fill_opacity=0.85,
        stroke_width=1,
        stroke_color="#FFFFFF",
    )
    img_proxy.move_to(np.array([PANEL_RIGHT_X - 1.5, PANEL_Y + 0.6, 0.0]))
    img_label = Text("img", font_size=18, color="#FFFFFF")
    img_label.move_to(np.array([PANEL_RIGHT_X - 1.5, PANEL_Y + 0.6, 0.0]))

    text_proxy = Rectangle(
        width=1.0,
        height=1.0,
        fill_color=COLOR_TEXT,
        fill_opacity=0.85,
        stroke_width=1,
        stroke_color="#FFFFFF",
    )
    text_proxy.move_to(np.array([PANEL_RIGHT_X - 1.5, PANEL_Y - 0.6, 0.0]))
    text_label = Text("txt", font_size=18, color="#FFFFFF")
    text_label.move_to(np.array([PANEL_RIGHT_X - 1.5, PANEL_Y - 0.6, 0.0]))

    # Arrows to the shared space
    arrow_img = Arrow(
        np.array([PANEL_RIGHT_X - 1.0, PANEL_Y + 0.6, 0.0]),
        np.array([PANEL_RIGHT_X + 0.3, PANEL_Y, 0.0]),
        buff=0.1,
        color=COLOR_ACCENT,
        stroke_width=4,
    )
    arrow_txt = Arrow(
        np.array([PANEL_RIGHT_X - 1.0, PANEL_Y - 0.6, 0.0]),
        np.array([PANEL_RIGHT_X + 0.3, PANEL_Y, 0.0]),
        buff=0.1,
        color=COLOR_ACCENT,
        stroke_width=4,
    )

    # The shared space — 6 dots from the 2D PCA
    dot_positions = pca_to_2d_dots(pca_2d)
    text_idxs = list(range(5))
    dots: list[Dot] = []
    for i, (x, y) in enumerate(dot_positions):
        color = COLOR_TEXT if i in text_idxs else COLOR_IMAGE
        radius = 0.08 if i in text_idxs else 0.12
        dot = Dot(
            point=np.array([PANEL_RIGHT_X + 1.0 + x, y, 0.0]),
            color=color,
            radius=radius,
            fill_opacity=0.85,
        )
        dots.append(dot)
    dot_group = VGroup(*dots)

    panel = Group(
        right_title,
        img_proxy,
        img_label,
        text_proxy,
        text_label,
        arrow_img,
        arrow_txt,
        dot_group,
    )
    scene.play(FadeIn(panel, run_time=0.7))


def _step_warning_overlay(scene: Scene) -> None:
    """Red Cross warning over the left panel's bottleneck."""
    # Manim 0.18 Cross() takes stroke_color directly; move it to position.
    cross = Cross(stroke_color=COLOR_WARNING, stroke_width=6)
    cross.move_to(np.array([PANEL_LEFT_X, PANEL_Y, 0.0]))
    scene.play(FadeIn(cross, run_time=0.4))


def _add_captions(scene: Scene) -> None:
    """Italian captions under each panel."""
    left_caption = Text("Si perde sempre qualcosa", font_size=24, color=COLOR_WARNING)
    left_caption.move_to(np.array([PANEL_LEFT_X, PANEL_Y - 1.7, 0.0]))
    right_caption = Text("Niente traduttore", font_size=24, color=COLOR_TEXT)
    right_caption.move_to(np.array([PANEL_RIGHT_X, PANEL_Y - 1.7, 0.0]))
    scene.play(FadeIn(Group(left_caption, right_caption), run_time=0.5))


# --- Re-exports so tests can mock by name ---------------------------------
__all__ = [
    "TwoPathComparison",
    "COLOR_TEXT",
    "COLOR_IMAGE",
    "COLOR_WARNING",
    "load_pca_2d",
    "load_shared_text_embeds",
    "load_shared_image_embeds",
    "pca_to_2d_dots",
    "panel_split_offsets",
    "build_two_path_comparison",
    "Text",
    "Rectangle",
    "ImageMobject",
    "Arrow",
    "Line",
    "Cross",
    "FadeIn",
    "Group",
    "VGroup",
]
