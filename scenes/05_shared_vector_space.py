"""Scene 05 — shared vector space (3D) (issue #11).

3D scatter of PCA-reduced CLIP embeddings: 5 Italian text prompts (blue
dots) + 1 image (orange dot), all in the same 3D space. Animate the
arrival of each modality group, then a slow camera rotation to convey
3D-ness. Italian captions throughout.

Backed by ``data/pca_coords_3d.npy`` and ``data/pca_labels.json`` from
issue #6. No inference at render time — the scene only reads the saved
arrays.

Design notes:
- Uses ``ThreeDScene`` (Manim's 3D scene class).
- The 5 text labels are full Italian sentences (up to ~200 chars). We
  truncate to ``LABEL_MAX_LEN`` with an ellipsis to avoid 3D-plot
  label overlap.
- PCA coords are centred (±1) — multiplied by ``COORD_SCALE`` (2.0)
  so dots are visible at scene scale.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from manim import (
    DEGREES,
    Dot3D,
    FadeIn,
    Group,
    RIGHT,
    Text,
    ThreeDAxes,
    ThreeDScene,
    UP,
    VGroup,
)

# 3D direction constants: OUT (away from camera) used for the Z-axis label.
from manim.constants import OUT

from scenes._common import DEFAULT_DATA_DIR, add_italian_caption

LOGGER = logging.getLogger("glassbox.scene_05")

# --- Visual configuration --------------------------------------------------

# Multiplier on the centred PCA coords (range ~ ±1) for visibility.
COORD_SCALE = 2.0

# Maximum characters per label (long Italian sentences truncated).
LABEL_MAX_LEN = 24

# Camera orientation (3D spherical coords)
CAMERA_PHI_INIT = 75 * DEGREES
CAMERA_THETA_INIT = -45 * DEGREES
# Camera rotation target
CAMERA_THETA_END = 45 * DEGREES


# --- Pure data helpers (extracted for testability) -------------------------


def load_pca_coords(data_dir: Path = DEFAULT_DATA_DIR) -> np.ndarray:
    """Load ``pca_coords_3d.npy`` (N, 3) float32."""
    return np.load(data_dir / "pca_coords_3d.npy")


def load_pca_labels(data_dir: Path = DEFAULT_DATA_DIR) -> dict:
    """Load ``pca_labels.json`` (schema: {modality, label, colors})."""
    return json.loads((data_dir / "pca_labels.json").read_text())


def truncate_label(text: str, *, max_len: int = LABEL_MAX_LEN) -> str:
    """Truncate ``text`` to ``max_len`` chars, appending '…' if cut."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 1].rstrip() + "…"


def scale_pca_coords(coords: np.ndarray, *, factor: float = COORD_SCALE) -> np.ndarray:
    """Scale PCA coords by ``factor`` for visibility in Manim coords."""
    return (coords * factor).astype(np.float32)


def color_for_modality(modality: str, colors: dict) -> str:
    """Map a modality string ('text' | 'image') to its hex color."""
    return colors.get(modality, "#FFFFFF")


# --- The Scene ------------------------------------------------------------


class SharedVectorSpace(ThreeDScene):
    """Manim 3D scene: PCA-reduced CLIP shared space."""

    def construct(self) -> None:  # type: ignore[override]
        """Manim entry point — delegates to the free function for testability."""
        build_shared_vector_space(self, data_dir=DEFAULT_DATA_DIR)


# --- Orchestration (free function so tests can call with a fake scene) ---


def build_shared_vector_space(
    scene: ThreeDScene,
    *,
    data_dir: Path = DEFAULT_DATA_DIR,
) -> None:
    """Run the four animation steps against the given 3D scene."""
    coords = load_pca_coords(data_dir)
    labels = load_pca_labels(data_dir)
    LOGGER.info(
        "Scene 05: coords=%s, n_text=%d, n_image=%d",
        coords.shape,
        sum(1 for m in labels["modality"] if m == "text"),
        sum(1 for m in labels["modality"] if m == "image"),
    )

    _step1_axes(scene)
    _step2_text_dots(scene, coords, labels)
    _step3_image_dot(scene, coords, labels)
    _step4_camera_rotation(scene)
    add_italian_caption(scene, "Tutto nello stesso spazio")


# --- Step builders --------------------------------------------------------


def _step1_axes(scene: ThreeDScene) -> None:
    """3D axes with Italian labels: 'Dimensione 1/2/3'."""
    axes = ThreeDAxes(
        x_range=[-3, 3, 1],
        y_range=[-3, 3, 1],
        z_range=[-3, 3, 1],
        x_length=6,
        y_length=6,
        z_length=6,
    )
    # ThreeDAxes exposes .x_axis / .y_axis / .z_axis (Line3D mobjects);
    # place labels at the tip of each axis using get_end().
    x_label = Text("Dimensione 1", font_size=24).next_to(
        axes.x_axis.get_end(), RIGHT, buff=0.1
    )
    y_label = Text("Dimensione 2", font_size=24).next_to(
        axes.y_axis.get_end(), UP, buff=0.1
    )
    z_label = Text("Dimensione 3", font_size=24).next_to(
        axes.z_axis.get_end(), OUT, buff=0.1
    )
    scene.play(FadeIn(Group(axes, x_label, y_label, z_label), run_time=0.5))
    # Set the initial camera orientation
    scene.set_camera_orientation(phi=CAMERA_PHI_INIT, theta=CAMERA_THETA_INIT)


def _step2_text_dots(scene: ThreeDScene, coords: np.ndarray, labels: dict) -> None:
    """5 text dots, blue, animated one by one with labels."""
    colors = labels["colors"]
    text_idxs = [i for i, m in enumerate(labels["modality"]) if m == "text"]
    if not text_idxs:
        return
    scaled = scale_pca_coords(coords)
    dots_group: list[VGroup] = []
    for i in text_idxs:
        x, y, z = scaled[i].tolist()
        dot = Dot3D(
            point=[x, y, z], color=color_for_modality("text", colors), radius=0.08
        )
        label_text = Text(
            truncate_label(labels["label"][i]), font_size=18, color="#FFFFFF"
        )
        label_text.next_to(dot, RIGHT, buff=0.1)
        group = VGroup(dot, label_text)
        scene.play(FadeIn(group, run_time=0.3))
        dots_group.append(group)


def _step3_image_dot(scene: ThreeDScene, coords: np.ndarray, labels: dict) -> None:
    """1 image dot, orange, with a label."""
    colors = labels["colors"]
    img_idxs = [i for i, m in enumerate(labels["modality"]) if m == "image"]
    if not img_idxs:
        return
    scaled = scale_pca_coords(coords)
    for i in img_idxs:
        x, y, z = scaled[i].tolist()
        dot = Dot3D(
            point=[x, y, z], color=color_for_modality("image", colors), radius=0.1
        )
        label_text = Text(
            truncate_label(labels["label"][i]), font_size=18, color="#FFFFFF"
        )
        label_text.next_to(dot, RIGHT, buff=0.1)
        scene.play(FadeIn(VGroup(dot, label_text), run_time=0.3))


def _step4_camera_rotation(scene: ThreeDScene) -> None:
    """Slow camera rotation to convey 3D-ness."""
    # move_camera with a small theta increment
    scene.move_camera(
        phi=CAMERA_PHI_INIT,
        theta=CAMERA_THETA_END,
        run_time=2.0,  # type: ignore[arg-type]
    )


# --- Re-exports so tests can mock by name ---------------------------------
__all__ = [
    "SharedVectorSpace",
    "load_pca_coords",
    "load_pca_labels",
    "truncate_label",
    "scale_pca_coords",
    "color_for_modality",
    "build_shared_vector_space",
    "Text",
    "ThreeDAxes",
    "Dot3D",
    "FadeIn",
    "Group",
    "VGroup",
]
