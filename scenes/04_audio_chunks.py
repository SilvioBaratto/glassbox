"""Scene 04 — audio chunking (issue #10).

Visualises: raw waveform → log-mel spectrogram → Conv1d compression to
encoder positions. Italian captions throughout, green modality colour.

Backed by ``data/audio_waveform.npy``, ``data/audio_frames.npy`` (80, T)
and ``data/audio_encoder.npy`` (T/2, 512) from issue #5. No inference at
render time — the scene only reads the saved numpy arrays.

Design notes (reviewer-bug-fix-aware from issues #3 and #5):
- We use the *actual* T from the data, NOT the Whisper-padded 3000.
  For our 5-second sample T ≈ 501, so step 3's "3000 frame markers"
  is replaced with T markers + stride-2 brackets, and step 4 shows
  the 1500 (= T/2) encoder positions that the Whisper Conv1d stem
  actually produces from the padded input.
- The mel grid (80 × T) is downsampled in time to ≤ 50 columns for
  visual clarity; drawing 40,080 squares would be slow.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from manim import (
    UP,
    Axes,
    Brace,
    Dot,
    FadeIn,
    Group,
    Line,
    Rectangle,
    Scene,
    Text,
    VGroup,
)

from scenes._common import COLOR_CAPTION, DEFAULT_DATA_DIR, add_italian_caption

LOGGER = logging.getLogger("glassbox.scene_04")

# --- Modality colour map (matches requirements.md) -------------------------

COLOR_AUDIO = "#10B981"  # green — audio modality
COLOR_ACCENT = "#FBBF24"  # yellow for callouts

# --- Visual configuration --------------------------------------------------

# Issue #5's reviewer fix: the Whisper feature extractor pads to 3000,
# but our display uses the *actual* T from `audio_frames.npy`.
# Step 3 shows T frame markers with stride-2 brackets.
# Step 4 shows T/2 encoder positions (= 1500 for a 3000-frame padded
# input, OR the actual T/2 for our real data).

# Downsample factor for the waveform line plot (80000 samples → 800).
WAVEFORM_DOWNSAMPLE_FACTOR = 100
# Maximum number of mel grid columns (downsample T if necessary).
MEL_MAX_COLS = 50
# Maximum number of Conv1d brackets to draw (stride-2 indicator).
BRACE_MAX_COUNT = 10
# Maximum number of encoder dots in the collapsed view.
ENCODER_DOT_MAX = 60


# --- Pure data helpers (extracted for testability) -------------------------


def load_waveform(data_dir: Path = DEFAULT_DATA_DIR) -> np.ndarray:
    """Load ``audio_waveform.npy`` (mono float32 in (-1, 1))."""
    return np.load(data_dir / "audio_waveform.npy")


def load_mel_frames(data_dir: Path = DEFAULT_DATA_DIR) -> np.ndarray:
    """Load ``audio_frames.npy`` (80, T) log-mel spectrogram in dB."""
    return np.load(data_dir / "audio_frames.npy")


def load_encoder(data_dir: Path = DEFAULT_DATA_DIR) -> np.ndarray:
    """Load ``audio_encoder.npy`` (T/2, 512) post-Conv1d hidden state."""
    return np.load(data_dir / "audio_encoder.npy")


def downsample_waveform(
    waveform: np.ndarray, *, factor: int = WAVEFORM_DOWNSAMPLE_FACTOR
) -> np.ndarray:
    """Strided decimation for fast line plotting.

    80000 samples / 100 = 800 plot points — small enough for Manim
    to draw as a single ``Line`` in <0.5s.
    """
    if factor <= 1:
        return waveform.astype(np.float32, copy=False)
    return waveform[::factor].astype(np.float32, copy=False)


def mel_to_opacity(mel: np.ndarray) -> np.ndarray:
    """Map log-mel dB values in [-80, 0] to opacity in [0, 1]."""
    # Linear interpolation: -80 dB → 0, 0 dB → 1, clipped.
    return np.clip((mel + 80.0) / 80.0, 0.0, 1.0).astype(np.float32)


def mel_grid_columns(mel: np.ndarray, *, max_cols: int = MEL_MAX_COLS) -> int:
    """Return the number of columns to render, capped at ``max_cols``.

    For T > max_cols we downsample by integer factor; for T <= max_cols
    we use the full T. The returned count is used to size the grid.
    """
    t = mel.shape[1]
    return min(t, max_cols)


# --- The Scene ------------------------------------------------------------


class AudioChunks(Scene):
    """Manim scene: audio waveform → mel spectrogram → Conv1d compression."""

    def construct(self) -> None:  # type: ignore[override]
        """Manim entry point — delegates to the free function for testability."""
        build_audio_chunks(self, data_dir=DEFAULT_DATA_DIR)


# --- Orchestration (free function so tests can call with a fake scene) ---


def build_audio_chunks(
    scene: Scene,
    *,
    data_dir: Path = DEFAULT_DATA_DIR,
) -> None:
    """Run the four animation steps against the given scene."""
    waveform = load_waveform(data_dir)
    mel = load_mel_frames(data_dir)
    encoder = load_encoder(data_dir)
    t = mel.shape[1]
    LOGGER.info(
        "Scene 04: waveform=%d, mel=(80, %d), encoder=(%d, 512)",
        waveform.shape[0],
        t,
        encoder.shape[0],
    )

    _step1_waveform(scene, waveform)
    _step2_mel_grid(scene, mel)
    _step3_conv1d_brackets(scene, t)
    _step4_encoder_dots(scene, encoder)
    add_italian_caption(scene, "Ogni frammento diventa un vettore")


# --- Step builders --------------------------------------------------------


def _step1_waveform(scene: Scene, waveform: np.ndarray) -> None:
    """Show the raw waveform as a Manim line plot."""
    ds = downsample_waveform(waveform)
    n = len(ds)

    axes = Axes(
        x_range=[0, n, n // 4],
        y_range=[-0.5, 0.5, 0.25],
        x_length=10.0,
        y_length=2.0,
        tips=False,
        axis_config={"stroke_width": 1, "color": "#888888"},
    )
    axes.shift(UP * 2.0)

    # Build a polyline from the downsampled samples
    points = [axes.c2p(i, float(v)) for i, v in enumerate(ds)]
    # Group adjacent points as Line segments
    lines: list[Line] = []
    for i in range(len(points) - 1):
        lines.append(Line(points[i], points[i + 1], stroke_width=1, color=COLOR_AUDIO))
    waveform_group = VGroup(*lines)
    caption = Text("Forma d'onda (5s @ 16kHz)", font_size=24, color=COLOR_CAPTION)
    caption.next_to(axes, UP, buff=0.2)
    scene.play(FadeIn(Group(axes, waveform_group, caption), run_time=0.5))


def _step2_mel_grid(scene: Scene, mel: np.ndarray) -> None:
    """Show the log-mel spectrogram as an 80×N grid of small squares.

    T is downsampled to ``MEL_MAX_COLS`` (50) columns for visual clarity.
    Each cell's opacity is the normalised mel value.
    """
    t = mel.shape[1]
    n_cols = mel_grid_columns(mel)
    if n_cols < t:
        # Resample to n_cols columns
        indices = np.linspace(0, t - 1, n_cols).astype(int)
        mel = mel[:, indices]
    opacity = mel_to_opacity(mel)

    cell_w, cell_h = 0.18, 0.05
    cells: list[Rectangle] = []
    for r in range(80):
        for c in range(n_cols):
            cell = Rectangle(
                width=cell_w,
                height=cell_h,
                fill_color=COLOR_AUDIO,
                fill_opacity=float(opacity[r, c]),
                stroke_width=0,
            )
            cell.move_to(
                np.array(
                    [
                        (c - n_cols / 2 + 0.5) * cell_w,
                        (r - 40) * cell_h,
                        0,
                    ]
                )
            )
            cells.append(cell)
    grid = VGroup(*cells)
    caption = Text(
        f"Spettrogramma log-mel (80 × {n_cols})", font_size=24, color=COLOR_CAPTION
    )
    caption.next_to(grid, UP, buff=0.2)
    scene.play(FadeIn(Group(grid, caption), run_time=0.5))


def _step3_conv1d_brackets(scene: Scene, t: int) -> None:
    """Show T frame markers with stride-2 brackets (Conv1d compression).

    Draws ``BRACE_MAX_COUNT`` brackets along a horizontal time axis,
    each grouping 2 frames. Real Whisper's T=3000 → 1500 brackets is
    too many; we draw a representative subset.
    """
    n_braces = min(BRACE_MAX_COUNT, t // 2)
    if n_braces == 0:
        return

    # Build a horizontal axis
    axis_length = 8.0
    step = axis_length / n_braces
    # Each bracket covers 2 frames: width = step
    braces: list[Brace] = []
    for i in range(n_braces):
        x_left = -axis_length / 2 + i * step
        # Brace takes a LabeledDot? Simpler: a small Line
        brace = Brace(
            Line(np.array([x_left, 0, 0]), np.array([x_left + step, 0, 0]), stroke_width=1),
            direction=tuple(UP.tolist()),
        )
        braces.append(brace)
    group = VGroup(*braces)
    caption = Text(
        f"Compressione Conv1d: stride=2 ({t} frame → {t // 2} posizioni)",
        font_size=24,
        color=COLOR_CAPTION,
    )
    caption.next_to(group, UP, buff=0.3)
    scene.play(FadeIn(Group(group, caption), run_time=0.5))


def _step4_encoder_dots(scene: Scene, encoder: np.ndarray) -> None:
    """Show the collapsed encoder view: a row of dots for each position.

    Downsample to ``ENCODER_DOT_MAX`` (60) dots for visual clarity.
    The 1500 actual positions would be too dense.
    """
    n_pos = encoder.shape[0]
    n_dots = min(ENCODER_DOT_MAX, n_pos)
    if n_pos > n_dots:
        indices = np.linspace(0, n_pos - 1, n_dots).astype(int)
        encoder_ds = encoder[indices]
    else:
        encoder_ds = encoder

    spacing = 0.12
    dots: list[Dot] = []
    for i, vec in enumerate(encoder_ds):
        # Size dot by vector norm (visual indicator of energy)
        norm = float(np.linalg.norm(vec))
        radius = 0.04 + min(0.08, norm / 50.0)
        dot = Dot(radius=radius, color=COLOR_AUDIO, fill_opacity=0.85)
        dot.move_to(np.array([(i - n_dots / 2 + 0.5) * spacing, -2.0, 0]))
        dots.append(dot)
    grid = VGroup(*dots)
    caption = Text(
        f"Posizioni encoder ({n_pos} totali, mostro {n_dots})",
        font_size=24,
        color=COLOR_CAPTION,
    )
    caption.next_to(grid, UP, buff=0.3)
    scene.play(FadeIn(Group(grid, caption), run_time=0.5))


# --- Re-exports so tests can mock by name ---------------------------------
__all__ = [
    "AudioChunks",
    "COLOR_AUDIO",
    "load_waveform",
    "load_mel_frames",
    "load_encoder",
    "downsample_waveform",
    "mel_to_opacity",
    "mel_grid_columns",
    "build_audio_chunks",
    "Text",
    "Axes",
    "Line",
    "Brace",
    "Rectangle",
    "Dot",
    "FadeIn",
    "Group",
    "VGroup",
]
