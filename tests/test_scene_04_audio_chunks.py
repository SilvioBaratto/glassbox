"""Structural tests for scene 04 (issue #10) — audio chunking.

We test the orchestration logic (data loading, waveform line, mel grid,
Conv1d compression brackets, encoder dot grid) without invoking Manim's
render loop. Mirrors the pattern from earlier scene tests.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCENE_FILE = ROOT / "scenes" / "04_audio_chunks.py"
DATA_DIR = ROOT / "data"
WAVEFORM_PATH = DATA_DIR / "audio_waveform.npy"
FRAMES_PATH = DATA_DIR / "audio_frames.npy"
ENCODER_PATH = DATA_DIR / "audio_encoder.npy"


def _load_scene_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "_scene_04_audio_chunks_under_test",
        SCENE_FILE,
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Pre-load scenes._common so the patch.object targets the right module namespace.
common_mod = importlib.import_module("scenes._common")


# --- Fake data builders ----------------------------------------------------


def _write_fake_data(
    data_dir: Path,
    waveform: np.ndarray,
    frames: np.ndarray,
    encoder: np.ndarray,
) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    np.save(data_dir / "audio_waveform.npy", waveform)
    np.save(data_dir / "audio_frames.npy", frames)
    np.save(data_dir / "audio_encoder.npy", encoder)


@pytest.fixture
def fake_data_dir(tmp_path: Path) -> Path:
    """Build representative 5s @ 16kHz audio + (80, T) mel + (T/2, 512) encoder."""
    out = tmp_path / "data"
    sr = 16000
    duration = 5
    n = sr * duration
    rng = np.random.default_rng(2024)
    waveform = rng.standard_normal(n).astype(np.float32) * 0.3
    # T depends on librosa hop_length=160; ~501 for 5s. Use 100 for test speed.
    T = 100
    frames = rng.uniform(-80.0, 0.0, (80, T)).astype(np.float32)
    encoder = rng.standard_normal((T // 2, 512)).astype(np.float32) * 0.1
    _write_fake_data(out, waveform, frames, encoder)
    return out


# --- Manim mobject mocks (mirror of earlier scene tests) -------------------


class _Mobject:
    """Base mock for any Manim mobject — captures constructor args."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs
        self.calls: list[str] = []

    def __getattr__(self, name: str) -> Any:
        def _stub(*args: Any, **kwargs: Any) -> "_Mobject":
            self.calls.append(name)
            return self

        return _stub


class _Text(_Mobject):
    """Text(text=..., font_size=...)"""


class _Line(_Mobject):
    """Line(...)"""


class _Axes(_Mobject):
    """Axes(...)"""


class _Brace(_Mobject):
    """Brace(mobject, direction=...)"""


class _Rectangle(_Mobject):
    """Rectangle(...)"""


class _Dot(_Mobject):
    """Dot(...)"""


class _FadeIn:
    """Stand-in for manim.FadeIn. Records the wrapped mobject."""

    def __init__(self, mobject: Any, *args: Any, **kwargs: Any) -> None:
        self.mobject = mobject
        self.args = args
        self.kwargs = kwargs


class _Group:
    """Stand-in for manim.Group. Records submobjects in ``.mobjects``."""

    def __init__(self, *mobjects: Any, **kwargs: Any) -> None:
        self.mobjects = list(mobjects)
        self.kwargs = kwargs


class _VGroup(_Group):
    """Stand-in for manim.VGroup."""


class _FakeScene:
    """Stand-in for manim.Scene. Records all animation calls."""

    def __init__(self) -> None:
        self.added: list[Any] = []
        self.played: list[Any] = []
        self.waits: list[float] = []

    def add(self, *mobs: Any) -> None:
        self.added.extend(mobs)

    def play(self, *anims: Any, run_time: float = 0.0) -> None:
        self.played.extend(anims)

    def wait(self, duration: float = 1.0) -> None:
        self.waits.append(duration)


def _unwrap_mobjects(scene: _FakeScene) -> list[Any]:
    out: list[Any] = []
    stack: list[Any] = list(scene.added) + [
        anim.mobject if isinstance(anim, _FadeIn) else anim for anim in scene.played
    ]
    while stack:
        m = stack.pop()
        if isinstance(m, (_Group, _VGroup)):
            stack.extend(m.mobjects)
        else:
            out.append(m)
    return out


def _all_text_strings(scene: _FakeScene) -> list[str]:
    return [
        str(m.args[0]) if m.args else ""
        for m in _unwrap_mobjects(scene)
        if isinstance(m, _Text)
    ]


# --- Subprocess smoke ------------------------------------------------------


def test_when_scene_loaded_as_module_then_AudioChunks_class_exists() -> None:
    mod = _load_scene_module()
    cls = getattr(mod, "AudioChunks", None)
    assert cls is not None, (
        "scenes/04_audio_chunks.py must define class AudioChunks(Scene)"
    )


def test_when_scene_loaded_then_it_subclasses_manim_Scene() -> None:
    import manim

    mod = _load_scene_module()
    cls = mod.AudioChunks
    assert issubclass(cls, manim.Scene), "AudioChunks must subclass manim.Scene"


def test_when_scene_imported_then_no_syntax_errors() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            f"import importlib.util; "
            f"spec = importlib.util.spec_from_file_location('m', '{SCENE_FILE}'); "
            f"m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); "
            f"assert hasattr(m, 'AudioChunks')",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert proc.returncode == 0, (
        f"scene import failed: rc={proc.returncode}\nstderr={proc.stderr}"
    )


def test_when_scene_file_inspected_then_under_500_lines() -> None:
    lines = SCENE_FILE.read_text().count("\n")
    assert lines < 500, f"scene file too long: {lines} lines (limit 500)"


# --- Orchestration (mocked rendering) -------------------------------------


def test_when_construct_called_then_data_loaded_from_provided_paths(
    fake_data_dir: Path,
) -> None:
    """The scene must read audio_waveform, audio_frames, audio_encoder."""
    mod = _load_scene_module()
    scene = _FakeScene()
    with (
        patch.object(mod, "Text", _Text),
        patch.object(common_mod, "Text", _Text),
        patch.object(common_mod, "FadeIn", _FadeIn),
        patch.object(common_mod, "Text", _Text),
        patch.object(common_mod, "FadeIn", _FadeIn),
        patch.object(mod, "Line", _Line),
        patch.object(mod, "Axes", _Axes),
        patch.object(mod, "Brace", _Brace),
        patch.object(mod, "Rectangle", _Rectangle),
        patch.object(mod, "Dot", _Dot),
        patch.object(mod, "FadeIn", _FadeIn),
        patch.object(mod, "Group", _Group),
        patch.object(mod, "VGroup", _VGroup),
    ):
        mod.build_audio_chunks(scene, data_dir=fake_data_dir)
    # An Axes mobject must be present (for the waveform plot)
    axes = [m for m in _unwrap_mobjects(scene) if isinstance(m, _Axes)]
    assert len(axes) >= 1, f"expected >= 1 Axes mobject (waveform), got {len(axes)}"


def test_when_construct_called_then_italian_caption_present(
    fake_data_dir: Path,
) -> None:
    """The 'Ogni frammento diventa un vettore' caption must be visible."""
    mod = _load_scene_module()
    scene = _FakeScene()
    with (
        patch.object(mod, "Text", _Text),
        patch.object(common_mod, "Text", _Text),
        patch.object(common_mod, "FadeIn", _FadeIn),
        patch.object(mod, "Line", _Line),
        patch.object(mod, "Axes", _Axes),
        patch.object(mod, "Brace", _Brace),
        patch.object(mod, "Rectangle", _Rectangle),
        patch.object(mod, "Dot", _Dot),
        patch.object(mod, "FadeIn", _FadeIn),
        patch.object(mod, "Group", _Group),
        patch.object(mod, "VGroup", _VGroup),
    ):
        mod.build_audio_chunks(scene, data_dir=fake_data_dir)
    texts = _all_text_strings(scene)
    assert any("Ogni frammento diventa un vettore" in s for s in texts), (
        f"missing 'Ogni frammento diventa un vettore' caption. Got: {texts}"
    )


def test_when_construct_called_then_green_modality_color_used(
    fake_data_dir: Path,
) -> None:
    """Audio (modality) must use the green color #10B981 per AC."""
    mod = _load_scene_module()
    green = getattr(mod, "COLOR_AUDIO", None) or getattr(mod, "GREEN_AUDIO", None)
    assert green is not None, "scene must define a COLOR_AUDIO / GREEN_AUDIO constant"
    assert green.upper() == "#10B981", (
        f"audio color must be #10B981 per modality convention, got {green}"
    )


def test_when_construct_called_then_brace_for_conv1d_stride(
    fake_data_dir: Path,
) -> None:
    """The Conv1d compression brackets must be present in the scene."""
    mod = _load_scene_module()
    scene = _FakeScene()
    with (
        patch.object(mod, "Text", _Text),
        patch.object(common_mod, "Text", _Text),
        patch.object(common_mod, "FadeIn", _FadeIn),
        patch.object(mod, "Line", _Line),
        patch.object(mod, "Axes", _Axes),
        patch.object(mod, "Brace", _Brace),
        patch.object(mod, "Rectangle", _Rectangle),
        patch.object(mod, "Dot", _Dot),
        patch.object(mod, "FadeIn", _FadeIn),
        patch.object(mod, "Group", _Group),
        patch.object(mod, "VGroup", _VGroup),
    ):
        mod.build_audio_chunks(scene, data_dir=fake_data_dir)
    braces = [m for m in _unwrap_mobjects(scene) if isinstance(m, _Brace)]
    assert len(braces) >= 1, (
        f"expected >= 1 Brace mobject (Conv1d stride=2 indicator), got {len(braces)}"
    )


# --- Pure-data helpers -----------------------------------------------------


def test_when_load_waveform_called_then_shape_5s_16khz() -> None:
    mod = _load_scene_module()
    w = mod.load_waveform()
    assert w.ndim == 1, f"expected 1-D mono, got shape {w.shape}"
    assert w.dtype == np.float32, f"expected float32, got {w.dtype}"
    assert w.min() >= -1.0 and w.max() <= 1.0


def test_when_load_mel_frames_called_then_shape_80_T() -> None:
    mod = _load_scene_module()
    f = mod.load_mel_frames()
    assert f.ndim == 2 and f.shape[0] == 80, f"expected (80, T), got {f.shape}"
    assert f.dtype == np.float32, f"expected float32, got {f.dtype}"


def test_when_load_encoder_called_then_shape_T_over_2_512() -> None:
    mod = _load_scene_module()
    e = mod.load_encoder()
    assert e.ndim == 2 and e.shape[1] == 512, f"expected (T/2, 512), got {e.shape}"
    assert e.dtype == np.float32, f"expected float32, got {e.dtype}"


def test_when_downsample_waveform_called_then_returns_smaller_array() -> None:
    """The waveform must be downsampled for fast line plotting."""
    mod = _load_scene_module()
    w = np.linspace(-0.5, 0.5, 80000, dtype=np.float32)
    out = mod.downsample_waveform(w, factor=100)
    assert len(out) == 800, f"expected 800 samples, got {len(out)}"
    assert out.dtype == np.float32


def test_when_mel_to_opacity_called_then_range_in_zero_one() -> None:
    """Mel values in [-80, 0] dB must map to [0, 1] opacity."""
    mod = _load_scene_module()
    mel = np.array([[-80.0, -40.0, 0.0]], dtype=np.float32)
    opacity = mod.mel_to_opacity(mel)
    assert opacity.shape == (1, 3)
    assert opacity.min() >= 0.0, f"min < 0: {opacity.min()}"
    assert opacity.max() <= 1.0, f"max > 1: {opacity.max()}"
    # -80 dB → 0 opacity (quiet), 0 dB → 1 (loud)
    assert opacity[0, 0] < opacity[0, 1] < opacity[0, 2]


def test_when_mel_grid_count_called_then_reasonable_for_T() -> None:
    """The mel grid must downsample T to <= 50 columns for visual clarity."""
    mod = _load_scene_module()
    mel = np.zeros((80, 501), dtype=np.float32)
    count = mod.mel_grid_columns(mel, max_cols=50)
    assert count <= 50, f"expected <= 50 grid columns, got {count}"
    assert count > 0, f"expected > 0 grid columns, got {count}"
