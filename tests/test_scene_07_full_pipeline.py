"""Structural tests for scene 07 (issue #12) — full pipeline.

We test the orchestration logic (data loading, step sequencing, arrow
labels, model output) without invoking Manim's render loop. Mirrors
the pattern from earlier scene tests.
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
SCENE_FILE = ROOT / "scenes" / "07_full_pipeline.py"
DATA_DIR = ROOT / "data"
IMAGE_PATH = DATA_DIR / "sample_image_224.npy"
TOKENS_PATH = DATA_DIR / "tokens.npy"
PCA_PATH = DATA_DIR / "pca_coords_3d.npy"


def _load_scene_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "_scene_07_full_pipeline_under_test",
        SCENE_FILE,
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --- Fake data builders ----------------------------------------------------


def _write_fake_data(
    data_dir: Path,
    image: np.ndarray,
    tokens: np.ndarray,
    pca: np.ndarray,
) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    np.save(data_dir / "sample_image_224.npy", image)
    np.save(data_dir / "tokens.npy", tokens)
    np.save(data_dir / "pca_coords_3d.npy", pca)


@pytest.fixture
def fake_data_dir(tmp_path: Path) -> Path:
    """Build a representative data dir (image, tokens, 2D PCA)."""
    out = tmp_path / "data"
    rng = np.random.default_rng(12)
    image = rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)
    # 9 content tokens (row 0: 'Come si fa a capire tutto?') — 77 padded
    content_ids = [891, 2990, 2800, 320, 1289, 1454, 764, 8105, 286]
    tokens = np.array([[49406] + content_ids + [49407] * 67], dtype=np.int64)
    pca = rng.standard_normal((6, 3)).astype(np.float32) * 0.5
    _write_fake_data(out, image, tokens, pca)
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


class _Rectangle(_Mobject):
    """Rectangle(...)"""


class _ImageMobject(_Mobject):
    """ImageMobject(filename_or_array)"""


class _Arrow(_Mobject):
    """Arrow(...)"""


class _FadeIn:
    """Stand-in for manim.FadeIn."""

    def __init__(self, mobject: Any, *args: Any, **kwargs: Any) -> None:
        self.mobject = mobject
        self.args = args
        self.kwargs = kwargs


class _Group(_Mobject):
    """Stand-in for manim.Group."""

    def __init__(self, *mobjects: Any, **kwargs: Any) -> None:
        super().__init__()
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


def test_when_scene_loaded_as_module_then_FullPipeline_class_exists() -> None:
    mod = _load_scene_module()
    cls = getattr(mod, "FullPipeline", None)
    assert cls is not None, (
        "scenes/07_full_pipeline.py must define class FullPipeline(Scene)"
    )


def test_when_scene_loaded_then_it_subclasses_manim_Scene() -> None:
    import manim

    mod = _load_scene_module()
    cls = mod.FullPipeline
    assert issubclass(cls, manim.Scene), "FullPipeline must subclass manim.Scene"


def test_when_scene_imported_then_no_syntax_errors() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            f"import importlib.util; "
            f"spec = importlib.util.spec_from_file_location('m', '{SCENE_FILE}'); "
            f"m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); "
            f"assert hasattr(m, 'FullPipeline')",
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


def test_when_construct_called_then_italian_arrow_labels_present(
    fake_data_dir: Path,
) -> None:
    """The 4 AC-mandated Italian labels must be in the scene."""
    mod = _load_scene_module()
    scene = _FakeScene()
    with (
        patch.object(mod, "Text", _Text),
        patch.object(mod, "Rectangle", _Rectangle),
        patch.object(mod, "ImageMobject", _ImageMobject),
        patch.object(mod, "Arrow", _Arrow),
        patch.object(mod, "FadeIn", _FadeIn),
        patch.object(mod, "Group", _Group),
        patch.object(mod, "VGroup", _VGroup),
    ):
        mod.build_full_pipeline(scene, data_dir=fake_data_dir)
    texts = _all_text_strings(scene)
    for label in ("Patch", "Token", "Spazio condiviso", "Risposta"):
        assert any(label in s for s in texts), (
            f"missing AC arrow label {label!r}. Got: {texts}"
        )


def test_when_construct_called_then_input_caption_present(
    fake_data_dir: Path,
) -> None:
    """The opening caption 'Un gatto sul divano' must be visible."""
    mod = _load_scene_module()
    scene = _FakeScene()
    with (
        patch.object(mod, "Text", _Text),
        patch.object(mod, "Rectangle", _Rectangle),
        patch.object(mod, "ImageMobject", _ImageMobject),
        patch.object(mod, "Arrow", _Arrow),
        patch.object(mod, "FadeIn", _FadeIn),
        patch.object(mod, "Group", _Group),
        patch.object(mod, "VGroup", _VGroup),
    ):
        mod.build_full_pipeline(scene, data_dir=fake_data_dir)
    texts = _all_text_strings(scene)
    assert any("Un gatto sul divano" in s for s in texts), (
        f"missing 'Un gatto sul divano' caption. Got: {texts}"
    )


def test_when_construct_called_then_modality_colors_defined(
    fake_data_dir: Path,
) -> None:
    """All three modality colors must be defined per the AC."""
    mod = _load_scene_module()
    for name in ("COLOR_TEXT", "COLOR_IMAGE", "COLOR_AUDIO"):
        assert hasattr(mod, name), f"scene must define {name} constant"
        val = getattr(mod, name)
        assert val.startswith("#") and len(val) == 7, (
            f"{name} must be a hex string, got {val!r}"
        )


def test_when_construct_called_then_image_mobject_used(
    fake_data_dir: Path,
) -> None:
    """The original photo must be displayed via ImageMobject (Step 1)."""
    mod = _load_scene_module()
    scene = _FakeScene()
    with (
        patch.object(mod, "Text", _Text),
        patch.object(mod, "Rectangle", _Rectangle),
        patch.object(mod, "ImageMobject", _ImageMobject),
        patch.object(mod, "Arrow", _Arrow),
        patch.object(mod, "FadeIn", _FadeIn),
        patch.object(mod, "Group", _Group),
        patch.object(mod, "VGroup", _VGroup),
    ):
        mod.build_full_pipeline(scene, data_dir=fake_data_dir)
    images = [m for m in _unwrap_mobjects(scene) if isinstance(m, _ImageMobject)]
    assert len(images) >= 1, f"expected >= 1 ImageMobject (photo), got {len(images)}"


def test_when_construct_called_then_animation_count_bounded(
    fake_data_dir: Path,
) -> None:
    """Each step ≤ 3s, total ≤ 12s. Animation count should be small."""
    mod = _load_scene_module()
    scene = _FakeScene()
    with (
        patch.object(mod, "Text", _Text),
        patch.object(mod, "Rectangle", _Rectangle),
        patch.object(mod, "ImageMobject", _ImageMobject),
        patch.object(mod, "Arrow", _Arrow),
        patch.object(mod, "FadeIn", _FadeIn),
        patch.object(mod, "Group", _Group),
        patch.object(mod, "VGroup", _VGroup),
    ):
        mod.build_full_pipeline(scene, data_dir=fake_data_dir)
    # 4 AC steps + 1 final = 5+ plays. Cap at 12 to leave wiggle room.
    assert len(scene.played) >= 5, (
        f"expected >= 5 play() calls (4 steps + final), got {len(scene.played)}"
    )
    assert len(scene.played) <= 12, (
        f"too many play() calls ({len(scene.played)}); "
        "12s budget requires tight orchestration"
    )


# --- Pure-data helpers -----------------------------------------------------


def test_when_load_image_called_then_uint8_hwc_224() -> None:
    mod = _load_scene_module()
    img = mod.load_image()
    assert img.shape == (224, 224, 3), f"expected (224, 224, 3), got {img.shape}"
    assert img.dtype == np.uint8, f"expected uint8, got {img.dtype}"


def test_when_load_tokens_called_then_n_rows_of_77() -> None:
    mod = _load_scene_module()
    t = mod.load_tokens()
    assert t.ndim == 2, f"expected 2-D, got {t.ndim}"
    assert t.shape[1] == 77, f"expected 77 cols, got {t.shape[1]}"
    assert t.dtype == np.int64, f"expected int64, got {t.dtype}"


def test_when_load_pca_2d_called_then_n_rows_3_columns() -> None:
    """The 2D PCA slice for the recap scene."""
    mod = _load_scene_module()
    p = mod.load_pca_2d()
    assert p.ndim == 2 and p.shape[1] == 2, f"expected (N, 2), got {p.shape}"


def test_when_pick_first_content_row_called_then_skips_specials() -> None:
    """The first non-pad content row from tokens.npy."""
    mod = _load_scene_module()
    row0 = mod.pick_first_content_row()
    # row 0 is the 'Come si fa a capire tutto?' row, 9 content tokens
    assert row0.shape[0] == 9, f"expected 9 content tokens, got {row0.shape[0]}"
    assert 49406 not in row0 and 49407 not in row0, f"special tokens leaked: {row0}"
