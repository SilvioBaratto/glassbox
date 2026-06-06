"""Structural tests for scene 03 (issue #9) — image patch splitting.

We test the orchestration logic (data loading, slicing animation,
196-dot grid) without invoking Manim's render loop. Mirrors the
pattern from test_scene_01_tokenization.py and test_scene_02_llm_numbers.py.
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
SCENE_FILE = ROOT / "scenes" / "03_image_patches.py"
DATA_DIR = ROOT / "data"
PATCHES_PATH = DATA_DIR / "patch_embeddings.npy"
GRID_PATH = DATA_DIR / "patch_grid.npy"
SIDE_PATH = DATA_DIR / "sample_image_224.npy"


def _load_scene_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "_scene_03_image_patches_under_test",
        SCENE_FILE,
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Pre-load scenes._common so the patch.object targets the right module namespace.
common_mod = importlib.import_module("scenes._common")


# --- Fake data builders (in-memory, never touches real data/) -------------


def _write_fake_data(
    data_dir: Path,
    patches: np.ndarray,
    image: np.ndarray,
) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    np.save(data_dir / "patch_embeddings.npy", patches)
    grid = np.arange(patches.shape[0], dtype=np.int64).reshape(14, 14)
    np.save(data_dir / "patch_grid.npy", grid)
    np.save(data_dir / "sample_image_224.npy", image)


@pytest.fixture
def fake_data_dir(tmp_path: Path) -> Path:
    """Build a representative 14x14 patch + 224x224 RGB image in tmp_path."""
    out = tmp_path / "data"
    n_patches = 14 * 14
    embed_dim = 768
    rng = np.random.default_rng(1234)
    patches = rng.standard_normal((n_patches, embed_dim)).astype(np.float32) * 0.1
    image = rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)
    _write_fake_data(out, patches, image)
    return out


# --- Manim mobject mocks (mirror of test_scene_01) -------------------------


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
    """Stand-in for manim.VGroup (vectorised Group)."""


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
    """Recursively unwrap FadeIn.mobject and Group/VGroup.mobjects."""
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


def test_when_scene_loaded_as_module_then_ImagePatches_class_exists() -> None:
    mod = _load_scene_module()
    cls = getattr(mod, "ImagePatches", None)
    assert cls is not None, (
        "scenes/03_image_patches.py must define class ImagePatches(Scene)"
    )


def test_when_scene_loaded_then_it_subclasses_manim_Scene() -> None:
    import manim

    mod = _load_scene_module()
    cls = mod.ImagePatches
    assert issubclass(cls, manim.Scene), "ImagePatches must subclass manim.Scene"


def test_when_scene_imported_then_no_syntax_errors() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            f"import importlib.util; "
            f"spec = importlib.util.spec_from_file_location('m', '{SCENE_FILE}'); "
            f"m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); "
            f"assert hasattr(m, 'ImagePatches')",
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
    """The scene must read from data/sample_image_224.npy + patch_embeddings.npy.

    Reviewer's bug fix: the AC's "data/preprocessed_image.npy" path was
    for the normalised CHW float tensor (which renders as black). We
    use ``sample_image_224.npy`` (uint8 HWC) for display.
    """
    mod = _load_scene_module()
    scene = _FakeScene()
    with (
        patch.object(mod, "Text", _Text),
        patch.object(common_mod, "Text", _Text),
        patch.object(common_mod, "FadeIn", _FadeIn),
        patch.object(common_mod, "Text", _Text),
        patch.object(common_mod, "FadeIn", _FadeIn),
        patch.object(mod, "Rectangle", _Rectangle),
        patch.object(mod, "ImageMobject", _ImageMobject),
        patch.object(mod, "FadeIn", _FadeIn),
        patch.object(mod, "Group", _Group),
        patch.object(mod, "VGroup", _VGroup),
    ):
        mod.build_image_patches(scene, data_dir=fake_data_dir)

    # The scene must construct an ImageMobject for the original photo
    image_objs = [m for m in _unwrap_mobjects(scene) if isinstance(m, _ImageMobject)]
    assert len(image_objs) >= 1, (
        f"expected >= 1 ImageMobject (full-frame image), got {len(image_objs)}"
    )


def test_when_construct_called_then_italian_caption_present(
    fake_data_dir: Path,
) -> None:
    """The 'Ogni quadrato diventa un vettore' caption must be visible."""
    mod = _load_scene_module()
    scene = _FakeScene()
    with (
        patch.object(mod, "Text", _Text),
        patch.object(common_mod, "Text", _Text),
        patch.object(common_mod, "FadeIn", _FadeIn),
        patch.object(mod, "Rectangle", _Rectangle),
        patch.object(mod, "ImageMobject", _ImageMobject),
        patch.object(mod, "FadeIn", _FadeIn),
        patch.object(mod, "Group", _Group),
        patch.object(mod, "VGroup", _VGroup),
    ):
        mod.build_image_patches(scene, data_dir=fake_data_dir)
    texts = _all_text_strings(scene)
    assert any("Ogni quadrato diventa un vettore" in s for s in texts), (
        f"missing 'Ogni quadrato diventa un vettore' caption. Got: {texts}"
    )


def test_when_construct_called_then_orange_modality_color_used(
    fake_data_dir: Path,
) -> None:
    """Image (modality) must use the orange color #F97316 per AC."""
    mod = _load_scene_module()
    orange = getattr(mod, "COLOR_IMAGE", None) or getattr(mod, "ORANGE_IMAGE", None)
    assert orange is not None, "scene must define a COLOR_IMAGE / ORANGE_IMAGE constant"
    assert orange.upper() == "#F97316", (
        f"image color must be #F97316 per modality convention, got {orange}"
    )


def test_when_construct_called_then_animation_count_keeps_pace(
    fake_data_dir: Path,
) -> None:
    """Animation count must be bounded per the reviewer's 8s-budget fix.

    14 rows at 1s/row = 14s. We animate 2 rows at a time → 7 beats for
    the slicing step. With 4 steps, total play() calls should be small.
    """
    mod = _load_scene_module()
    scene = _FakeScene()
    with (
        patch.object(mod, "Text", _Text),
        patch.object(common_mod, "Text", _Text),
        patch.object(common_mod, "FadeIn", _FadeIn),
        patch.object(mod, "Rectangle", _Rectangle),
        patch.object(mod, "ImageMobject", _ImageMobject),
        patch.object(mod, "FadeIn", _FadeIn),
        patch.object(mod, "Group", _Group),
        patch.object(mod, "VGroup", _VGroup),
    ):
        mod.build_image_patches(scene, data_dir=fake_data_dir)
    # The AC describes 4 steps. Each step is at least one play() call.
    # 7 (slicing 2-rows-at-a-time) + 1 (vector bars) + 1 (dot grid) +
    # 1 (image) + 1 (caption) = 11. Cap at 14 to leave headroom.
    assert len(scene.played) >= 4, (
        f"expected >= 4 scene.play() calls, got {len(scene.played)}"
    )
    assert len(scene.played) <= 14, (
        f"too many play() calls ({len(scene.played)}); reviewer's 8s budget fix "
        "requires 2 rows at a time, not 1"
    )


# --- Pure-data helpers -----------------------------------------------------


def test_when_load_image_for_display_called_then_returns_uint8_hwc() -> None:
    """``sample_image_224.npy`` is uint8 HWC 224x224x3 — displayable as-is."""
    mod = _load_scene_module()
    img = mod.load_image_for_display()
    assert img.shape == (224, 224, 3), f"expected (224, 224, 3), got {img.shape}"
    assert img.dtype == np.uint8, f"expected uint8, got {img.dtype}"


def test_when_load_patch_embeddings_called_then_shape_is_196_768() -> None:
    mod = _load_scene_module()
    p = mod.load_patch_embeddings()
    assert p.shape == (196, 768), f"expected (196, 768), got {p.shape}"
    assert p.dtype == np.float32, f"expected float32, got {p.dtype}"


def test_when_normalise_first_32_dims_called_then_range_in_zero_one() -> None:
    """The bar chart shows the first 32 dims of each patch, normalised to [0, 1]."""
    mod = _load_scene_module()
    vec = np.array([1.0, -2.0, 3.0, 0.0, 5.0] + [0.0] * 27, dtype=np.float32)
    normed = mod.normalise_first_32_dims(vec)
    assert normed.shape == (32,)
    assert normed.min() >= 0.0, f"min < 0: {normed.min()}"
    assert normed.max() <= 1.0, f"max > 1: {normed.max()}"
    # Original non-zero values should be present, scaled
    assert normed[0] > 0  # 1.0 is positive
    assert normed[1] == 0  # -2.0 is the min, normalised to 0


def test_when_build_dot_grid_layout_called_then_shape_14_14() -> None:
    """The 14x14 dot grid layout helper must produce 196 positions."""
    mod = _load_scene_module()
    positions = mod.build_dot_grid_layout()
    assert len(positions) == 196, f"expected 196 positions, got {len(positions)}"
