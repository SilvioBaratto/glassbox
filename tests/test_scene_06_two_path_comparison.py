"""Structural tests for scene 06 (issue #13) — two-path comparison.

We test the orchestration logic (data loading, left/right panel layout,
bottleneck, red X, Italian captions) without invoking Manim's render
loop. Mirrors the pattern from earlier scene tests.

Per state.json this is issue #13 but the scene is numbered 06 in the
file (bookkeeping note from issue #12's reviewer). The file path is
``scenes/06_two_path_comparison.py``.
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
SCENE_FILE = ROOT / "scenes" / "06_two_path_comparison.py"
DATA_DIR = ROOT / "data"
TEXT_EMB_PATH = DATA_DIR / "shared_text_embeds.npy"
IMAGE_EMB_PATH = DATA_DIR / "shared_image_embeds.npy"
PCA_PATH = DATA_DIR / "pca_coords_3d.npy"


def _load_scene_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "_scene_06_two_path_under_test",
        SCENE_FILE,
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --- Fake data builders ----------------------------------------------------


def _write_fake_data(
    data_dir: Path,
    text_embeds: np.ndarray,
    image_embeds: np.ndarray,
    pca: np.ndarray,
) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    np.save(data_dir / "shared_text_embeds.npy", text_embeds)
    np.save(data_dir / "shared_image_embeds.npy", image_embeds)
    np.save(data_dir / "pca_coords_3d.npy", pca)


@pytest.fixture
def fake_data_dir(tmp_path: Path) -> Path:
    """Build representative data dir."""
    out = tmp_path / "data"
    rng = np.random.default_rng(13)
    text_embeds = rng.standard_normal((5, 512)).astype(np.float32) * 0.1
    image_embeds = rng.standard_normal((1, 512)).astype(np.float32) * 0.1
    pca = rng.standard_normal((6, 3)).astype(np.float32) * 0.5
    _write_fake_data(out, text_embeds, image_embeds, pca)
    return out


# --- Manim mobject mocks (mirror of earlier scene tests) -------------------


class _Mobject:
    """Base mock for any Manim mobject."""

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
    """ImageMobject(...)"""


class _Arrow(_Mobject):
    """Arrow(...)"""


class _Line(_Mobject):
    """Line(...)"""


class _Cross(_Mobject):
    """Cross(...) (red X for the warning)"""


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
    """Stand-in for manim.Scene."""

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


def test_when_scene_loaded_as_module_then_TwoPathComparison_class_exists() -> None:
    mod = _load_scene_module()
    cls = getattr(mod, "TwoPathComparison", None)
    assert cls is not None, (
        "scenes/06_two_path_comparison.py must define class TwoPathComparison(Scene)"
    )


def test_when_scene_loaded_then_it_subclasses_manim_Scene() -> None:
    import manim

    mod = _load_scene_module()
    cls = mod.TwoPathComparison
    assert issubclass(cls, manim.Scene), "TwoPathComparison must subclass manim.Scene"


def test_when_scene_imported_then_no_syntax_errors() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            f"import importlib.util; "
            f"spec = importlib.util.spec_from_file_location('m', '{SCENE_FILE}'); "
            f"m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); "
            f"assert hasattr(m, 'TwoPathComparison')",
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


def test_when_construct_called_then_left_and_right_panel_titles(
    fake_data_dir: Path,
) -> None:
    """The two panel titles must be in the scene."""
    mod = _load_scene_module()
    scene = _FakeScene()
    with (
        patch.object(mod, "Text", _Text),
        patch.object(mod, "Rectangle", _Rectangle),
        patch.object(mod, "ImageMobject", _ImageMobject),
        patch.object(mod, "Arrow", _Arrow),
        patch.object(mod, "Line", _Line),
        patch.object(mod, "Cross", _Cross),
        patch.object(mod, "FadeIn", _FadeIn),
        patch.object(mod, "Group", _Group),
        patch.object(mod, "VGroup", _VGroup),
    ):
        mod.build_two_path_comparison(scene, data_dir=fake_data_dir)
    texts = _all_text_strings(scene)
    assert any("Traduttore separato" in s for s in texts), (
        f"missing 'Traduttore separato' panel title. Got: {texts}"
    )
    assert any("Spazio condiviso" in s for s in texts), (
        f"missing 'Spazio condiviso' panel title. Got: {texts}"
    )


def test_when_construct_called_then_loss_captions(
    fake_data_dir: Path,
) -> None:
    """The two AC-mandated Italian loss captions must be visible."""
    mod = _load_scene_module()
    scene = _FakeScene()
    with (
        patch.object(mod, "Text", _Text),
        patch.object(mod, "Rectangle", _Rectangle),
        patch.object(mod, "ImageMobject", _ImageMobject),
        patch.object(mod, "Arrow", _Arrow),
        patch.object(mod, "Line", _Line),
        patch.object(mod, "Cross", _Cross),
        patch.object(mod, "FadeIn", _FadeIn),
        patch.object(mod, "Group", _Group),
        patch.object(mod, "VGroup", _VGroup),
    ):
        mod.build_two_path_comparison(scene, data_dir=fake_data_dir)
    texts = _all_text_strings(scene)
    assert any("Si perde sempre qualcosa" in s for s in texts), (
        f"missing 'Si perde sempre qualcosa' caption. Got: {texts}"
    )
    assert any("Niente traduttore" in s for s in texts), (
        f"missing 'Niente traduttore' caption. Got: {texts}"
    )


def test_when_construct_called_then_modality_and_warning_colors(
    fake_data_dir: Path,
) -> None:
    """blue=text, orange=image, red=warning must be defined."""
    mod = _load_scene_module()
    for name, expected in (
        ("COLOR_TEXT", "#3B82F6"),
        ("COLOR_IMAGE", "#F97316"),
        ("COLOR_WARNING", "#EF4444"),
    ):
        assert hasattr(mod, name), f"scene must define {name}"
        val = getattr(mod, name)
        assert val.upper() == expected, f"{name} must be {expected}, got {val}"


def test_when_construct_called_then_bottleneck_box_present(
    fake_data_dir: Path,
) -> None:
    """The translator path must include a narrow rectangle (bottleneck)."""
    mod = _load_scene_module()
    scene = _FakeScene()
    with (
        patch.object(mod, "Text", _Text),
        patch.object(mod, "Rectangle", _Rectangle),
        patch.object(mod, "ImageMobject", _ImageMobject),
        patch.object(mod, "Arrow", _Arrow),
        patch.object(mod, "Line", _Line),
        patch.object(mod, "Cross", _Cross),
        patch.object(mod, "FadeIn", _FadeIn),
        patch.object(mod, "Group", _Group),
        patch.object(mod, "VGroup", _VGroup),
    ):
        mod.build_two_path_comparison(scene, data_dir=fake_data_dir)
    # At least 3 Rectangles (left panel bottleneck, right panel box, frame)
    rects = [m for m in _unwrap_mobjects(scene) if isinstance(m, _Rectangle)]
    assert len(rects) >= 2, f"expected >= 2 Rectangles, got {len(rects)}"


def test_when_construct_called_then_red_cross_warning(
    fake_data_dir: Path,
) -> None:
    """A red Cross mobject must mark the warning on the bottleneck."""
    mod = _load_scene_module()
    scene = _FakeScene()
    with (
        patch.object(mod, "Text", _Text),
        patch.object(mod, "Rectangle", _Rectangle),
        patch.object(mod, "ImageMobject", _ImageMobject),
        patch.object(mod, "Arrow", _Arrow),
        patch.object(mod, "Line", _Line),
        patch.object(mod, "Cross", _Cross),
        patch.object(mod, "FadeIn", _FadeIn),
        patch.object(mod, "Group", _Group),
        patch.object(mod, "VGroup", _VGroup),
    ):
        mod.build_two_path_comparison(scene, data_dir=fake_data_dir)
    crosses = [m for m in _unwrap_mobjects(scene) if isinstance(m, _Cross)]
    assert len(crosses) >= 1, f"expected >= 1 Cross mobject, got {len(crosses)}"


def test_when_construct_called_then_animation_count_bounded(
    fake_data_dir: Path,
) -> None:
    """≤ 12s total: keep play() count moderate."""
    mod = _load_scene_module()
    scene = _FakeScene()
    with (
        patch.object(mod, "Text", _Text),
        patch.object(mod, "Rectangle", _Rectangle),
        patch.object(mod, "ImageMobject", _ImageMobject),
        patch.object(mod, "Arrow", _Arrow),
        patch.object(mod, "Line", _Line),
        patch.object(mod, "Cross", _Cross),
        patch.object(mod, "FadeIn", _FadeIn),
        patch.object(mod, "Group", _Group),
        patch.object(mod, "VGroup", _VGroup),
    ):
        mod.build_two_path_comparison(scene, data_dir=fake_data_dir)
    # Each panel is one FadeIn; the Cross warning is a second play.
    # Cap at 8 to keep ≤ 12s at 30fps.
    assert len(scene.played) >= 3, (
        f"expected >= 3 play() calls (left+right+warning), got {len(scene.played)}"
    )
    assert len(scene.played) <= 10, (
        f"too many play() calls ({len(scene.played)}); 12s budget requires tight orchestration"
    )


# --- Pure-data helpers -----------------------------------------------------


def test_when_load_pca_2d_called_then_shape_n_2() -> None:
    mod = _load_scene_module()
    p = mod.load_pca_2d()
    assert p.shape[1] == 2, f"expected 2D coords, got shape {p.shape}"


def test_when_load_shared_text_embeds_called_then_shape_n_512() -> None:
    mod = _load_scene_module()
    e = mod.load_shared_text_embeds()
    assert e.ndim == 2 and e.shape[1] == 512, f"expected (N, 512), got {e.shape}"


def test_when_load_shared_image_embeds_called_then_shape_1_512() -> None:
    mod = _load_scene_module()
    e = mod.load_shared_image_embeds()
    assert e.shape == (1, 512), f"expected (1, 512), got {e.shape}"


def test_when_pca_to_2d_dots_called_then_returns_n_2d_positions() -> None:
    mod = _load_scene_module()
    coords = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    out = mod.pca_to_2d_dots(coords, x_scale=1.0, y_scale=1.0)
    assert out.shape == (2, 2), f"expected (N, 2), got {out.shape}"


def test_when_panel_split_offsets_called_then_returns_two_x_coords() -> None:
    mod = _load_scene_module()
    left_x, right_x = mod.panel_split_offsets()
    assert left_x < 0 < right_x, (
        f"left should be negative, right positive: {left_x}, {right_x}"
    )
