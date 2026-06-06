"""Structural tests for scene 05 (issue #11) — shared vector space (3D).

We test the orchestration logic (data loading, 3D axes, dot placement,
camera rotation) without invoking Manim's 3D render loop. Mirrors the
pattern from earlier scene tests.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCENE_FILE = ROOT / "scenes" / "05_shared_vector_space.py"
DATA_DIR = ROOT / "data"
PCA_PATH = DATA_DIR / "pca_coords_3d.npy"
LABELS_PATH = DATA_DIR / "pca_labels.json"


def _load_scene_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "_scene_05_shared_vector_space_under_test",
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
    coords: np.ndarray,
    labels: dict[str, list[str] | dict[str, str]],
) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    np.save(data_dir / "pca_coords_3d.npy", coords)
    (data_dir / "pca_labels.json").write_text(json.dumps(labels))


@pytest.fixture
def fake_data_dir(tmp_path: Path) -> Path:
    """Build representative PCA coords + labels (5 text + 1 image)."""
    out = tmp_path / "data"
    rng = np.random.default_rng(11)
    coords = rng.standard_normal((6, 3)).astype(np.float32) * 0.5
    labels = {
        "modality": ["text"] * 5 + ["image"],
        "label": ["come", "si", "fa", "a", "capire", "sample_image.jpg"],
        "colors": {"text": "#3B82F6", "image": "#F97316"},
    }
    _write_fake_data(out, coords, labels)
    return out


# --- Manim mobject mocks (mirror of earlier scene tests) -------------------


class _Mobject:
    """Base mock for any Manim mobject — captures constructor args."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs
        self.calls: list[str] = []

    def __getattr__(self, name: str) -> Any:
        # Return a no-op recorder for any chained method (shift, scale, etc.)
        def _stub(*args: Any, **kwargs: Any) -> "_Mobject":
            self.calls.append(name)
            return self

        return _stub


class _Text(_Mobject):
    """Text(text=..., font_size=...)"""


class _ThreeDAxes(_Mobject):
    """ThreeDAxes(...). Pre-built sub-axes for ``.x_axis.get_end()`` chains."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.x_axis = _Mobject()
        self.y_axis = _Mobject()
        self.z_axis = _Mobject()


class _Dot3D(_Mobject):
    """Dot3D(...)"""


class _FadeIn:
    """Stand-in for manim.FadeIn."""

    def __init__(self, mobject: Any, *args: Any, **kwargs: Any) -> None:
        self.mobject = mobject
        self.args = args
        self.kwargs = kwargs


class _Group:
    """Stand-in for manim.Group."""

    def __init__(self, *mobjects: Any, **kwargs: Any) -> None:
        self.mobjects = list(mobjects)
        self.kwargs = kwargs


class _VGroup(_Group):
    """Stand-in for manim.VGroup."""


class _ThreeDScene:
    """Stand-in for manim.ThreeDScene. Records camera moves too."""

    def __init__(self) -> None:
        self.added: list[Any] = []
        self.played: list[Any] = []
        self.waits: list[float] = []
        self.camera_moves: list[dict[str, Any]] = []

    def add(self, *mobs: Any) -> None:
        self.added.extend(mobs)

    def play(self, *anims: Any, run_time: float = 0.0) -> None:
        self.played.extend(anims)

    def wait(self, duration: float = 1.0) -> None:
        self.waits.append(duration)

    def set_camera_orientation(self, *, phi: float = 0.0, theta: float = 0.0) -> None:
        self.camera_moves.append({"phi": phi, "theta": theta, "animate": False})

    def move_camera(
        self, *, phi: float = 0.0, theta: float = 0.0, **kwargs: Any
    ) -> None:
        self.camera_moves.append(
            {"phi": phi, "theta": theta, "animate": True, **kwargs}
        )


def _unwrap_mobjects(scene: _ThreeDScene) -> list[Any]:
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


def _all_text_strings(scene: _ThreeDScene) -> list[str]:
    return [
        str(m.args[0]) if m.args else ""
        for m in _unwrap_mobjects(scene)
        if isinstance(m, _Text)
    ]


# --- Subprocess smoke ------------------------------------------------------


def test_when_scene_loaded_as_module_then_SharedVectorSpace_class_exists() -> None:
    mod = _load_scene_module()
    cls = getattr(mod, "SharedVectorSpace", None)
    assert cls is not None, (
        "scenes/05_shared_vector_space.py must define class SharedVectorSpace(ThreeDScene)"
    )


def test_when_scene_loaded_then_it_subclasses_manim_ThreeDScene() -> None:
    import manim

    mod = _load_scene_module()
    cls = mod.SharedVectorSpace
    assert issubclass(cls, manim.ThreeDScene), (
        "SharedVectorSpace must subclass manim.ThreeDScene"
    )


def test_when_scene_imported_then_no_syntax_errors() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            f"import importlib.util; "
            f"spec = importlib.util.spec_from_file_location('m', '{SCENE_FILE}'); "
            f"m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); "
            f"assert hasattr(m, 'SharedVectorSpace')",
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


def test_when_construct_called_then_axes_present(
    fake_data_dir: Path,
) -> None:
    """The 3D axes must be present in the scene."""
    mod = _load_scene_module()
    scene = _ThreeDScene()
    with (
        patch.object(mod, "Text", _Text),
        patch.object(common_mod, "Text", _Text),
        patch.object(common_mod, "FadeIn", _FadeIn),
        patch.object(common_mod, "Text", _Text),
        patch.object(common_mod, "FadeIn", _FadeIn),
        patch.object(mod, "ThreeDAxes", _ThreeDAxes),
        patch.object(mod, "Dot3D", _Dot3D),
        patch.object(mod, "FadeIn", _FadeIn),
        patch.object(mod, "Group", _Group),
        patch.object(mod, "VGroup", _VGroup),
    ):
        mod.build_shared_vector_space(scene, data_dir=fake_data_dir)
    axes = [m for m in _unwrap_mobjects(scene) if isinstance(m, _ThreeDAxes)]
    assert len(axes) >= 1, f"expected >= 1 ThreeDAxes, got {len(axes)}"


def test_when_construct_called_then_blue_and_orange_dots_present(
    fake_data_dir: Path,
) -> None:
    """5 text dots (blue) and 1 image dot (orange) must be in the scene."""
    mod = _load_scene_module()
    scene = _ThreeDScene()
    with (
        patch.object(mod, "Text", _Text),
        patch.object(common_mod, "Text", _Text),
        patch.object(common_mod, "FadeIn", _FadeIn),
        patch.object(mod, "ThreeDAxes", _ThreeDAxes),
        patch.object(mod, "Dot3D", _Dot3D),
        patch.object(mod, "FadeIn", _FadeIn),
        patch.object(mod, "Group", _Group),
        patch.object(mod, "VGroup", _VGroup),
    ):
        mod.build_shared_vector_space(scene, data_dir=fake_data_dir)
    dot3ds = [m for m in _unwrap_mobjects(scene) if isinstance(m, _Dot3D)]
    assert len(dot3ds) == 6, f"expected 6 Dot3D (5 text + 1 image), got {len(dot3ds)}"


def test_when_construct_called_then_italian_caption_present(
    fake_data_dir: Path,
) -> None:
    """The 'Tutto nello stesso spazio' caption must be visible."""
    mod = _load_scene_module()
    scene = _ThreeDScene()
    with (
        patch.object(mod, "Text", _Text),
        patch.object(common_mod, "Text", _Text),
        patch.object(common_mod, "FadeIn", _FadeIn),
        patch.object(mod, "ThreeDAxes", _ThreeDAxes),
        patch.object(mod, "Dot3D", _Dot3D),
        patch.object(mod, "FadeIn", _FadeIn),
        patch.object(mod, "Group", _Group),
        patch.object(mod, "VGroup", _VGroup),
    ):
        mod.build_shared_vector_space(scene, data_dir=fake_data_dir)
    texts = _all_text_strings(scene)
    assert any("Tutto nello stesso spazio" in s for s in texts), (
        f"missing 'Tutto nello stesso spazio' caption. Got: {texts}"
    )


def test_when_construct_called_then_italian_axis_labels_present(
    fake_data_dir: Path,
) -> None:
    """The X/Y/Z axes must be labeled in Italian: 'Dimensione 1/2/3'."""
    mod = _load_scene_module()
    scene = _ThreeDScene()
    with (
        patch.object(mod, "Text", _Text),
        patch.object(common_mod, "Text", _Text),
        patch.object(common_mod, "FadeIn", _FadeIn),
        patch.object(mod, "ThreeDAxes", _ThreeDAxes),
        patch.object(mod, "Dot3D", _Dot3D),
        patch.object(mod, "FadeIn", _FadeIn),
        patch.object(mod, "Group", _Group),
        patch.object(mod, "VGroup", _VGroup),
    ):
        mod.build_shared_vector_space(scene, data_dir=fake_data_dir)
    texts = _all_text_strings(scene)
    for label in ("Dimensione 1", "Dimensione 2", "Dimensione 3"):
        assert any(label in t for t in texts), (
            f"missing axis label {label!r}. Got: {texts}"
        )


def test_when_construct_called_then_camera_orientation_set(
    fake_data_dir: Path,
) -> None:
    """The scene must call set_camera_orientation / move_camera at least once."""
    mod = _load_scene_module()
    scene = _ThreeDScene()
    with (
        patch.object(mod, "Text", _Text),
        patch.object(common_mod, "Text", _Text),
        patch.object(common_mod, "FadeIn", _FadeIn),
        patch.object(mod, "ThreeDAxes", _ThreeDAxes),
        patch.object(mod, "Dot3D", _Dot3D),
        patch.object(mod, "FadeIn", _FadeIn),
        patch.object(mod, "Group", _Group),
        patch.object(mod, "VGroup", _VGroup),
    ):
        mod.build_shared_vector_space(scene, data_dir=fake_data_dir)
    assert len(scene.camera_moves) >= 1, (
        f"expected >= 1 camera move for 3D effect, got {len(scene.camera_moves)}"
    )


# --- Pure-data helpers -----------------------------------------------------


def test_when_load_pca_coords_called_then_shape_6_3() -> None:
    mod = _load_scene_module()
    c = mod.load_pca_coords()
    assert c.shape == (6, 3), f"expected (6, 3), got {c.shape}"
    assert c.dtype == np.float32, f"expected float32, got {c.dtype}"


def test_when_load_pca_labels_called_then_returns_dict_with_required_keys() -> None:
    mod = _load_scene_module()
    labels = mod.load_pca_labels()
    assert "modality" in labels
    assert "label" in labels
    assert "colors" in labels
    # text + image counts
    assert sum(1 for m in labels["modality"] if m == "text") >= 1
    assert sum(1 for m in labels["modality"] if m == "image") >= 1


def test_when_truncate_label_called_then_short_text_unchanged() -> None:
    mod = _load_scene_module()
    assert mod.truncate_label("gatto", max_len=20) == "gatto"


def test_when_truncate_label_called_then_long_text_truncated_with_ellipsis() -> None:
    mod = _load_scene_module()
    long = "a" * 50
    out = mod.truncate_label(long, max_len=20)
    assert len(out) <= 20, f"expected <= 20 chars, got {len(out)}"
    assert out.endswith("...") or out.endswith("…"), f"expected ellipsis, got {out!r}"


def test_when_scale_pca_coords_called_then_factor_applied() -> None:
    """PCA coords (range ~ ±1) are scaled to Manim's coordinate system."""
    mod = _load_scene_module()
    coords = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    out = mod.scale_pca_coords(coords, factor=2.0)
    assert out.shape == (2, 3)
    np.testing.assert_array_equal(out, np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]))


def test_when_color_for_modality_called_then_returns_expected_hex() -> None:
    mod = _load_scene_module()
    colors = {"text": "#3B82F6", "image": "#F97316"}
    assert mod.color_for_modality("text", colors) == "#3B82F6"
    assert mod.color_for_modality("image", colors) == "#F97316"
