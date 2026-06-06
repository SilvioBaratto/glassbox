"""Structural tests for scene 01 (issue #7) — tokenization.

We test the orchestration logic (data loading, picking the BPE-split
example, displaying correct integer IDs) without invoking Manim's render
loop (which produces a video and takes 30+ seconds). Manim mobject
creators are mocked so the suite stays fast and hermetic.
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
SCENE_FILE = ROOT / "scenes" / "01_tokenization.py"
DATA_DIR = ROOT / "data"
TOKENS_PATH = DATA_DIR / "tokens.npy"
STRINGS_PATH = DATA_DIR / "token_strings.json"


def _load_scene_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "_scene_01_tokenization_under_test",
        SCENE_FILE,
    )
    assert spec is not None and spec.loader is not None, (
        f"Cannot load {SCENE_FILE} as a module"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --- Fake data builders ---------------------------------------------------


def _write_fake_data(
    data_dir: Path, rows: list[list[int]], strings: list[list[str]]
) -> None:
    """Write a fake ``tokens.npy`` and ``token_strings.json`` to a tmp dir."""
    data_dir.mkdir(parents=True, exist_ok=True)
    np.save(data_dir / "tokens.npy", np.asarray(rows, dtype=np.int64))
    (data_dir / "token_strings.json").write_text(
        json.dumps(
            {
                "model": "openai/clip-vit-base-patch32",
                "max_len": 77,
                "rows": strings,
            }
        )
    )


@pytest.fixture
def fake_data_dir(tmp_path: Path) -> Path:
    """Build a representative data dir for testing.

    Row 0: ['Come', 'si', 'fa', 'a', 'cap', '##ire', 'tut', '##to</w>', '?</w>']
    has BPE splits (cap/##ire, tut/##to). The scene must surface this.

    All rows are padded to length 77 with <|endoftext|> so np.save works
    on a (N, 77) array.
    """
    out = tmp_path / "data"
    max_len = 77
    pad_id = 49407  # CLIP end-of-text
    content_ids_row0 = [891, 2990, 2800, 320, 1289, 1454, 764, 8105, 286]
    content_ids_row1 = [
        535,
        534,
        pad_id,
        pad_id,
        pad_id,
        pad_id,
        pad_id,
        pad_id,
        pad_id,
    ]
    pad = [pad_id] * (max_len - 1 - len(content_ids_row0))
    rows = [
        [49406] + content_ids_row0 + pad,
        [49406] + content_ids_row1 + [pad_id] * (max_len - 1 - len(content_ids_row1)),
    ]
    strings = [
        [
            "<|startoftext|>",
            "come",
            "si",
            "fa",
            "a",
            "cap",
            "##ire",
            "tut",
            "##to</w>",
            "?</w>",
        ]
        + ["<|endoftext|>"] * (max_len - 10),
        ["<|startoftext|>", "un", "cane", "<|endoftext|>"]
        + ["<|endoftext|>"] * (max_len - 4),
    ]
    _write_fake_data(out, rows, strings)
    return out


# --- Manim mobject mocks --------------------------------------------------


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


class _MathTex(_Mobject):
    """MathTex(text=...)"""


class _FadeIn:
    """Stand-in for manim.FadeIn(mobject). Records the mobject argument."""

    def __init__(self, mobject: Any, *args: Any, **kwargs: Any) -> None:
        self.mobject = mobject
        self.args = args
        self.kwargs = kwargs


class _Group:
    """Stand-in for manim.Group. Records all submobjects in ``.mobjects``."""

    def __init__(self, *mobjects: Any, **kwargs: Any) -> None:
        self.mobjects = list(mobjects)
        self.kwargs = kwargs


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


# --- Subprocess smoke: scene file is syntactically valid Python -----------


def test_when_scene_loaded_as_module_then_tokenization_class_exists() -> None:
    mod = _load_scene_module()
    cls = getattr(mod, "Tokenization", None)
    assert cls is not None, (
        "scenes/01_tokenization.py must define class Tokenization(Scene)"
    )


def test_when_scene_loaded_then_it_subclasses_manim_Scene() -> None:
    """Tokenization must inherit from manim.Scene."""
    import manim

    mod = _load_scene_module()
    cls = mod.Tokenization
    assert issubclass(cls, manim.Scene), "Tokenization must subclass manim.Scene"


# --- construct() orchestration (mocked rendering) ------------------------


def _unwrap_mobjects(scene: _FakeScene) -> list[Any]:
    """Collect every Mobject that was constructed during a scene run.

    Mobjects are created with Text/MathTex and then either:
      - added to the scene via Scene.add() — lands in scene.added
      - wrapped in FadeIn(...) and passed to Scene.play() — lands in
        scene.played, and the mobject is in FadeIn.mobject
      - grouped via Group(...) before being wrapped in FadeIn — we
        recursively unwrap via .mobjects

    All paths are walked so the helper is robust to either style.
    """
    out: list[Any] = []
    stack: list[Any] = list(scene.added) + [
        anim.mobject if isinstance(anim, _FadeIn) else anim for anim in scene.played
    ]
    while stack:
        m = stack.pop()
        if isinstance(m, _Group):
            stack.extend(m.mobjects)
        else:
            out.append(m)
    return out


def _all_text_strings(scene: _FakeScene) -> list[str]:
    """Collect every text string that was constructed during a scene run."""
    return [
        str(m.args[0]) if m.args else ""
        for m in _unwrap_mobjects(scene)
        if isinstance(m, _Text)
    ]


def test_when_construct_called_then_data_loaded_from_provided_paths(
    fake_data_dir: Path,
) -> None:
    """The scene must read from data/tokens.npy and data/token_strings.json.

    We redirect the loader via dependency injection (data_dir kwarg) so the
    test doesn't need to read the real data/.
    """
    mod = _load_scene_module()
    scene = _FakeScene()
    with (
        patch.object(mod, "Text", _Text),
        patch.object(mod, "MathTex", _MathTex),
        patch.object(mod, "FadeIn", _FadeIn),
        patch.object(mod, "Group", _Group),
    ):
        mod.build_tokenization(scene, data_dir=fake_data_dir)
    texts = _all_text_strings(scene)
    assert any("gatto" in s for s in texts), f"raw 'gatto' text not in scene: {texts}"


def test_when_construct_called_then_bpe_subtokens_displayed(
    fake_data_dir: Path,
) -> None:
    """The scene must show BPE sub-tokens (reviewer's bug fix: real splits)."""
    mod = _load_scene_module()
    scene = _FakeScene()
    with (
        patch.object(mod, "Text", _Text),
        patch.object(mod, "MathTex", _MathTex),
        patch.object(mod, "FadeIn", _FadeIn),
        patch.object(mod, "Group", _Group),
    ):
        mod.build_tokenization(scene, data_dir=fake_data_dir)
    texts = _all_text_strings(scene)
    has_subtoken = any("##" in s for s in texts)
    assert has_subtoken, f"no BPE sub-tokens in scene. Got: {texts}"


def test_when_construct_called_then_integer_ids_match_tokens(
    fake_data_dir: Path,
) -> None:
    """The integer IDs displayed must match the saved tokens.npy for the chosen row."""
    mod = _load_scene_module()
    scene = _FakeScene()
    with (
        patch.object(mod, "Text", _Text),
        patch.object(mod, "MathTex", _MathTex),
        patch.object(mod, "FadeIn", _FadeIn),
        patch.object(mod, "Group", _Group),
    ):
        mod.build_tokenization(scene, data_dir=fake_data_dir)
    tokens = np.load(fake_data_dir / "tokens.npy")
    expected_ids = set(int(i) for i in tokens[0, :9] if i not in (49406, 49407))
    rendered_ids = set()
    for m in _unwrap_mobjects(scene):
        if isinstance(m, _MathTex):
            for arg in m.args:
                try:
                    rendered_ids.add(int(arg))
                except (ValueError, TypeError):
                    pass
    overlap = expected_ids & rendered_ids
    assert len(overlap) >= 5, (
        f"expected >= 5 token IDs in scene, got {len(overlap)}: "
        f"expected {expected_ids}, rendered {rendered_ids}"
    )


def test_when_construct_called_then_italian_captions_present(
    fake_data_dir: Path,
) -> None:
    """All visible text must be in Italian per the AC."""
    mod = _load_scene_module()
    scene = _FakeScene()
    with (
        patch.object(mod, "Text", _Text),
        patch.object(mod, "MathTex", _MathTex),
        patch.object(mod, "FadeIn", _FadeIn),
        patch.object(mod, "Group", _Group),
    ):
        mod.build_tokenization(scene, data_dir=fake_data_dir)
    texts = _all_text_strings(scene)
    assert any("Testo grezzo" in s for s in texts), (
        f"missing 'Testo grezzo' caption. Got: {texts}"
    )


def test_when_construct_called_then_blue_modality_color_used(
    fake_data_dir: Path,
) -> None:
    """Text (modality) must use the blue color #3B82F6 per AC."""
    mod = _load_scene_module()
    blue = getattr(mod, "COLOR_TEXT", None) or getattr(mod, "BLUE_TEXT", None)
    assert blue is not None, "scene must define a COLOR_TEXT / BLUE_TEXT constant"
    assert blue.upper() == "#3B82F6", (
        f"text color must be #3B82F6 per modality convention, got {blue}"
    )


# --- File size guard -----------------------------------------------------


def test_when_scene_file_inspected_then_under_500_lines() -> None:
    lines = SCENE_FILE.read_text().count("\n")
    assert lines < 500, f"scene file too long: {lines} lines (limit 500)"


# --- import smoke --------------------------------------------------------


def test_when_scene_imported_then_no_syntax_errors() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            f"import importlib.util, sys; "
            f"spec = importlib.util.spec_from_file_location('m', '{SCENE_FILE}'); "
            f"m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); "
            f"assert hasattr(m, 'Tokenization')",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert proc.returncode == 0, (
        f"scene import failed: rc={proc.returncode}\nstderr={proc.stderr}"
    )
