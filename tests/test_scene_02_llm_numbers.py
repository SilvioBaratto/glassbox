"""Structural tests for scene 02 (issue #8) — LLM number pipeline.

We test the orchestration logic (data loading, attention matrix shape,
animation calls) without invoking Manim's render loop. Mirrors the
pattern from test_scene_01_tokenization.py.
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
SCENE_FILE = ROOT / "scenes" / "02_llm_numbers.py"
DATA_DIR = ROOT / "data"
TOKENS_PATH = DATA_DIR / "tokens.npy"
STRINGS_PATH = DATA_DIR / "token_strings.json"


def _load_scene_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "_scene_02_llm_numbers_under_test",
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
    data_dir: Path, rows: list[list[int]], strings: list[list[str]]
) -> None:
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

    Row 0 mirrors the real "Come si fa a capire tutto?" with 9 content tokens
    so the heatmap can slice to 10 non-pad positions (T=9 actually; we'll
    pad to T=10 in the scene).
    """
    out = tmp_path / "data"
    max_len = 77
    pad_id = 49407
    content_ids = [891, 2990, 2800, 320, 1289, 1454, 764, 8105, 286]
    pad = [pad_id] * (max_len - 1 - len(content_ids))
    rows = [[49406] + content_ids + pad]
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
    ]
    _write_fake_data(out, rows, strings)
    return out


# --- Manim mobject mocks (mirror of test_scene_01_tokenization) ------------


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


class _MathTex(_Mobject):
    """MathTex(text=...)"""


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
    """Recursively unwrap FadeIn.mobject and Group.mobjects."""
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
    return [
        str(m.args[0]) if m.args else ""
        for m in _unwrap_mobjects(scene)
        if isinstance(m, _Text)
    ]


# --- Subprocess smoke ------------------------------------------------------


def test_when_scene_loaded_as_module_then_LLMNumbers_class_exists() -> None:
    mod = _load_scene_module()
    cls = getattr(mod, "LLMNumbers", None)
    assert cls is not None, (
        "scenes/02_llm_numbers.py must define class LLMNumbers(Scene)"
    )


def test_when_scene_loaded_then_it_subclasses_manim_Scene() -> None:
    """LLMNumbers must inherit from manim.Scene."""
    import manim

    mod = _load_scene_module()
    cls = mod.LLMNumbers
    assert issubclass(cls, manim.Scene), "LLMNumbers must subclass manim.Scene"


def test_when_scene_imported_then_no_syntax_errors() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            f"import importlib.util; "
            f"spec = importlib.util.spec_from_file_location('m', '{SCENE_FILE}'); "
            f"m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); "
            f"assert hasattr(m, 'LLMNumbers')",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert proc.returncode == 0, (
        f"scene import failed: rc={proc.returncode}\nstderr={proc.stderr}"
    )


# --- File size guard -------------------------------------------------------


def test_when_scene_file_inspected_then_under_500_lines() -> None:
    lines = SCENE_FILE.read_text().count("\n")
    assert lines < 500, f"scene file too long: {lines} lines (limit 500)"


# --- Orchestration (mocked rendering) -------------------------------------


def test_when_construct_called_then_data_loaded_from_provided_paths(
    fake_data_dir: Path,
) -> None:
    """The scene must read from data/tokens.npy and data/token_strings.json.

    We redirect the loader via dependency injection (data_dir kwarg).
    """
    mod = _load_scene_module()
    scene = _FakeScene()
    with (
        patch.object(mod, "Text", _Text),
        patch.object(common_mod, "Text", _Text),
        patch.object(common_mod, "FadeIn", _FadeIn),
        patch.object(common_mod, "Text", _Text),
        patch.object(common_mod, "FadeIn", _FadeIn),
        patch.object(mod, "MathTex", _MathTex),
        patch.object(mod, "FadeIn", _FadeIn),
        patch.object(mod, "Group", _Group),
    ):
        mod.build_llm_numbers(scene, data_dir=fake_data_dir)
    # The scene must have rendered MathTex numbers
    math_texes = [m for m in _unwrap_mobjects(scene) if isinstance(m, _MathTex)]
    assert len(math_texes) >= 5, (
        f"expected >= 5 MathTex elements (token IDs), got {len(math_texes)}"
    )


def test_when_construct_called_then_token_ids_match_data(
    fake_data_dir: Path,
) -> None:
    """The integer IDs displayed must match the saved tokens.npy for the chosen row."""
    mod = _load_scene_module()
    scene = _FakeScene()
    with (
        patch.object(mod, "Text", _Text),
        patch.object(common_mod, "Text", _Text),
        patch.object(common_mod, "FadeIn", _FadeIn),
        patch.object(mod, "MathTex", _MathTex),
        patch.object(mod, "FadeIn", _FadeIn),
        patch.object(mod, "Group", _Group),
    ):
        mod.build_llm_numbers(scene, data_dir=fake_data_dir)
    tokens = np.load(fake_data_dir / "tokens.npy")
    expected_ids = set(int(i) for i in tokens[0, :10] if i not in (49406, 49407))
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


def test_when_construct_called_then_italian_caption_present(
    fake_data_dir: Path,
) -> None:
    """The 'Leggere il contesto' caption must be visible."""
    mod = _load_scene_module()
    scene = _FakeScene()
    with (
        patch.object(mod, "Text", _Text),
        patch.object(common_mod, "Text", _Text),
        patch.object(common_mod, "FadeIn", _FadeIn),
        patch.object(mod, "MathTex", _MathTex),
        patch.object(mod, "FadeIn", _FadeIn),
        patch.object(mod, "Group", _Group),
    ):
        mod.build_llm_numbers(scene, data_dir=fake_data_dir)
    texts = _all_text_strings(scene)
    assert any("Leggere il contesto" in s for s in texts), (
        f"missing 'Leggere il contesto' caption. Got: {texts}"
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


def test_when_construct_called_then_animation_count_keeps_pace(
    fake_data_dir: Path,
) -> None:
    """The scene must call ``scene.play`` for the AC's 4 steps.

    The reviewer's bug fix: with T=9 and an 8s budget at 30 fps, the
    highlight animation must be bounded. The scene uses:
      - 1 play for step 1 (token ID row)
      - T-1 plays for step 2 (highlight slides one per token)
      - 1 play for step 3 (heatmap reveal)
      - 5 plays for step 4 (5 evenly-spaced row highlights)
      - 1 play for the Italian caption
    Total: 1 + (T-1) + 1 + 5 + 1 = 16 (for T=9).
    We assert >= 4 (the AC's 4 steps) and <= 18 (room for slight wiggle,
    but still well within the 8s budget at 30 fps when each play
    is short).
    """
    mod = _load_scene_module()
    scene = _FakeScene()
    with (
        patch.object(mod, "Text", _Text),
        patch.object(common_mod, "Text", _Text),
        patch.object(common_mod, "FadeIn", _FadeIn),
        patch.object(mod, "MathTex", _MathTex),
        patch.object(mod, "FadeIn", _FadeIn),
        patch.object(mod, "Group", _Group),
    ):
        mod.build_llm_numbers(scene, data_dir=fake_data_dir)
    assert len(scene.played) >= 4, (
        f"expected >= 4 scene.play() calls (one per AC step), got {len(scene.played)}"
    )
    assert len(scene.played) <= 18, (
        f"too many play() calls ({len(scene.played)}); reviewer's 8s budget fix "
        "requires bounded highlight animation, not every row"
    )


# --- Pure-data helpers -----------------------------------------------------


def test_when_attention_matrix_built_then_shape_matches_sequence_length() -> None:
    """The attention matrix must be (T, T) where T is the chosen sequence length."""
    mod = _load_scene_module()
    # Build a fake sequence of 10 token IDs
    seq = list(range(100, 110))
    attn = mod.build_attention_matrix(seq, seed=42)
    assert attn.shape == (10, 10), f"expected (10, 10), got {attn.shape}"
    assert attn.dtype == np.float32, f"expected float32, got {attn.dtype}"


def test_when_attention_matrix_built_then_values_in_zero_one_range() -> None:
    """Attention values must be in [0, 1] (opacity-friendly)."""
    mod = _load_scene_module()
    seq = list(range(100, 110))
    attn = mod.build_attention_matrix(seq, seed=42)
    assert attn.min() >= 0.0, f"attention min < 0: {attn.min()}"
    assert attn.max() <= 1.0, f"attention max > 1: {attn.max()}"


def test_when_attention_matrix_built_then_deterministic() -> None:
    """Same seed + sequence must produce identical matrix (reviewer requirement)."""
    mod = _load_scene_module()
    seq = list(range(100, 110))
    a1 = mod.build_attention_matrix(seq, seed=42)
    a2 = mod.build_attention_matrix(seq, seed=42)
    np.testing.assert_array_equal(a1, a2)


def test_when_pick_sequence_called_then_returns_first_n_non_pad() -> None:
    """The sequence picker must return the first n non-special token IDs."""
    mod = _load_scene_module()
    # 49406=start, 49407=end/pad
    tokens = np.array([49406, 100, 101, 102, 49407, 49407, 49407], dtype=np.int64)
    seq = mod.pick_sequence(tokens, row_idx=0, n=3)
    assert seq == [100, 101, 102], f"expected [100, 101, 102], got {seq}"


def test_when_pick_sequence_called_then_skips_49407_padding() -> None:
    """Reviewer's bug fix: must skip pad IDs even at non-leading positions."""
    mod = _load_scene_module()
    tokens = np.array([49406, 100, 49407, 101, 102, 49407, 49407], dtype=np.int64)
    seq = mod.pick_sequence(tokens, row_idx=0, n=3)
    # Should skip the leading 49407 and pick 100, 101, 102
    assert seq == [100, 101, 102], f"expected [100, 101, 102] (skip pad), got {seq}"
