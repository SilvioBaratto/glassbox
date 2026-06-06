"""Structural tests for the Glassbox shared-space extractor (issue #6).

HuggingFace ``CLIPModel``/``CLIPProcessor`` and ``sklearn.decomposition.PCA``
are mocked to keep the suite hermetic. The real extractor runs once
offline (CLIP-base ≈ 600MB) and is exercised by ``--help`` and manual runs.
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
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
EXTRACTOR = ROOT / "extract" / "extract_shared_space.py"
DATA_DIR = ROOT / "data"
TEXT_PATH = DATA_DIR / "shared_text_embeds.npy"
IMAGE_PATH = DATA_DIR / "shared_image_embeds.npy"
PCA_PATH = DATA_DIR / "pca_coords_3d.npy"
LABELS_PATH = DATA_DIR / "pca_labels.json"
SAMPLE_IMAGE = ROOT / "assets" / "sample_image.jpg"
SCRIPT_PATH = ROOT / "script.md"


def _load_extractor_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "_extract_shared_space_under_test",
        EXTRACTOR,
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --- Fake CLIP + PCA -------------------------------------------------------


class _FakeCLIPOutput:
    def __init__(self, text_embeds: torch.Tensor, image_embeds: torch.Tensor) -> None:
        self.text_embeds = text_embeds
        self.image_embeds = image_embeds


class _FakeCLIPModel:
    """Returns deterministic L2-normalised embeddings matching CLIP's contract."""

    def __init__(self, embed_seed: int = 11) -> None:
        self._seed = embed_seed

    def eval(self) -> "_FakeCLIPModel":
        return self

    def __call__(self, **kwargs: Any) -> _FakeCLIPOutput:
        # The fake receives whatever the processor produced, but we ignore
        # its exact contents and emit deterministic embeddings.
        gen_t = torch.Generator().manual_seed(self._seed)
        gen_i = torch.Generator().manual_seed(self._seed + 1)
        # 5 text embeds (matches the 5 prompts in the fake test)
        text = torch.randn(5, 512, generator=gen_t, dtype=torch.float32)
        text = text / text.norm(dim=1, keepdim=True)
        # 1 image embed
        image = torch.randn(1, 512, generator=gen_i, dtype=torch.float32)
        image = image / image.norm(dim=1, keepdim=True)
        return _FakeCLIPOutput(text_embeds=text, image_embeds=image)


class _FakeTokenizer:
    def __call__(self, text: list[str], **kwargs: Any) -> dict[str, Any]:
        # Return a fake input_ids tensor of shape (N, 77)
        n = len(text)
        gen = torch.Generator().manual_seed(n)
        ids = torch.randint(0, 49000, (n, 77), generator=gen, dtype=torch.int64)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


class _FakeImageProcessor:
    def __call__(self, images: Any, return_tensors: str = "pt") -> dict[str, Any]:
        return {"pixel_values": torch.zeros(1, 3, 224, 224, dtype=torch.float32)}


class _FakeProcessor:
    """Stub that exposes .tokenizer and .image_processor like real CLIPProcessor."""

    def __init__(self) -> None:
        self.tokenizer = _FakeTokenizer()
        self.image_processor = _FakeImageProcessor()
        self.calls: list[dict[str, Any]] = []


@pytest.fixture
def fake_image_path(tmp_path: Path) -> Path:
    """Side-effect-free 100x100 test image in tmp_path."""
    img = Image.new("RGB", (100, 100), color=(50, 150, 200))
    p = tmp_path / "test_image.jpg"
    img.save(p)
    return p


# --- Subprocess smoke ------------------------------------------------------


def test_when_extractor_help_invoked_then_exits_zero() -> None:
    proc = subprocess.run(
        [sys.executable, str(EXTRACTOR), "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert proc.returncode == 0, f"--help failed: {proc.stderr}"


def test_when_extractor_help_invoked_then_required_cli_args_listed() -> None:
    proc = subprocess.run(
        [sys.executable, str(EXTRACTOR), "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    for flag in ("--image", "--out-dir", "--script"):
        assert flag in proc.stdout, f"--help missing required flag {flag}"


# --- Module surface --------------------------------------------------------


def test_when_extractor_imported_then_run_callable_exists() -> None:
    mod = _load_extractor_module()
    assert callable(getattr(mod, "main", None))
    assert callable(getattr(mod, "run", None))


# --- shared_text_embeds ----------------------------------------------------


def test_when_extractor_run_then_shared_text_embeds_shape_and_dtype(
    tmp_path: Path, fake_image_path: Path
) -> None:
    out_dir = tmp_path / "data"
    mod = _load_extractor_module()
    fake_model = _FakeCLIPModel()
    fake_proc = _FakeProcessor()
    fake_texts = ["gatto", "un cane", "il sole", "una montagna", "un cielo azzurro"]
    with (
        patch.object(mod, "CLIPModel") as MockModel,
        patch.object(mod, "CLIPProcessor") as MockProc,
        patch.object(mod, "parse_script_text", return_value=fake_texts),
        patch.object(mod, "_pca_projection") as MockPCA,
    ):
        MockModel.from_pretrained.return_value = fake_model
        MockProc.from_pretrained.return_value = fake_proc
        # PCA must be called with random_state=0 for determinism
        MockPCA.return_value = np.zeros((6, 3), dtype=np.float32)
        mod.run(image_path=fake_image_path, out_dir=out_dir)

    text = np.load(out_dir / "shared_text_embeds.npy")
    assert text.shape == (5, 512), f"expected (5, 512), got {text.shape}"
    assert text.dtype == np.float32, f"expected float32, got {text.dtype}"


# --- shared_image_embeds ---------------------------------------------------


def test_when_extractor_run_then_shared_image_embeds_shape_and_dtype(
    tmp_path: Path, fake_image_path: Path
) -> None:
    out_dir = tmp_path / "data"
    mod = _load_extractor_module()
    fake_model = _FakeCLIPModel()
    fake_proc = _FakeProcessor()
    fake_texts = ["gatto", "un cane", "il sole", "una montagna", "un cielo azzurro"]
    with (
        patch.object(mod, "CLIPModel") as MockModel,
        patch.object(mod, "CLIPProcessor") as MockProc,
        patch.object(mod, "parse_script_text", return_value=fake_texts),
        patch.object(mod, "_pca_projection") as MockPCA,
    ):
        MockModel.from_pretrained.return_value = fake_model
        MockProc.from_pretrained.return_value = fake_proc
        MockPCA.return_value = np.zeros((6, 3), dtype=np.float32)
        mod.run(image_path=fake_image_path, out_dir=out_dir)

    img = np.load(out_dir / "shared_image_embeds.npy")
    assert img.shape == (1, 512), f"expected (1, 512), got {img.shape}"
    assert img.dtype == np.float32, f"expected float32, got {img.dtype}"


# --- pca_coords_3d ---------------------------------------------------------


def test_when_extractor_run_then_pca_coords_3d_shape_and_centred(
    tmp_path: Path, fake_image_path: Path
) -> None:
    """PCA must produce (N_text + N_img, 3) float32, centred around 0."""
    out_dir = tmp_path / "data"
    mod = _load_extractor_module()
    fake_model = _FakeCLIPModel()
    fake_proc = _FakeProcessor()
    fake_texts = ["gatto", "un cane", "il sole", "una montagna", "un cielo azzurro"]
    with (
        patch.object(mod, "CLIPModel") as MockModel,
        patch.object(mod, "CLIPProcessor") as MockProc,
        patch.object(mod, "parse_script_text", return_value=fake_texts),
        patch.object(mod, "_pca_projection") as MockPCA,
    ):
        MockModel.from_pretrained.return_value = fake_model
        MockProc.from_pretrained.return_value = fake_proc
        # Mock PCA to return a non-zero array so centring is observable
        MockPCA.return_value = np.array(
            [[1, 2, 3], [4, 5, 6], [7, 8, 9], [-1, -2, -3], [-4, -5, -6], [-7, -8, -9]],
            dtype=np.float32,
        )
        mod.run(image_path=fake_image_path, out_dir=out_dir)

    pca = np.load(out_dir / "pca_coords_3d.npy")
    assert pca.shape == (6, 3), f"expected (6, 3), got {pca.shape}"
    assert pca.dtype == np.float32, f"expected float32, got {pca.dtype}"
    # Centred: mean per axis ~ 0
    mean_per_axis = pca.mean(axis=0)
    assert np.allclose(mean_per_axis, 0, atol=1e-6), (
        f"PCA not centred: mean per axis = {mean_per_axis}"
    )


# --- pca_labels.json schema (reviewer's bug fix) --------------------------


def test_when_extractor_run_then_pca_labels_json_schema(
    tmp_path: Path, fake_image_path: Path
) -> None:
    """The labels JSON must include modality, label, AND a top-level color map.

    Reviewer's bug fix: scenes 05 and 11 need stable color choices; embed
    the modality->color mapping in the JSON so scenes don't drift.
    """
    out_dir = tmp_path / "data"
    mod = _load_extractor_module()
    fake_model = _FakeCLIPModel()
    fake_proc = _FakeProcessor()
    fake_texts = ["gatto", "un cane", "il sole", "una montagna", "un cielo azzurro"]
    with (
        patch.object(mod, "CLIPModel") as MockModel,
        patch.object(mod, "CLIPProcessor") as MockProc,
        patch.object(mod, "parse_script_text", return_value=fake_texts),
        patch.object(mod, "_pca_projection") as MockPCA,
    ):
        MockModel.from_pretrained.return_value = fake_model
        MockProc.from_pretrained.return_value = fake_proc
        MockPCA.return_value = np.zeros((6, 3), dtype=np.float32)
        mod.run(image_path=fake_image_path, out_dir=out_dir)

    labels = json.loads((out_dir / "pca_labels.json").read_text())
    # Top-level schema
    assert set(labels.keys()) >= {"modality", "label", "colors"}, (
        f"pca_labels.json missing top-level keys: {labels.keys()}"
    )
    assert len(labels["modality"]) == 6
    assert len(labels["label"]) == 6
    # First 5 are text, last 1 is image
    assert all(m == "text" for m in labels["modality"][:5])
    assert labels["modality"][5] == "image"
    # Labels match the input texts + filename
    assert labels["label"][:5] == fake_texts
    # Color map
    assert "text" in labels["colors"] and "image" in labels["colors"]
    # Hex strings
    for v in labels["colors"].values():
        assert v.startswith("#") and len(v) == 7, f"bad hex color: {v}"


# --- Determinism: PCA random_state=0 --------------------------------------


def test_when_extractor_run_then_pca_called_with_random_state_zero(
    tmp_path: Path, fake_image_path: Path
) -> None:
    """The reviewer flagged determinism: PCA must be called with random_state=0."""
    out_dir = tmp_path / "data"
    mod = _load_extractor_module()
    fake_model = _FakeCLIPModel()
    fake_proc = _FakeProcessor()
    fake_texts = ["gatto", "un cane", "il sole", "una montagna", "un cielo azzurro"]
    with (
        patch.object(mod, "CLIPModel") as MockModel,
        patch.object(mod, "CLIPProcessor") as MockProc,
        patch.object(mod, "parse_script_text", return_value=fake_texts),
        patch.object(mod, "_pca_projection") as MockPCA,
    ):
        MockModel.from_pretrained.return_value = fake_model
        MockProc.from_pretrained.return_value = fake_proc
        MockPCA.return_value = np.zeros((6, 3), dtype=np.float32)
        mod.run(image_path=fake_image_path, out_dir=out_dir)
    # _pca_projection was called once
    assert MockPCA.call_count == 1
    # The PCA random_state must be 0 (passed as keyword to sklearn PCA)
    _, kwargs = MockPCA.call_args
    assert kwargs.get("random_state") == 0, (
        f"PCA random_state must be 0 for determinism, got {kwargs.get('random_state')}"
    )


# --- Idempotency ----------------------------------------------------------


def test_when_extractor_run_twice_then_outputs_identical(
    tmp_path: Path, fake_image_path: Path
) -> None:
    out_dir = tmp_path / "data"
    mod = _load_extractor_module()
    fake_model = _FakeCLIPModel()
    fake_proc = _FakeProcessor()
    fake_texts = ["gatto", "un cane", "il sole", "una montagna", "un cielo azzurro"]
    with (
        patch.object(mod, "CLIPModel") as MockModel,
        patch.object(mod, "CLIPProcessor") as MockProc,
        patch.object(mod, "parse_script_text", return_value=fake_texts),
        patch.object(mod, "_pca_projection") as MockPCA,
    ):
        MockModel.from_pretrained.return_value = fake_model
        MockProc.from_pretrained.return_value = fake_proc
        MockPCA.return_value = np.zeros((6, 3), dtype=np.float32)
        mod.run(image_path=fake_image_path, out_dir=out_dir)
        first = (out_dir / "shared_text_embeds.npy").read_bytes()
        mod.run(image_path=fake_image_path, out_dir=out_dir)
        second = (out_dir / "shared_text_embeds.npy").read_bytes()
    assert first == second, "re-running extractor must produce identical output"
