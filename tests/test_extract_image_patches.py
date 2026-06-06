"""Structural tests for the Glassbox image patch extractor (issue #4).

The HuggingFace ViTModel + ViTImageProcessor are mocked to keep the suite
hermetic and deterministic. The real extractor is exercised by its
``--help`` smoke test (and by manual `python extract/extract_image_patches.py`
runs).
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
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
EXTRACTOR = ROOT / "extract" / "extract_image_patches.py"
DATA_DIR = ROOT / "data"
PATCH_PATH = DATA_DIR / "patch_embeddings.npy"
GRID_PATH = DATA_DIR / "patch_grid.npy"
PREPROC_PATH = DATA_DIR / "sample_image_224.npy"


def _load_extractor_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "_extract_image_patches_under_test",
        EXTRACTOR,
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --- Fake ViT --------------------------------------------------------------


class _FakeImageProcessor:
    """Stand-in for ViTImageProcessor: just records the call and resizes."""

    def __init__(self, size: int = 224) -> None:
        self.size = size
        self.calls: list[Any] = []

    def __call__(self, images: Any, return_tensors: str = "pt") -> Any:
        # Record the raw image so we can verify the caller did NOT pre-resize
        self.calls.append(
            {
                "mode": getattr(images, "mode", None),
                "size": getattr(images, "size", None),
            }
        )
        # Pretend to do resize+normalize — return a deterministic tensor
        if hasattr(images, "convert"):
            arr = (
                np.asarray(
                    images.convert("RGB").resize(
                        (self.size, self.size), Image.Resampling.BILINEAR
                    ),
                    dtype=np.float32,
                )
                / 255.0
            )
        else:
            arr = np.zeros((self.size, self.size, 3), dtype=np.float32)
        # Reorder HWC -> CHW, normalise
        chw = arr.transpose(2, 0, 1)[None, ...]
        return {"pixel_values": torch.from_numpy(chw)}


class _FakeViTConfig:
    hidden_size = 768
    patch_size = 16
    image_size = 224


class _FakeViTOutput:
    def __init__(self, last_hidden_state: torch.Tensor) -> None:
        self.last_hidden_state = last_hidden_state


class _FakeViTModel:
    """Stand-in for ViTModel that returns a deterministic last_hidden_state."""

    config = _FakeViTConfig()

    def __init__(self, embed_seed: int = 1234) -> None:
        self._seed = embed_seed

    def eval(self) -> "_FakeViTModel":  # mimic torch.nn.Module API
        return self

    def __call__(
        self, *, pixel_values: torch.Tensor, output_hidden_states: bool = True
    ) -> _FakeViTOutput:  # noqa: ARG002
        b, _c, h, w = pixel_values.shape
        assert h == 224 and w == 224
        n_patches = (h // 16) * (w // 16)  # 196
        seq_len = n_patches + 1  # + CLS token
        gen = torch.Generator().manual_seed(self._seed)
        # CLS token at index 0, patch tokens at indices 1..196
        last = torch.randn(b, seq_len, 768, generator=gen, dtype=torch.float32)
        return _FakeViTOutput(last_hidden_state=last)


# --- Helper: build a real test image in tmp --------------------------------


def _make_test_image(path: Path, size: tuple[int, int] = (300, 200)) -> Path:
    """Write a non-224 test image and return the path. Side-effect-free."""
    img = Image.new("RGB", size, color=(120, 80, 200))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return path


@pytest.fixture
def test_image_path(tmp_path: Path) -> Path:
    """Side-effect-free 300x200 test image in tmp_path (does not touch the real asset)."""
    return _make_test_image(tmp_path / "test_input.jpg", size=(300, 200))


# --- Subprocess smoke (no model download) ----------------------------------


def test_when_extractor_help_invoked_then_exits_zero() -> None:
    proc = subprocess.run(
        [sys.executable, str(EXTRACTOR), "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert proc.returncode == 0, (
        f"--help failed: rc={proc.returncode}\nstderr={proc.stderr}"
    )


# --- Module surface --------------------------------------------------------


def test_when_extractor_imported_then_run_callable_exists() -> None:
    mod = _load_extractor_module()
    assert callable(getattr(mod, "main", None))
    assert callable(getattr(mod, "run", None))


# --- Patch embeddings shape and dtype --------------------------------------


def test_when_extractor_run_then_patch_embeddings_shape_and_dtype(
    tmp_path: Path, test_image_path: Path
) -> None:
    out_dir = tmp_path / "data"
    mod = _load_extractor_module()
    fake_proc = _FakeImageProcessor()
    fake_model = _FakeViTModel()
    with (
        patch.object(mod, "ViTImageProcessor") as MockProc,
        patch.object(mod, "ViTModel") as MockModel,
    ):
        MockProc.from_pretrained.return_value = fake_proc
        MockModel.from_pretrained.return_value = fake_model
        rc = mod.run(image_path=test_image_path, out_dir=out_dir)
    assert rc == 0

    arr = np.load(out_dir / "patch_embeddings.npy")
    assert arr.shape == (196, 768), f"expected (196, 768), got {arr.shape}"
    assert arr.dtype == np.float32, f"expected float32, got {arr.dtype}"


# --- patch_grid: 14x14 of indices ----------------------------------------


def test_when_extractor_run_then_patch_grid_is_14x14_indices(
    tmp_path: Path, test_image_path: Path
) -> None:
    out_dir = tmp_path / "data"
    mod = _load_extractor_module()
    fake_proc = _FakeImageProcessor()
    fake_model = _FakeViTModel()
    with (
        patch.object(mod, "ViTImageProcessor") as MockProc,
        patch.object(mod, "ViTModel") as MockModel,
    ):
        MockProc.from_pretrained.return_value = fake_proc
        MockModel.from_pretrained.return_value = fake_model
        mod.run(image_path=test_image_path, out_dir=out_dir)
    grid = np.load(out_dir / "patch_grid.npy")
    assert grid.shape == (14, 14), f"expected (14, 14), got {grid.shape}"
    assert grid.dtype in (np.int32, np.int64), f"expected int dtype, got {grid.dtype}"
    assert int(grid.min()) == 0 and int(grid.max()) == 195, (
        f"expected indices 0..195, got min={grid.min()} max={grid.max()}"
    )


# --- sample_image_224: uint8 HWC for display ------------------------------


def test_when_extractor_run_then_sample_image_224_is_uint8_hwc(
    tmp_path: Path, test_image_path: Path
) -> None:
    """The displayable sidecar must be uint8 HWC 224x224x3, not normalised CHW.

    This is the reviewer's bug fix from issue #4: post-normalization CHW
    float would render as black in Manim. Save the resized original.
    """
    out_dir = tmp_path / "data"
    mod = _load_extractor_module()
    fake_proc = _FakeImageProcessor()
    fake_model = _FakeViTModel()
    with (
        patch.object(mod, "ViTImageProcessor") as MockProc,
        patch.object(mod, "ViTModel") as MockModel,
    ):
        MockProc.from_pretrained.return_value = fake_proc
        MockModel.from_pretrained.return_value = fake_model
        mod.run(image_path=test_image_path, out_dir=out_dir)
    img_arr = np.load(out_dir / "sample_image_224.npy")
    assert img_arr.shape == (224, 224, 3), (
        f"sample_image_224 must be HWC uint8 shape (224, 224, 3), got {img_arr.shape}"
    )
    assert img_arr.dtype == np.uint8, (
        f"sample_image_224 must be uint8 for direct Manim display, got {img_arr.dtype}"
    )


# --- Processor is called with the raw image (no double-resize) ------------


def test_when_extractor_run_then_processor_receives_raw_image(
    tmp_path: Path, test_image_path: Path
) -> None:
    """The agent must NOT do a manual Pillow resize before calling the processor.

    Our test image is 300x200 (non-square) to make this check meaningful.
    The processor must receive the same 300x200 image we passed in, not a
    pre-resized 224x224.
    """
    out_dir = tmp_path / "data"
    mod = _load_extractor_module()
    fake_proc = _FakeImageProcessor()
    fake_model = _FakeViTModel()
    with (
        patch.object(mod, "ViTImageProcessor") as MockProc,
        patch.object(mod, "ViTModel") as MockModel,
    ):
        MockProc.from_pretrained.return_value = fake_proc
        MockModel.from_pretrained.return_value = fake_model
        mod.run(image_path=test_image_path, out_dir=out_dir)
    assert len(fake_proc.calls) == 1
    recorded = fake_proc.calls[0]
    real = Image.open(test_image_path)
    assert recorded["size"] == real.size, (
        f"processor received {recorded['size']}, file is {real.size}; "
        "double-resize detected"
    )


# --- Idempotency ----------------------------------------------------------


def test_when_extractor_run_twice_then_outputs_identical(
    tmp_path: Path, test_image_path: Path
) -> None:
    out_dir = tmp_path / "data"
    mod = _load_extractor_module()
    fake_proc = _FakeImageProcessor()
    fake_model = _FakeViTModel()
    with (
        patch.object(mod, "ViTImageProcessor") as MockProc,
        patch.object(mod, "ViTModel") as MockModel,
    ):
        MockProc.from_pretrained.return_value = fake_proc
        MockModel.from_pretrained.return_value = fake_model
        mod.run(image_path=test_image_path, out_dir=out_dir)
        first = (out_dir / "patch_embeddings.npy").read_bytes()
        mod.run(image_path=test_image_path, out_dir=out_dir)
        second = (out_dir / "patch_embeddings.npy").read_bytes()
    assert first == second, "re-running extractor must produce identical output"


# --- CLI args present -----------------------------------------------------


def test_when_extractor_help_invoked_then_required_cli_args_listed() -> None:
    proc = subprocess.run(
        [sys.executable, str(EXTRACTOR), "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    for flag in ("--image", "--out-dir"):
        assert flag in proc.stdout, f"--help missing required flag {flag}"
