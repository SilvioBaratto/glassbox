"""Offline image patch extraction for Glassbox (issue #4).

Loads ``google/vit-base-patch16-224`` once, runs it on the sample image, and
saves the 196 patch embeddings (one per 16×16 patch in the 14×14 grid) plus
a displayable sidecar for scene 03.

Outputs in ``data/``:
  - ``patch_embeddings.npy``  — shape (196, 768), dtype float32
  - ``patch_grid.npy``         — shape (14, 14), int — index lookup per patch
  - ``sample_image_224.npy``   — shape (224, 224, 3), dtype uint8 — displayable
                                 sidecar (NOT the normalised CHW tensor; that
                                 would render as black in Manim — see the
                                 reviewer's note in issue #4)

Usage:
    python extract/extract_image_patches.py                       # default
    python extract/extract_image_patches.py --image path/to.jpg
    python extract/extract_image_patches.py --out-dir /tmp/glass
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from transformers import ViTImageProcessor, ViTModel

LOGGER = logging.getLogger("glassbox.extract_image_patches")

# --- Constants -------------------------------------------------------------

DEFAULT_MODEL = "google/vit-base-patch16-224"
DEFAULT_IMAGE = Path(__file__).resolve().parents[1] / "assets" / "sample_image.jpg"
DEFAULT_OUT_DIR = Path(__file__).resolve().parents[1] / "data"
IMAGE_SIZE = 224
PATCH_SIZE = 16
N_PATCHES_PER_SIDE = IMAGE_SIZE // PATCH_SIZE  # 14
N_PATCHES_TOTAL = N_PATCHES_PER_SIDE**2  # 196
HIDDEN_SIZE = 768


# --- Loader helpers --------------------------------------------------------


def _load_inputs(
    image_path: Path, model_name: str
) -> tuple[ViTImageProcessor, Image.Image]:
    """Load the processor and the raw image.

    The processor handles BOTH the 224×224 resize and the
    mean/std normalisation. We deliberately do NOT pre-resize the image
    with Pillow — that would double-resize it.
    """
    if not image_path.is_file():
        raise FileNotFoundError(f"image not found: {image_path}")
    processor = ViTImageProcessor.from_pretrained(model_name)
    image = Image.open(image_path).convert("RGB")
    return processor, image


def _encode(processor: ViTImageProcessor, image: Image.Image) -> torch.Tensor:
    """Run the processor — returns ``pixel_values`` of shape (1, 3, 224, 224)."""
    enc = processor(images=image, return_tensors="pt")
    return enc["pixel_values"]


# --- Model forward ---------------------------------------------------------


def _extract_patches(model: ViTModel, pixel_values: torch.Tensor) -> np.ndarray:
    """Forward pass + drop CLS token. Returns (196, 768) float32 numpy array."""
    # ViTModel's pooler weights are randomly initialised; we only use
    # last_hidden_state, so silence the noise.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with torch.no_grad():
            out = model(pixel_values=pixel_values, output_hidden_states=True)
    patches = out.last_hidden_state[:, 1:, :].squeeze(0)
    return patches.detach().cpu().numpy().astype(np.float32)


# --- Public entry point ---------------------------------------------------


def run(
    *,
    image_path: Path,
    out_dir: Path,
    model_name: str = DEFAULT_MODEL,
) -> int:
    """Run the full extraction pipeline. Returns 0 on success, non-zero on error."""
    out_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Loading processor: %s", model_name)
    processor, image = _load_inputs(image_path, model_name)

    LOGGER.info("Loading model: %s", model_name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ViTModel.from_pretrained(model_name).eval()

    pixel_values = _encode(processor, image)
    if pixel_values.shape != (1, 3, IMAGE_SIZE, IMAGE_SIZE):
        LOGGER.error("Unexpected pixel_values shape %s", tuple(pixel_values.shape))
        return 2

    patches = _extract_patches(model, pixel_values)
    if patches.shape != (N_PATCHES_TOTAL, HIDDEN_SIZE):
        LOGGER.error(
            "Unexpected patch shape %s, expected (%d, %d)",
            patches.shape,
            N_PATCHES_TOTAL,
            HIDDEN_SIZE,
        )
        return 3

    # Persist patches
    np.save(out_dir / "patch_embeddings.npy", patches)

    # Persist grid index lookup
    grid = np.arange(N_PATCHES_TOTAL, dtype=np.int64).reshape(
        N_PATCHES_PER_SIDE, N_PATCHES_PER_SIDE
    )
    np.save(out_dir / "patch_grid.npy", grid)

    # Persist a displayable sidecar: resized but NOT normalised, uint8 HWC.
    # The reviewer's bug fix: do NOT save the post-normalisation tensor
    # (that would be float32 in roughly [-1, 1] for this model, which
    # renders as black in Manim).
    display_image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
    display_arr = np.asarray(display_image, dtype=np.uint8)
    np.save(out_dir / "sample_image_224.npy", display_arr)

    # Stdout preview
    first_norm = float(np.linalg.norm(patches[0]))
    LOGGER.info(
        "Wrote %s (shape %s, dtype %s)",
        out_dir / "patch_embeddings.npy",
        patches.shape,
        patches.dtype,
    )
    LOGGER.info("Wrote %s (shape %s)", out_dir / "patch_grid.npy", grid.shape)
    LOGGER.info(
        "Wrote %s (shape %s, dtype %s)",
        out_dir / "sample_image_224.npy",
        display_arr.shape,
        display_arr.dtype,
    )
    print(f"patch_embeddings.npy: shape={patches.shape} dtype={patches.dtype}")
    print(f"patch_grid.npy:       shape={grid.shape}")
    print(f"sample_image_224.npy: shape={display_arr.shape} dtype={display_arr.dtype}")
    print(f"sample L2 norm of first patch embedding: {first_norm:.4f}")
    return 0


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image",
        type=Path,
        default=DEFAULT_IMAGE,
        help=f"input image (default: {DEFAULT_IMAGE})",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"output directory (default: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"HuggingFace model id (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="enable DEBUG logging",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    return run(
        image_path=args.image,
        out_dir=args.out_dir,
        model_name=args.model,
    )


if __name__ == "__main__":
    sys.exit(main())
