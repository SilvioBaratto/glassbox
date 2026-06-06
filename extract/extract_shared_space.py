"""Offline CLIP shared-space extraction for Glassbox (issue #6).

Projects both Italian text prompts and the sample image into the 512-dim
shared embedding space of ``openai/clip-vit-base-patch32``, then reduces
to 3D via ``sklearn.decomposition.PCA`` for scene 05 (3D scatter).

Outputs in ``data/``:
  - ``shared_text_embeds.npy``  — shape (N_text, 512), float32, L2-normalised
  - ``shared_image_embeds.npy`` — shape (N_img, 512), float32, L2-normalised
  - ``pca_coords_3d.npy``        — shape (N_text + N_img, 3), float32, centred
  - ``pca_labels.json``          — top-level: ``{modality, label, colors}``

The 3D scene shows 5 text dots + 1 image dot (N=6). This is a deliberately
sparse visualisation — see the reviewer's note in issue #6: a follow-up
issue can add sample-image variants for a richer cloud. The contract is
documented here so downstream scenes know what to expect.

Usage:
    python extract/extract_shared_space.py                       # default
    python extract/extract_shared_space.py --image path/to.jpg
    python extract/extract_shared_space.py --out-dir /tmp/glass
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from sklearn.decomposition import PCA
from transformers import CLIPModel, CLIPProcessor

try:
    from extract._common import parse_script_text
except ModuleNotFoundError:  # allow ``python extract/extract_shared_space.py`` direct
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parent))
    from _common import parse_script_text  # type: ignore[no-redef]

LOGGER = logging.getLogger("glassbox.extract_shared_space")

# --- Constants -------------------------------------------------------------

DEFAULT_MODEL = "openai/clip-vit-base-patch32"
DEFAULT_IMAGE = Path(__file__).resolve().parents[1] / "assets" / "sample_image.jpg"
DEFAULT_OUT_DIR = Path(__file__).resolve().parents[1] / "data"
DEFAULT_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "script.md"
EMBED_DIM = 512
PCA_COMPONENTS = 3
PCA_RANDOM_STATE = 0  # reviewer-flagged: must be 0 for determinism

# Modality color map (per requirements.md: blue=text, orange=image)
COLORS = {
    "text": "#3B82F6",  # blue
    "image": "#F97316",  # orange
}

# Minimum number of distinct prompts required by the AC.
MIN_TEXTS = 5


def _select_texts(script_path: Path) -> list[str]:
    """Pick the first N clean sentences from the script, N >= MIN_TEXTS."""
    sentences = parse_script_text(script_path)
    if len(sentences) < MIN_TEXTS:
        raise ValueError(
            f"Need at least {MIN_TEXTS} sentences in {script_path}, got {len(sentences)}"
        )
    return sentences[:MIN_TEXTS]


# --- CLIP forward ----------------------------------------------------------


def _encode(
    processor: CLIPProcessor,
    model: CLIPModel,
    texts: Sequence[str],
    image: Image.Image,
) -> tuple[np.ndarray, np.ndarray]:
    """Run CLIPModel(**inputs) and return (text_embeds, image_embeds) as numpy.

    Bug-fix for transformers 4.56: ``CLIPProcessor.__call__`` forwards
    kwargs to the fast image processor, which lacks
    ``_valid_processor_keys`` and crashes on ``padding=...``. We call the
    tokenizer and image processor explicitly with their own kwargs.
    """
    text_inputs = processor.tokenizer(  # type: ignore[attr-defined]
        list(texts),
        padding="max_length",
        truncation=True,
        max_length=77,
        return_tensors="pt",
    )
    image_inputs = processor.image_processor(images=image, return_tensors="pt")  # type: ignore[attr-defined]
    inputs = {**text_inputs, **image_inputs}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with torch.no_grad():
            out = model(**inputs)
    text = out.text_embeds.detach().cpu().numpy().astype(np.float32)
    image_arr = out.image_embeds.detach().cpu().numpy().astype(np.float32)
    return text, image_arr


# --- PCA projection (extracted for testability + reviewer's random_state=0) --


def _pca_projection(concat: np.ndarray, *, random_state: int) -> np.ndarray:
    """Reduce (N, 512) → (N, 3) via PCA, then centre on the mean.

    Extracted as its own function so tests can assert ``random_state=0`` is
    passed to sklearn without mocking the whole module.
    """
    coords = PCA(n_components=PCA_COMPONENTS, random_state=random_state).fit_transform(
        concat
    )
    coords = coords - coords.mean(axis=0, keepdims=True)
    return coords.astype(np.float32)


# --- Public entry point ---------------------------------------------------


def run(
    *,
    image_path: Path,
    out_dir: Path,
    model_name: str = DEFAULT_MODEL,
    script_path: Path = DEFAULT_SCRIPT_PATH,
) -> int:
    """Run the full extraction pipeline. Returns 0 on success, non-zero on error."""
    if not image_path.is_file():
        raise FileNotFoundError(f"image not found: {image_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    texts = _select_texts(script_path)
    image = Image.open(image_path).convert("RGB")

    LOGGER.info("Loading CLIP: %s", model_name)
    with warnings.catch_warnings():
        # transformers 4.56 emits a one-time warning about use_fast defaults;
        # we explicitly set use_fast=False to match the model card and silence
        # the noise.
        warnings.simplefilter("ignore")
        processor = CLIPProcessor.from_pretrained(model_name, use_fast=False)
        model = CLIPModel.from_pretrained(model_name).eval()

    text_embeds, image_embeds = _encode(processor, model, texts, image)

    if text_embeds.shape != (len(texts), EMBED_DIM):
        LOGGER.error("Unexpected text_embeds shape %s", text_embeds.shape)
        return 2
    if image_embeds.ndim != 2 or image_embeds.shape[1] != EMBED_DIM:
        LOGGER.error("Unexpected image_embeds shape %s", image_embeds.shape)
        return 3

    np.save(out_dir / "shared_text_embeds.npy", text_embeds)
    np.save(out_dir / "shared_image_embeds.npy", image_embeds)

    # Concat text first, then image — match the labels order below
    concat = np.concatenate([text_embeds, image_embeds], axis=0)
    pca = _pca_projection(concat, random_state=PCA_RANDOM_STATE)
    if pca.shape != (len(texts) + image_embeds.shape[0], PCA_COMPONENTS):
        LOGGER.error("Unexpected PCA shape %s", pca.shape)
        return 4
    np.save(out_dir / "pca_coords_3d.npy", pca)

    # pca_labels.json schema (reviewer's bug fix):
    #   {modality: [...], label: [...], colors: {text: "#3B82F6", image: "#F97316"}}
    labels = {
        "modality": ["text"] * len(texts) + ["image"] * image_embeds.shape[0],
        "label": list(texts) + [image_path.name],
        "colors": COLORS,
    }
    (out_dir / "pca_labels.json").write_text(
        json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Stdout preview
    LOGGER.info(
        "Wrote %s (shape %s)", out_dir / "shared_text_embeds.npy", text_embeds.shape
    )
    LOGGER.info(
        "Wrote %s (shape %s)", out_dir / "shared_image_embeds.npy", image_embeds.shape
    )
    LOGGER.info("Wrote %s (shape %s)", out_dir / "pca_coords_3d.npy", pca.shape)
    LOGGER.info("Wrote %s", out_dir / "pca_labels.json")
    print(
        f"shared_text_embeds.npy:  shape={text_embeds.shape} dtype={text_embeds.dtype}"
    )
    print(
        f"shared_image_embeds.npy: shape={image_embeds.shape} dtype={image_embeds.dtype}"
    )
    print(f"pca_coords_3d.npy:       shape={pca.shape} dtype={pca.dtype}")
    print(
        f"pca_labels.json:         {len(labels['modality'])} entries, colors={labels['colors']}"
    )
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
        "--script",
        type=Path,
        default=DEFAULT_SCRIPT_PATH,
        help=f"script.md (default: {DEFAULT_SCRIPT_PATH})",
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
        script_path=args.script,
    )


if __name__ == "__main__":
    sys.exit(main())
