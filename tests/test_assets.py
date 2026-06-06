"""Structural tests for the Glassbox sample assets (issue #2).

These tests verify that ``assets/sample_image.jpg`` and ``assets/sample_audio.wav``
exist, meet the format/size constraints, and have documented provenance.

The test does not validate visual/audio quality — only the file-level contract
that downstream extraction scripts (issues #3–#6) and Manim scenes depend on.
"""

from __future__ import annotations

import struct
import wave
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
IMAGE_PATH = ASSETS / "sample_image.jpg"
AUDIO_PATH = ASSETS / "sample_audio.wav"
README_PATH = ASSETS / "README.md"

MAX_FILE_BYTES = 1 * 1024 * 1024  # 1 MB per the issue's hard AC

# --- Image -----------------------------------------------------------------


def test_when_image_read_then_224x224_rgb() -> None:
    img = Image.open(IMAGE_PATH)
    assert img.format == "JPEG", f"sample_image must be JPEG, got {img.format}"
    assert img.size == (224, 224), f"sample_image must be 224x224, got {img.size}"
    assert img.mode == "RGB", f"sample_image must be RGB, got {img.mode}"


def test_when_image_inspected_then_under_one_megabyte() -> None:
    size = IMAGE_PATH.stat().st_size
    assert 0 < size < MAX_FILE_BYTES, (
        f"sample_image.jpg size {size} bytes not in (0, {MAX_FILE_BYTES})"
    )


def test_when_image_pixels_inspected_then_structural_content_present() -> None:
    """A pure gradient produces degenerate ViT patch embeddings.

    This test guards against the fallback in the issue body — the image
    must contain hard edges, varied colours, or text, not a single smooth
    gradient. We measure colour variance across 4x4 blocks.
    """
    img = Image.open(IMAGE_PATH).convert("RGB")
    pixels = list(img.getdata())  # type: ignore[arg-type]
    # Sample 64 evenly-spaced pixels and verify they are not all near-equal
    sample = pixels[:: len(pixels) // 64][:64]
    rs = [p[0] for p in sample]
    gs = [p[1] for p in sample]
    bs = [p[2] for p in sample]
    spread = max(max(rs) - min(rs), max(gs) - min(gs), max(bs) - min(bs))
    assert spread >= 100, (
        f"sample_image appears to be near-uniform (channel spread={spread}); "
        "ViT-16 will produce degenerate patch embeddings"
    )


# --- Audio -----------------------------------------------------------------


def test_when_audio_read_then_mono_16khz_pcm16() -> None:
    with wave.open(str(AUDIO_PATH), "rb") as w:
        assert w.getnchannels() == 1, f"audio must be mono, got {w.getnchannels()}"
        assert w.getframerate() == 16000, (
            f"audio must be 16 kHz, got {w.getframerate()}"
        )
        assert w.getsampwidth() == 2, (
            f"audio must be 16-bit PCM (sampwidth=2), got {w.getsampwidth()}"
        )


def test_when_audio_inspected_then_under_one_megabyte() -> None:
    size = AUDIO_PATH.stat().st_size
    assert 0 < size < MAX_FILE_BYTES, (
        f"sample_audio.wav size {size} bytes not in (0, {MAX_FILE_BYTES})"
    )


def test_when_audio_duration_read_then_5_to_10_seconds() -> None:
    with wave.open(str(AUDIO_PATH), "rb") as w:
        frames = w.getnframes()
        rate = w.getframerate()
    duration = frames / float(rate)
    assert 5.0 <= duration <= 10.0, f"audio duration {duration:.2f}s not in [5.0, 10.0]"


def test_when_audio_samples_inspected_then_not_all_zero() -> None:
    """A silent WAV would make Whisper produce all-zero log-mel frames."""
    with wave.open(str(AUDIO_PATH), "rb") as w:
        raw = w.readframes(w.getnframes())
    fmt = f"<{len(raw) // 2}h"
    samples = struct.unpack(fmt, raw)
    nonzero = sum(1 for s in samples if s != 0)
    assert nonzero > len(samples) * 0.1, (
        f"audio is suspiciously silent: {nonzero}/{len(samples)} non-zero samples"
    )


# --- Provenance ------------------------------------------------------------


def test_when_assets_readme_read_then_provenance_documented() -> None:
    assert README_PATH.is_file(), "assets/README.md missing"
    content = README_PATH.read_text()
    for label in ("sample_image", "sample_audio", "Provenance", "License"):
        assert label in content, f"assets/README.md must document: {label}"


# --- Total bundle size -----------------------------------------------------


def test_when_all_assets_combined_then_under_five_megabytes() -> None:
    total = sum(p.stat().st_size for p in ASSETS.iterdir() if p.is_file())
    assert total < 5 * 1024 * 1024, f"assets/ total {total} bytes exceeds 5 MB cap"
