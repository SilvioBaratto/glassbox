"""Generate deterministic sample assets for Glassbox (issue #2).

Run once to (re)create ``assets/sample_image.jpg`` and
``assets/sample_audio.wav``. Both outputs are deterministic given a fixed seed.

Why programmatic, not "a real cat photo":
  - No network dependency at build time.
  - The extraction pipeline (issue #4) needs *content* (edges, varied colours)
    to produce meaningful ViT patch embeddings; a real photograph would be
    better aesthetically but a structured synthetic image gives us full
    provenance and reproducibility.
  - License is then unambiguously MIT (this project) rather than "some
    random public-domain photo" with unclear attribution.

Usage:
    python scripts/generate_assets.py
"""

from __future__ import annotations

import argparse
import sys
import wave
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ASSETS_DIR = Path(__file__).resolve().parents[1] / "assets"
IMAGE_PATH = ASSETS_DIR / "sample_image.jpg"
AUDIO_PATH = ASSETS_DIR / "sample_audio.wav"

# --- Image -----------------------------------------------------------------

IMAGE_SIZE = 224
RNG_SEED = 20260606


def _draw_background_sky(draw: ImageDraw.ImageDraw) -> None:
    """Top-to-bottom sky gradient (blue → pale yellow at horizon)."""
    top = np.array([70, 130, 220], dtype=np.float32)  # steel blue
    bottom = np.array([255, 235, 180], dtype=np.float32)  # warm horizon
    for y in range(IMAGE_SIZE):
        t = y / (IMAGE_SIZE - 1)
        colour = tuple(int(round(c)) for c in (1 - t) * top + t * bottom)
        draw.line([(0, y), (IMAGE_SIZE, y)], fill=colour)


def _draw_sun(img: Image.Image) -> None:
    """Solid yellow disc in the upper-right, with a soft halo."""
    draw = ImageDraw.Draw(img)
    cx, cy, r = IMAGE_SIZE - 50, 50, 28
    for ring, alpha in ((40, 30), (32, 60), (26, 120)):
        draw.ellipse(
            (cx - ring, cy - ring, cx + ring, cy + ring),
            fill=(255, 230, 120, alpha),
        )
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(255, 215, 0))


def _draw_mountains(img: Image.Image) -> None:
    """Two overlapping triangles forming a mountain range silhouette."""
    draw = ImageDraw.Draw(img)
    horizon = 140
    # Far mountain (lighter, behind)
    draw.polygon(
        [(0, horizon), (60, 70), (130, horizon)],
        fill=(110, 130, 150),
    )
    # Near mountain (darker, in front)
    draw.polygon(
        [(80, horizon), (170, 50), (224, horizon)],
        fill=(70, 90, 110),
    )
    # Snow caps (small white triangles)
    draw.polygon([(54, 78), (60, 70), (66, 78)], fill=(245, 245, 250))
    draw.polygon([(162, 60), (170, 50), (178, 60)], fill=(245, 245, 250))


def _draw_ground(img: Image.Image, rng: np.random.Generator) -> None:
    """Green ground strip with a darker grass texture stripe."""
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 140, IMAGE_SIZE, IMAGE_SIZE), fill=(60, 140, 70))
    # Diagonal grass strokes (deterministic pattern)
    for _ in range(60):
        x = int(rng.integers(0, IMAGE_SIZE))
        y = int(rng.integers(140, IMAGE_SIZE - 2))
        draw.line([(x, y), (x + 2, y + 4)], fill=(40, 110, 50))


def _draw_cat_silhouette(img: Image.Image) -> None:
    """Stylised cat silhouette — hard black edges, no gradient.

    The hard black contour guarantees strong edges that ViT-16 will
    pick up as salient patches (the whole point of issue #4's
    visualisation).
    """
    draw = ImageDraw.Draw(img)
    base_x, base_y = 80, 180
    # Body
    draw.ellipse(
        (base_x - 28, base_y - 18, base_x + 28, base_y + 14), fill=(20, 20, 25)
    )
    # Head
    draw.ellipse((base_x + 18, base_y - 32, base_x + 46, base_y - 8), fill=(20, 20, 25))
    # Ears (triangles)
    draw.polygon(
        [
            (base_x + 22, base_y - 28),
            (base_x + 28, base_y - 40),
            (base_x + 34, base_y - 28),
        ],
        fill=(20, 20, 25),
    )
    draw.polygon(
        [
            (base_x + 36, base_y - 28),
            (base_x + 42, base_y - 40),
            (base_x + 48, base_y - 28),
        ],
        fill=(20, 20, 25),
    )
    # Tail
    draw.polygon(
        [
            (base_x - 28, base_y - 8),
            (base_x - 50, base_y - 30),
            (base_x - 38, base_y - 2),
        ],
        fill=(20, 20, 25),
    )
    # Eyes (two yellow dots)
    draw.ellipse(
        (base_x + 26, base_y - 24, base_x + 30, base_y - 20), fill=(255, 220, 0)
    )
    draw.ellipse(
        (base_x + 34, base_y - 24, base_x + 38, base_y - 20), fill=(255, 220, 0)
    )


def _draw_caption(img: Image.Image) -> None:
    """Small Italian caption — gives the image OCR-like text features."""
    draw = ImageDraw.Draw(img)
    try:
        # Use default PIL font (no external font file required)
        font = ImageFont.load_default(size=14)
    except TypeError:
        font = ImageFont.load_default()
    draw.text((6, IMAGE_SIZE - 22), "gatto", fill=(20, 20, 20), font=font)


def generate_image(out_path: Path, *, seed: int = RNG_SEED, quality: int = 80) -> None:
    """Render the structured sample image and save as JPEG."""
    img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), color=(0, 0, 0))
    draw = ImageDraw.Draw(img)
    rng = np.random.default_rng(seed)
    _draw_background_sky(draw)
    _draw_sun(img)
    _draw_mountains(img)
    _draw_ground(img, rng)
    _draw_cat_silhouette(img)
    _draw_caption(img)
    # JPEG quality 80 keeps the file well under 1 MB while preserving edges.
    img.save(out_path, format="JPEG", quality=quality, optimize=True)


# --- Audio -----------------------------------------------------------------

AUDIO_SR = 16_000
AUDIO_DURATION = 5.0  # seconds


def _synthesise_audio(rng: np.random.Generator) -> np.ndarray:
    """Build a 5s mono signal that *resembles* speech-like spectral content.

    Structure:
      - 4 formants (vowel-like harmonics): 440, 1200, 2400, 3500 Hz
      - Amplitude-modulated by a 4 Hz envelope (syllable rhythm)
      - Whisper-shaped low-pass tilt to keep the high frequencies subtle
      - Additive pink-ish noise at -30 dB so the file isn't a pure tone
        (a pure 440 Hz tone would still meet the AC but would not exercise
        Whisper's frame extraction well)
    """
    n = int(AUDIO_SR * AUDIO_DURATION)
    t = np.arange(n) / AUDIO_SR

    # Envelope: 4 Hz syllabic gate, smoothed, with random syllable onsets
    syllable = 0.5 + 0.5 * np.sin(2 * np.pi * 4.0 * t + rng.uniform(0, 2 * np.pi))
    syllable = np.clip(syllable, 0, 1) ** 1.5

    signal = np.zeros(n, dtype=np.float32)
    for f in (440.0, 1200.0, 2400.0, 3500.0):
        amp = 0.15 if f < 1000 else 0.08
        signal += amp * np.sin(2 * np.pi * f * t)

    # Pink-ish noise (1/f) — sum of octave-spaced random sines
    noise = np.zeros(n, dtype=np.float32)
    for f0 in (100, 200, 400, 800, 1600, 3200):
        phase = rng.uniform(0, 2 * np.pi)
        noise += (0.01 / (1 + f0 / 1000)) * np.sin(2 * np.pi * f0 * t + phase)

    signal = signal * syllable + noise
    # Bounded amplitude — must fit PCM_16 with headroom (≤ 0.5)
    signal = signal * 0.4
    peak = float(np.abs(signal).max())
    if peak > 0.5:
        signal = signal * (0.5 / peak)
    return signal.astype(np.float32)


def _write_wav_pcm16(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    """Write mono float32 audio as 16-bit PCM WAV.

    ``scipy.io.wavfile.write(..., subtype='PCM_16')`` would be the obvious
    choice, but we use the stdlib ``wave`` module to avoid the dependency
    on ``soundfile`` (not always available on macOS conda envs).
    """
    # Clip then convert to int16
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)  # 16-bit
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())


def generate_audio(out_path: Path, *, seed: int = RNG_SEED) -> None:
    rng = np.random.default_rng(seed)
    samples = _synthesise_audio(rng)
    _write_wav_pcm16(out_path, samples, AUDIO_SR)


# --- Entry point -----------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=RNG_SEED, help="deterministic seed")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ASSETS_DIR,
        help="output directory",
    )
    args = parser.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    generate_image(args.out_dir / "sample_image.jpg", seed=args.seed)
    generate_audio(args.out_dir / "sample_audio.wav", seed=args.seed)
    print(f"Generated: {args.out_dir / 'sample_image.jpg'}")
    print(f"Generated: {args.out_dir / 'sample_audio.wav'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
